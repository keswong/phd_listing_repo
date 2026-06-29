import ast
from .base import Agent

class PedagogyAgent(Agent):
    """Agent B: Pedagogy Agent"""

    def run(self, utterance_id: int) -> dict:
        """
        Generate pedagogical description from sequences and indicators.
        
        Returns:
            dict with keys: 'prompt', 'description'
        """
        data = self._load_data()
        
        # Find intervention by matched_at_utterance_id
        intervention_rows = data['df_interventions'][
            data['df_interventions']['matched_at_utterance_id'] == utterance_id
        ]
        
        if intervention_rows.empty:
            raise ValueError(f"No intervention found for utterance_id {utterance_id}")
            
        intervention = intervention_rows.iloc[0]
        
        # Parse the match field
        match_str = intervention["match"]
        match = []
        if isinstance(match_str, str):
            try:
                parsed_match = ast.literal_eval(match_str)
                if parsed_match:
                     if isinstance(parsed_match[0], list):
                         match = parsed_match
                     else:
                         match = [parsed_match]
            except (ValueError, SyntaxError):
                match = []
        else:
            if isinstance(match_str, list):
                 if match_str and isinstance(match_str[0], list):
                     match = match_str
                 elif match_str:
                     match = [match_str]
                 else:
                     match = []
            else:
                 match = []
        
        # Utterance lines
        # Helper to format match items
        def format_match_items(match_items):
            lines = []
            for i, match_item in enumerate(match_items, 1):
                utterance_id_match = match_item.get('Utterance_ID')
                indicator_id_match = match_item.get('Indicator_ID')
                
                if utterance_id_match and indicator_id_match:
                    utt_row = data['df_utterances'][data['df_utterances']['id'] == utterance_id_match]
                    ind_row = data['df_indicators'][data['df_indicators']['id'] == indicator_id_match]
                    if not utt_row.empty and not ind_row.empty:
                        lines.append(
                            f"{i}. Indicator: {ind_row.iloc[0]['description']}\n   Utterance: {utt_row.iloc[0]['utterance']}"
                        )
            return "\n".join(lines)

        # Sequence paths
        # Collect all potential paths first
        positive_paths = []
        negative_paths = []
        
        # Get possible_sequence_ids
        possible_sequence_ids = intervention.get("possible_sequence_ids", [])
        if isinstance(possible_sequence_ids, str):
            try:
                possible_sequence_ids = ast.literal_eval(possible_sequence_ids)
            except (ValueError, SyntaxError):
                possible_sequence_ids = []
        elif not isinstance(possible_sequence_ids, list):
             # Handle single int or other types if necessary, though list expectation is standard from cleaning
             possible_sequence_ids = [possible_sequence_ids] if possible_sequence_ids is not None else []
        
        for seq_id in possible_sequence_ids:
            seq_row = data['df_sequences'][data['df_sequences']['id'] == seq_id]
            if not seq_row.empty:
                seq = seq_row.iloc[0]['sequence']
                log_odds = seq_row.iloc[0]['log_odds']
                if log_odds > 0:
                    positive_paths.append({'seq': seq, 'log_odds': log_odds, 'id': seq_id})
                else:
                    negative_paths.append({'seq': seq, 'log_odds': log_odds, 'id': seq_id})
        
        # Select best positive and worst negative
        selected_paths = []
        
        if positive_paths:
            max_odds = max(p['log_odds'] for p in positive_paths)
            best_positives = [p for p in positive_paths if p['log_odds'] == max_odds]
            for p in best_positives:
                p['effect'] = 'positive'
                selected_paths.append(p)
            
        if negative_paths:
            min_odds = min(p['log_odds'] for p in negative_paths)
            worst_negatives = [p for p in negative_paths if p['log_odds'] == min_odds]
            for p in worst_negatives:
                p['effect'] = 'negative'
                selected_paths.append(p)
            
        # Format the selected paths as simple behavior descriptions AND find context
        positive_behaviors_list = []
        positive_context_lines = None
        negative_behaviors_list = []
        negative_context_lines = None
        
        subsequence_ids = intervention.get("subsequence_ids", [])
        # Ensure subsequence_ids is a list of ints
        if isinstance(subsequence_ids, str):
             try:
                 subsequence_ids = ast.literal_eval(subsequence_ids)
             except:
                 subsequence_ids = []
        elif isinstance(subsequence_ids, (int, float)):
             subsequence_ids = [subsequence_ids]
             
        
        for path_data in selected_paths:
            seq = path_data['seq']
            effect = path_data['effect']
            seq_id = path_data['id']
            
            # Find the context (subsequence) that led to this sequence
            # We match based on whether the sequence starts with the subsequence content
            
            context_match_items = []
            found_subsequence_indicators = None
            
            # Step 1: Find the subsequence that is a prefix of the selected sequence
            for idx, sub_id in enumerate(subsequence_ids):
                sub_row = data['df_subsequences'][data['df_subsequences']['id'] == sub_id]
                if not sub_row.empty:
                    sub_content = sub_row.iloc[0]['subsequence']
                    # sub_content is a list of indicator IDs e.g. ['PS06', 'PS04']
                    
                    if len(seq) >= len(sub_content):
                        if seq[:len(sub_content)] == sub_content:
                            found_subsequence_indicators = sub_content
                            break
            
            # Step 2: Find the match item that corresponds to this subsequence
            if found_subsequence_indicators and match:
                # Iterate through match items and find the one that produces the same sequence of indicators
                for m_item in match:
                    # m_item is a list of dicts or a single list if it was parsed differently, 
                    # but logic above handles standard list-of-lists intervention format
                    current_match_list = m_item if isinstance(m_item, list) else [m_item]
                    
                    # Extract indicators from this match item
                    current_indicators = []
                    for ev in current_match_list:
                        if 'Indicator_ID' in ev:
                            current_indicators.append(ev['Indicator_ID'])
                    
                    if current_indicators == found_subsequence_indicators:
                        context_match_items = current_match_list
                        break
            
            # Format context
            context_str = format_match_items(context_match_items) if context_match_items else "No specific context identified."

            # Calculate behaviors (future recommendations)
            matched_len = len(context_match_items)
            next_indicators = seq[matched_len:] if seq and matched_len < len(seq) else []
            
            behavior_descs = []
            for next_indicator in next_indicators:
                ind_row = data['df_indicators'][data['df_indicators']['id'] == next_indicator]
                ind_desc = ind_row.iloc[0]['description'] if not ind_row.empty else next_indicator
                behavior_descs.append(ind_desc)
            
            if behavior_descs:
                behavior_str = "; ".join(behavior_descs)
            else:
                behavior_str = "Continue current strategy" # Fallback

            if effect == 'positive':
                positive_behaviors_list.append(behavior_str)
                # Use the first valid context found, assuming consistent context for same starting point
                if positive_context_lines is None:
                    positive_context_lines = context_str
            else:
                negative_behaviors_list.append(behavior_str)
                if negative_context_lines is None:
                    negative_context_lines = context_str

        # Join behaviors with OR
        positive_behaviors = " **OR** ".join(positive_behaviors_list) if positive_behaviors_list else None
        negative_behaviors = " **OR** ".join(negative_behaviors_list) if negative_behaviors_list else None
        
        # Load prompt template from file
        prompt_template = self._load_prompt("pedagogy_agent")
        from jinja2 import Template
        template = Template(prompt_template)
        prompt = template.render(
            positive_behaviors=positive_behaviors,
            positive_context_lines=positive_context_lines,
            negative_behaviors=negative_behaviors,
            negative_context_lines=negative_context_lines
        )
        
        self.log_verbose("AGENT B: PEDAGOGY AGENT\nPROMPT:", prompt)
        
        description = self._generate_llm_text(prompt)
        
        self.log_verbose("RESULT:", description)
        
        return {
            'prompt': prompt,
            'description': description
        }
