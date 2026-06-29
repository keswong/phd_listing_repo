from typing import Optional
from .base import Agent

class ContextAgent(Agent):
    """Agent A: Context Agent"""

    def run(self, utterance_id: int, task: Optional[str] = None, task_solution: Optional[str] = None) -> dict:
        """
        Generate context description from transcript.
        
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
        
        matched_at_utterance_id = intervention["matched_at_utterance_id"]
        df_utterances_cut = data['df_utterances'][
            data['df_utterances']["id"] <= matched_at_utterance_id
        ]
        
        transcript = "\n".join(
            f"Student {row['student_id']}: {row['utterance']}"
            for _, row in df_utterances_cut.iterrows()
        )
        
        # Use provided task/task_solution or fallback to cached defaults
        task_text = task if task is not None else data['task']
        task_solution_text = task_solution if task_solution is not None else data['task_solution']
        
        # Load prompt template from file
        prompt_template = self._load_prompt("context_agent")
        from jinja2 import Template
        template = Template(prompt_template)
        prompt = template.render(
            student_id=intervention['student_id'],
            task_text=task_text,
            task_solution_text=task_solution_text,
            transcript=transcript
        )
        
        self.log_verbose("AGENT A: CONTEXT AGENT\nPROMPT:", prompt)
        
        description = self._generate_llm_text(prompt)
        
        self.log_verbose("RESULT:", description)
        
        return {
            'prompt': prompt,
            'description': description
        }
