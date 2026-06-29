import json
import ast
import pandas as pd
import os

def parse_match(match_val):
    """
    Parses the 'match' field which can be a string representation of a list or a list.
    Returns a list of indicator IDs found in the match.
    """
    try:
        if isinstance(match_val, str):
            match_data = ast.literal_eval(match_val)
        else:
            match_data = match_val
            
        # match_data is typically a list of dicts or list of lists of dicts
        # Flatten and extract 'Indicator_ID'
        indicators = []
        
        def extract_from_list(lst):
            for item in lst:
                if isinstance(item, list):
                    extract_from_list(item)
                elif isinstance(item, dict):
                    if 'Indicator_ID' in item:
                        indicators.append(item['Indicator_ID'])
        
        if isinstance(match_data, list):
            extract_from_list(match_data)
        elif isinstance(match_data, dict):
             if 'Indicator_ID' in match_data:
                indicators.append(match_data['Indicator_ID'])
                
        return indicators
    except Exception as e:
        return []

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming the script is in analysis/, data is in ../data
    data_path = os.path.join(base_dir, "..", "data", "clean_data_3", "interventions.json")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_json(data_path)
    
    # Previous utterances
    prev_ids = [924, 1018, 1059]
    
    print(f"\n--- Analysis of Previous Utterances ---")
    used_indicators = set()
    
    for uid in prev_ids:
        row = df[df['matched_at_utterance_id'] == uid]
        if not row.empty:
            match_val = row.iloc[0]['match']
            inds = parse_match(match_val)
            print(f"Utterance {uid}: Indicators {inds}")
            used_indicators.update(inds)
        else:
            print(f"Utterance {uid}: Not found in file")
            
    print(f"\nused_indicators: {used_indicators}")

    print(f"\n--- Finding 7 New Utterances ---")
    
    # Strategy: Find utterances that have valid sequences (possible_sequence_ids)
    # and preferably feature indicators NOT in the used set, or just distinct combinations.
    
    def has_sequences(val):
        if isinstance(val, list):
            return len(val) >= 1
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return isinstance(parsed, list) and len(parsed) >= 1
            except:
                return False
        return False
        
    candidates = df[df['possible_sequence_ids'].apply(has_sequences)].copy()
    
    # Exclude existing
    candidates = candidates[~candidates['matched_at_utterance_id'].isin(prev_ids)]
    
    print(f"Total valid candidates: {len(candidates)}")
    
    new_picks = []
    
    import random
    
    # Try to pick ones with new indicators first
    # For simplicity, let's just pick 7 distinct ones randomly from valid candidates
    
    candidates_list = []
    for index, row in candidates.iterrows():
        uid = row['matched_at_utterance_id']
        inds = parse_match(row['match'])
        candidates_list.append({
            "id": int(uid),
            "indicators": inds
        })
        
    random.seed(42) # For reproducibility
    if len(candidates_list) >= 7:
        new_picks = random.sample(candidates_list, 7)
    else:
        new_picks = candidates_list
    
    print("\nSelected 7 Random Candidates:")
    for pick in new_picks:
        print(f"Utterance {pick['id']}: {pick['indicators']}")

if __name__ == "__main__":
    main()
