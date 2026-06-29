import pandas as pd
import ast
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "..", "data", "clean_data_3", "interventions.json")
    
    print(f"Loading data from {data_path}...")
    df = pd.read_json(data_path)
    
    # Already configured utterances
    configured_ids = {
        924, 1018, 1059, # Original
        1027, 955, 926, 1074, 985, 968, 1061 # New 7
    }
    
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
        
    valid_utterances = df[df['possible_sequence_ids'].apply(has_sequences)]['matched_at_utterance_id'].unique()
    valid_ids = {int(uid) for uid in valid_utterances}
    
    remaining_ids = sorted(list(valid_ids - configured_ids))
    
    print(f"Total valid utterances: {len(valid_ids)}")
    print(f"Configured utterances: {len(configured_ids)}")
    print(f"Remaining utterances ({len(remaining_ids)}): {remaining_ids}")

if __name__ == "__main__":
    main()
