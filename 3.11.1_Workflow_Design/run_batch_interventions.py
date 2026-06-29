import argparse
import json
import os
import subprocess
import datetime
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import ast

"""
Script to generate batch interventions for ALL utterances of interest found in data/clean_data_2/interventions.json.
Filters for utterances that have at least 1 possible sequence ID.
Generates 1 intervention with pedagogy agent and 1 without for each utterance.
"""

def run_feedback_generation(utterance_id, include_pedagogy, temperature, model="o4-mini", verbose=False):
    """
    Runs generate_feedback.py and returns the parsed JSON output and the raw stdout log.
    """
    # Determine path to generate_feedback.py relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    generate_feedback_path = os.path.join(root_dir, "generate_feedback.py")
    
    cmd = [
        "python",
        generate_feedback_path,
        "--utterance_id", str(utterance_id),
        "--output-format", "json",
        "--temperature", str(temperature),
        "--model", str(model)
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    if not include_pedagogy:
        cmd.append("--no-pedagogy")
        
    try:
        # Run command and capture stdout
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse the last line of stdout which should be the JSON
        output_lines = result.stdout.strip().split('\n')
        if not output_lines:
            return None, result.stdout
            
        json_output = output_lines[-1] 
        return json.loads(json_output), result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running generation for utterance {utterance_id}: {e}")
        return None, e.stdout if e.stdout else str(e)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for utterance {utterance_id}: {e}")
        return None, result.stdout if 'result' in locals() else str(e)

def process_utterance(utterance_id, temperature, model, verbose, output_dir):
    """
    Generates both pedagogy and no-pedagogy interventions for a single utterance
    and logs detailed output to files.
    """
    
    # Run with Pedagogy
    pedagogy_result, pedagogy_log = run_feedback_generation(
        utterance_id, 
        include_pedagogy=True, 
        temperature=temperature, 
        model=model,
        verbose=verbose
    )
    
    # Run without Pedagogy
    no_pedagogy_result, no_pedagogy_log = run_feedback_generation(
        utterance_id, 
        include_pedagogy=False, 
        temperature=temperature, 
        model=model,
        verbose=verbose
    )

    # Save logs
    if output_dir:
        pedagogy_log_path = os.path.join(output_dir, f"utterance_{utterance_id}_pedagogy.log")
        with open(pedagogy_log_path, 'w', encoding='utf-8') as f:
            f.write(pedagogy_log)
            
        no_pedagogy_log_path = os.path.join(output_dir, f"utterance_{utterance_id}_no_pedagogy.log")
        with open(no_pedagogy_log_path, 'w', encoding='utf-8') as f:
            f.write(no_pedagogy_log)
    
    return {
        "utterance_id": utterance_id,
        "with_pedagogy": pedagogy_result,
        "without_pedagogy": no_pedagogy_result
    }

def main():
    parser = argparse.ArgumentParser(description="Generate batch interventions for all valid utterances.")
    parser.add_argument('--model', type=str, default='o4-mini', help='Model to use for generation (default: o4-mini)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--max-workers', type=int, default=10, help='Max parallel workers')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of utterances to process (for testing)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output', default=True)
    
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    interventions_path = os.path.join(root_dir, "data", "clean_data_3", "interventions.json")
    
    # Load data
    print(f"Loading interventions from {interventions_path}...")
    try:
        df_interventions = pd.read_json(interventions_path, orient="records")
    except ValueError as e:
        print(f"Error loading interventions.json: {e}")
        return

    # Filter data
    # We need to filter for rows where possible_sequence_ids has length >= 1
    # Note: pandas might load lists as lists, but sometimes as strings if malformed. 
    # Based on pedagogy_agent.py, it seems safe to assume they might be strings or lists.
    
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

    filtered_df = df_interventions[df_interventions['possible_sequence_ids'].apply(has_sequences)]
    unique_utterance_ids = filtered_df['matched_at_utterance_id'].unique()
    # Convert numpy int64 to native int
    unique_utterance_ids = [int(uid) for uid in unique_utterance_ids]
    
    if args.limit:
        unique_utterance_ids = unique_utterance_ids[:args.limit]
        print(f"Limiting to first {args.limit} utterances.")
        
    print(f"Found {len(unique_utterance_ids)} unique utterances with valid sequences.")
    
    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(script_dir, "batch_runs", f"run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}")
    
    # Run generation
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(process_utterance, uid, args.temperature, args.model, args.verbose, output_dir): uid 
            for uid in unique_utterance_ids
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id)):
            uid = future_to_id[future]
            try:
                res = future.result()
                
                # Save individual result immediately
                filename = os.path.join(output_dir, f"utterance_{uid}.json")
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                    
                results.append(res)
            except Exception as exc:
                print(f"Utterance {uid} generated an exception: {exc}")

    print(f"Completed. Generated {len(results)} intervention sets.")

if __name__ == "__main__":
    main()
