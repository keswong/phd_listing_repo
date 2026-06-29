import argparse
import json
import os
import subprocess
import datetime
import concurrent.futures
import threading
from tqdm import tqdm

"""
Script to generate batch interventions by running the multi-agent pipeline repeatedly.
Includes support for a flattened output format compatible with Kester Colab notebooks via the --kester-colab-format flag.

eg. `python analysis/generate_interventions.py --utterance_id 924 --iterations 2 --temperature 0.0 --model gpt-5-mini --kester-colab-format`
"""

def run_feedback_generation(utterance_id, include_pedagogy, temperature, run_number, model="o4-mini", log_file=None, log_lock=None):
    """
    Runs generate_feedback.py and returns the parsed JSON output.
    
    Args:
        utterance_id (int): The ID of the utterance to process.
        include_pedagogy (bool): Whether to include the pedagogy agent.
        temperature (float): Sampling temperature for the LLM.
        run_number (int): The current iteration number (for logging).
        model (str): The model to use for generation (default: o4-mini).
        log_file (str): Path to a log file to write verbose output to. If provided, enables verbose mode.
        log_lock (threading.Lock): Optional lock for thread-safe logging.
        
    Returns:
        dict: The parsed JSON output from generate_feedback.py, or None if failed.
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
    
    if log_file:
        cmd.append("--verbose")
    
    if not include_pedagogy:
        cmd.append("--no-pedagogy")
        
    try:
        # Run command and capture stdout
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if log_file:
            # Use lock if provided, strictly necessary for parallel runs
            lock_context = log_lock if log_lock else contextlib.nullcontext() 
            # Note: contextlib.nullcontext would require import contextlib. 
            # Simpler: just check inside if.
            
            if log_lock:
                with log_lock:
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(f"\n{'='*30} Run {run_number} {'='*30}\n")
                        f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                        f.write(result.stdout)
                        f.write(f"\n{'-'*80}\n")
            else:
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*30} Run {run_number} {'='*30}\n")
                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                    f.write(result.stdout)
                    f.write(f"\n{'-'*80}\n")

        # Parse the last line of stdout which should be the JSON
        # (generate_feedback.py prints verbose logs to stdout too if requested, but we aren't requesting verbose)
        # However, generate_feedback.py prints the JSON at the end.
        output_lines = result.stdout.strip().split('\n')
        json_output = output_lines[-1] # Optimistically take the last line
        
        return json.loads(json_output)
    except subprocess.CalledProcessError as e:
        # Print error immediately to console (thread-safe enough for print usually, or uses global lock)
        print(f"Error running generation for run {run_number}: {e}")
        # print(f"Stdout: {e.stdout}") # Too verbose for concurrent
        # print(f"Stderr: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for run {run_number}: {e}")
        # print(f"Output was: {output_lines[-1] if 'output_lines' in locals() else 'No output'}")
        return None

def main():
    """
    Main function to run the batch generation process.
    Iterates through conditions (with/without pedagogical agent) and generates the specified number of runs.
    Outputs data to JSON files.
    """
    parser = argparse.ArgumentParser(description="Generate batch interventions for analysis.")
    parser.add_argument('--utterance_id', type=int, default=924, help='Utterance ID to process')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations per condition')
    parser.add_argument('--temperature', type=float, default=0.0, help='Sampling temperature')
    parser.add_argument('--kester-colab-format', action='store_true', help='Output in Kester Colab flattened format')
    parser.add_argument('--model', type=str, default='o4-mini', help='Model to use for generation (default: o4-mini)')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save output files (default: current directory)')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output from generation steps')
    parser.add_argument('--max-workers', type=int, default=20, help='Max parallel workers (default: 20)')
    
    parser.add_argument('--condition', type=str, choices=['all', 'pedagogical_agent', 'no_pedagogical_agent'], default='all', help='Condition to run (default: all)')
    
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().isoformat()
    log_file = None
    log_lock = None
    
    if args.verbose:
        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Create a log file with timestamp
        sanitized_timestamp = timestamp.replace(":", "-").replace(".", "-")
        log_file = os.path.join(logs_dir, f"generation_log_{sanitized_timestamp}.txt")
        print(f"Verbose logging enabled. Logs will be saved to: {log_file}")
        log_lock = threading.Lock()
    
    # Define conditions
    all_conditions = [
        {"name": "pedagogical_agent", "include_pedagogy": True},
        {"name": "no_pedagogical_agent", "include_pedagogy": False}
    ]
    
    if args.condition == 'all':
        conditions = all_conditions
    else:
        conditions = [c for c in all_conditions if c['name'] == args.condition]
    
    for condition in conditions:
        print(f"\nGenerating {args.iterations} runs for condition: {condition['name']}...")
        
        runs_results = {}
        
        # Use ThreadPoolExecutor for parallel execution
        # Max workers limited to avoid overwhelming system with subprocesses
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_run_id = {
                executor.submit(
                    run_feedback_generation, 
                    args.utterance_id, 
                    condition["include_pedagogy"], 
                    args.temperature,
                    i,
                    model=args.model,
                    log_file=log_file,
                    log_lock=log_lock
                ): i for i in range(1, args.iterations + 1)
            }
            
            for future in tqdm(concurrent.futures.as_completed(future_to_run_id), total=len(future_to_run_id)):
                run_num = future_to_run_id[future]
                try:
                    res = future.result()
                    runs_results[run_num] = res
                except Exception as exc:
                    print(f"Run {run_num} generated an exception: {exc}")
                    runs_results[run_num] = None

        # Process results in order
        runs_data = []
        for i in range(1, args.iterations + 1):
            result = runs_results.get(i)
            
            if result:
                # Structure the run data cleanly
                # We need to extract the standard fields
                run_entry = {
                    "run_number": i,
                    "context_agent": {
                        "prompt": result['context']['prompt'],
                        "description": result['context']['description']
                    },
                    "recommendation_agent": result['recommendation'],
                    #"final_output": result.get('feedback') # Removed as requested
                }

                # Add pedagogy agent if present
                if result.get('pedagogy'):
                    run_entry["pedagogy_agent"] = {
                        "prompt": result['pedagogy']['prompt'],
                        "description": result['pedagogy']['description']
                    }
                else:
                     run_entry["pedagogy_agent"] = None

                runs_data.append(run_entry)
        
        # Turn into final format
        if args.kester_colab_format:
            colab_runs = {}
            for run in runs_data:
                idx = run["run_number"]
                prefix = f"markdown_run_{idx}_"
                
                # Context Agent
                colab_runs[f"{prefix}contextprompt"] = run["context_agent"]["prompt"]
                colab_runs[f"{prefix}contextdescription"] = run["context_agent"]["description"]
                
                # Pedagogy Agent
                if run["pedagogy_agent"]:
                    colab_runs[f"{prefix}pedagogyprompt"] = run["pedagogy_agent"]["prompt"]
                    colab_runs[f"{prefix}pedagogydescription"] = run["pedagogy_agent"]["description"]
                
                # Recommendation Agent
                rec = run["recommendation_agent"]
                if "prompt" in rec:
                    colab_runs[f"{prefix}recommendationprompt"] = rec["prompt"]
                
                if "reasoning" in rec:
                    colab_runs[f"{prefix}feedbackreasoning"] = rec["reasoning"]
                
                if "feedback" in rec:
                    colab_runs[f"{prefix}feedbackfeedback"] = rec["feedback"]


            
            final_output = {
                "utterance_id": args.utterance_id,
                "model": args.model,
                "temperature": args.temperature,
                "iterations": args.iterations,
                "timestamp": timestamp,
                "intervention_type": condition["name"],
                "runs": colab_runs
            }
        else:
            final_output = {
                "metadata": {
                    "utterance_id": args.utterance_id,
                    "timestamp": timestamp,
                    "intervention_type": condition["name"],
                    "temperature": args.temperature
                },
                "runs": runs_data
            }
        
        
        # Save to file
        if args.output_dir:
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            sanitized_filename_timestamp = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
            filename = os.path.join(output_dir, f"interventions_{args.utterance_id}_{args.model}_{condition['name']}_{sanitized_filename_timestamp}.json")
        else:
            sanitized_filename_timestamp = datetime.datetime.now().isoformat().replace(":", "-").replace(".", "-")
            filename = f"interventions_{condition['name']}_{args.utterance_id}_{sanitized_filename_timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
        print(f"Saved {len(runs_data)} runs to {filename}")

if __name__ == "__main__":
    main()
