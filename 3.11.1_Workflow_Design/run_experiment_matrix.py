import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

def run_single_experiment(utterance, config, iterations, output_dir):
    model = config["model"]
    temp = config["temperature"]
    print(f"Starting: Utterance {utterance} | Model {model} | Temp {temp}")
    
    cmd = [
        sys.executable, "analysis/generate_interventions.py",
        "--utterance_id", str(utterance),
        "--iterations", str(iterations),
        "--temperature", str(temp),
        "--model", model,
        "--kester-colab-format",
        "--output-dir", output_dir,
        "--max-workers", "10"  # Reduced from 30 to 10 to be safer
    ]
    
    try:
        # Check=True ensures we catch errors, but we want to capture output to avoid console spam mixing
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Finished: Utterance {utterance} | Model {model}")
    except subprocess.CalledProcessError as e:
        print(f"Error in {utterance} {model}: {e.stderr.decode('utf-8')}")

import argparse

def run_experiment():
    parser = argparse.ArgumentParser(description="Run experiment matrix split into setups.")
    parser.add_argument('--setup', type=int, choices=list(range(1, 22)), required=True, help='Setup number to run (1-21)')
    parser.add_argument('--output-dir', type=str, default="analysis/results", help='Directory to save output files')
    args = parser.parse_args()
    
    # The 7 randomly selected utterances
    all_utterances = [1027, 955, 926, 1074, 985, 968, 1061]
    
    # The original 3 utterances
    original_utterances = [924, 1018, 1059]
    
    # Model Configurations
    models_standard = [
        {"model": "gpt-4o", "temperature": 0},
        {"model": "gpt-4o-mini", "temperature": 0}
    ]
    
    models_gpt5 = [
        {"model": "gpt-5-mini", "temperature": 1}
    ]
    
    # Define Setups
    setup_config = {}
    
    # Setup 1: Utterances 1-3 (3 total), Standard Models
    setup_config[1] = {
        "utterances": all_utterances[:3], # 1027, 955, 926
        "configs": models_standard
    }
    
    # Setup 2: Utterances 4-5 (2 total), Standard Models
    setup_config[2] = {
        "utterances": all_utterances[3:5], # 1074, 985
        "configs": models_standard
    }
    
    # Setup 3: Utterances 6-7 (2 total), Standard Models
    setup_config[3] = {
        "utterances": all_utterances[5:], # 968, 1061
        "configs": models_standard
    }
    
    # Setup 4: Utterances 1-4 (4 total), GPT-5-Mini
    setup_config[4] = {
        "utterances": all_utterances[:4], 
        "configs": models_gpt5
    }
    
    # Setup 5: Utterances 5-7 (3 total), GPT-5-Mini
    setup_config[5] = {
        "utterances": all_utterances[4:], 
        "configs": models_gpt5
    }

    # Setup 6: Original Utterances (3 total), Standard Models
    setup_config[6] = {
        "utterances": original_utterances, # 924, 1018, 1059
        "configs": models_standard
    }

    # Setup 7: Original Utterances (3 total), GPT-5-Mini
    setup_config[7] = {
        "utterances": original_utterances, # 924, 1018, 1059
        "configs": models_gpt5
    }

    # Remaining 21 utterances
    remaining_utterances = [951, 952, 962, 963, 965, 987, 1000, 1001, 1005, 1014, 1016, 1019, 1020, 1023, 1024, 1026, 1028, 1075, 1094, 1096, 1098]
    
    # Split remaining into chunks of 3 and 4
    # 21 items: 3, 3, 3, 3, 3, 3, 3 (7 chunks of 3)
    chunks = [remaining_utterances[i:i + 3] for i in range(0, len(remaining_utterances), 3)]
    
    # Create setups for each chunk
    # Standard: Setups 8, 9, 10, 11, 12, 13, 14
    # GPT-5-Mini: Setups 15, 16, 17, 18, 19, 20, 21
    
    current_std_setup = 8
    current_gpt5_setup = 15
    
    for i, chunk in enumerate(chunks):
        # Standard Setup
        setup_config[current_std_setup + i] = {
            "utterances": chunk,
            "configs": models_standard
        }
        # GPT-5-Mini Setup
        setup_config[current_gpt5_setup + i] = {
            "utterances": chunk,
            "configs": models_gpt5
        }
    
    current_setup = setup_config[args.setup]
    utterances = current_setup["utterances"]
    configs = current_setup["configs"]
    
    iterations = 200
    output_dir = "analysis/results"
    
    jobs = []
    for utterance in utterances:
        for config in configs:
            jobs.append((utterance, config))
    
    print(f"Running Setup {args.setup}")
    print(f"Utterances: {utterances}")
    print(f"Models: {[c['model'] for c in configs]}")
    print(f"Total Jobs: {len(jobs)}")
    
    # Run all jobs in this setup in parallel
    # Max workers = number of jobs to run them all at once (setup sizes are small, max 6 jobs)
    max_workers = len(jobs)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for utterance, config in jobs:
            executor.submit(run_single_experiment, utterance, config, iterations, output_dir)
            time.sleep(1) # Stagger start times

if __name__ == "__main__":
    run_experiment()
