import subprocess
import os
import time

def main():
    start_setup = 8
    end_setup = 21
    
    base_dir = "analysis/results"
    
    for setup_num in range(start_setup, end_setup + 1):
        print(f"\n========================================")
        print(f"Starting Setup {setup_num}")
        print(f"========================================")
        
        # Define specific output directory for this setup
        output_dir = os.path.join(base_dir, f"setup_{setup_num}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        cmd = [
            "python",
            "analysis/run_experiment_matrix.py",
            "--setup", str(setup_num),
            "--output-dir", output_dir
        ]
        
        try:
            # Run calling script and wait for it to finish
            subprocess.run(cmd, check=True)
            print(f"Setup {setup_num} completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Setup {setup_num} failed with error: {e}")
            # Optionally stop or continue. Continuing seems safer to get as much done as possible.
            continue
            
        # Optional: brief pause between setups
        time.sleep(2)

if __name__ == "__main__":
    main()
