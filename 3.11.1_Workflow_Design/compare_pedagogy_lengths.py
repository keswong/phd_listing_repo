import json
import statistics
import sys
import os

def get_pedagogy_lengths(file_path):
    """
    Extracts the lengths of 'pedagogyprompt' fields from the given JSON file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        lengths = []
        if 'runs' in data:
            for key, value in data['runs'].items():
                if key.endswith('_pedagogydescription'):
                    lengths.append(len(value))
        return lengths
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {file_path}")
        return []

def print_stats(name, lengths):
    """
    Prints statistical summary of the given lengths.
    """
    print(f"\n--- Statistics for {name} ---")
    if not lengths:
        print("No data found.")
        return

    print(f"Count: {len(lengths)}")
    print(f"Mean: {statistics.mean(lengths):.2f}")
    print(f"Median: {statistics.median(lengths):.2f}")
    if len(lengths) > 1:
        print(f"Std Dev: {statistics.stdev(lengths):.2f}")
    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")

def main():
    new_file = "analysis/results/interventions_924_gpt-4o-mini_pedagogical_agent_2026-01-21T15-17-57-568596.json"
    baseline_file = "analysis/final-2026-1-13/924-peda-gpt-4o-mini.json"

    # Allow overriding via command line
    if len(sys.argv) > 1:
        new_file = sys.argv[1]
    if len(sys.argv) > 2:
        baseline_file = sys.argv[2]

    print(f"Analyzing character lengths of 'pedagogydescription'...")
    print(f"New File: {new_file}")
    print(f"Baseline File: {baseline_file}")

    new_lengths = get_pedagogy_lengths(new_file)
    baseline_lengths = get_pedagogy_lengths(baseline_file)

    print_stats("New Interventions (Generated Now)", new_lengths)
    print_stats("Baseline Interventions (final-2026-1-13)", baseline_lengths)

if __name__ == "__main__":
    main()
