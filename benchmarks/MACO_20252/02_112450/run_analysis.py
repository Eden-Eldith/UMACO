import os
import re

def extract_number(filename):
    """Extracts a number from the filename for sorting purposes."""
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def merge_txt_files(current_folder, output_filename="merged_results.txt"):
    """
    Merges all .txt files in the current folder (except the output file itself)
    into a single file sorted numerically by numbers in the filenames.
    """
    all_txt_files = [f for f in os.listdir(current_folder)
                     if f.endswith(".txt") and f != output_filename]
    sorted_files = sorted(all_txt_files, key=extract_number)
    
    output_file_path = os.path.join(current_folder, output_filename)
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        for filename in sorted_files:
            file_path = os.path.join(current_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read() + "\n")
    return output_file_path

def analyze_run(run_text):
    """Analyzes a single MACO run and extracts key metrics from the final section."""
    summary = {}

    # Extract Run Number
    match = re.search(r"\[Run (\d+)\]", run_text)
    summary["run_number"] = int(match.group(1)) if match else None

    # Extract Best Quality
    match = re.search(r"Best quality found \(GPU-MACO\):\s+(\d\.\d+)", run_text)
    if match:
        summary["best_q"] = float(match.group(1))
    else:
        match = re.search(r"\[Iter \d+\]\s+best_q=(\d\.\d+)", run_text)
        summary["best_q"] = float(match.group(1)) if match else None

    # Extract Final Quality from the final section using the [SOLUTION ANALYSIS] marker
    match = re.search(r"\[SOLUTION ANALYSIS\]\s+(\d+)/(\d+)", run_text)
    if match:
        satisfied = int(match.group(1))
        total = int(match.group(2))
        summary["final_q"] = satisfied / total
    else:
        iter_matches = re.findall(r"\[Iter \d+\]\s+best_q=(\d\.\d+)", run_text)
        summary["final_q"] = float(iter_matches[-1]) if iter_matches else None

    # Extract Average Quality
    avg_q_matches = re.findall(r"avg_q=(\d\.\d+)", run_text)
    summary["avg_q"] = float(avg_q_matches[-1]) if avg_q_matches else None

    # Extract Entropy Evolution
    entropy_matches = re.findall(r"entropy=(\d+\.\d+)", run_text)
    if entropy_matches:
        entropy_values = [float(e) for e in entropy_matches]
        num_values = len(entropy_values)
        if num_values > 5:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f} -> {entropy_values[num_values//4]:.3f} -> {entropy_values[num_values//2]:.3f} -> {entropy_values[-1]:.3f}"
        elif num_values > 1:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f} -> {entropy_values[-1]:.3f}"
        else:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f}"
    else:
        summary["entropy_evolution"] = "N/A"

    # Extract Alpha, Rho, and Noise_std Effect
    alpha_matches = re.findall(r"alpha=(\d+\.?\d*)", run_text)
    rho_matches = re.findall(r"rho=(\d+\.?\d*)", run_text)
    noise_matches = re.findall(r"noise_std=(\d+\.?\d*)", run_text)
    summary["alpha_growth"] = " -> ".join(f"{float(val):.1f}" for val in alpha_matches) if alpha_matches else "N/A"
    summary["rho_stability"] = " -> ".join(f"{float(val):.3f}" for val in rho_matches) if rho_matches else "N/A"
    summary["noise_std_effect"] = " -> ".join(f"{float(val):.3f}" for val in noise_matches) if noise_matches else "N/A"

    # Extract Partial Reset Triggers
    reset_triggers = re.findall(r"Partial reset triggered at iteration (\d+)", run_text)
    summary["reset_triggers"] = ", ".join(reset_triggers) if reset_triggers else "None"

    # Extract MiniSat Verdict using the last occurrence of Return code
    codes = re.findall(r"Return code:\s*(\d+)", run_text)
    if codes:
        return_code = int(codes[-1])
        if return_code == 10:
            summary["minisat_verdict"] = "SAT"
        elif return_code == 20:
            summary["minisat_verdict"] = "UNSAT"
        elif return_code == 124:
            summary["minisat_verdict"] = "TIMEOUT"
        else:
            summary["minisat_verdict"] = f"Unknown (Code {return_code})"
    else:
        summary["minisat_verdict"] = "Unknown"

    # Extract MiniSat Conflicts
    match = re.search(r"conflicts\s+:\s+(\d+)", run_text)
    summary["minisat_conflicts"] = int(match.group(1)) if match else None

    # Extract MiniSat Propagations
    match = re.search(r"propagations\s+:\s+(\d+)", run_text)
    summary["minisat_propagations"] = int(match.group(1)) if match else None

    # Extract UNSAT Clauses Count (if applicable)
    summary["unsat_clauses"] = None
    if summary.get("minisat_verdict") in ("UNSAT", "Unknown (Code 20)"):
        match = re.search(r"\[HARD CONSTRAINTS\]\s+(\d+)\s+clauses remained unsatisfied\.", run_text)
        if match:
            summary["unsat_clauses"] = int(match.group(1))

    return summary

def analyze_all_runs(file_path):
    """Analyzes all runs in the merged file using a separator similar to the v1 approach."""
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    runs = re.split(r"(-{2,}\n\n\[Run \d+\])", file_content)
    runs = [runs[i-1] + runs[i] for i in range(1, len(runs), 2)]
    all_summaries = [analyze_run(run_text) for run_text in runs]
    return all_summaries

def print_run_summaries(run_summaries):
    """Prints the extracted summaries in a formatted way."""
    for summary in run_summaries:
        print(f"📊 MACO Test Run {summary.get('run_number', 'N/A')} – Summary")
        print(f"  ✔ Best Quality: {summary.get('best_q', 'N/A')}")
        print(f"  ✔ Final Quality: {summary.get('final_q', 'N/A')}")
        print(f"  ✔ Average Quality: {summary.get('avg_q', 'N/A')}")
        print(f"  ✔ Entropy Evolution: {summary.get('entropy_evolution', 'N/A')}")
        print(f"  ✔ Alpha Growth: {summary.get('alpha_growth', 'N/A')}")
        print(f"  ✔ Rho Stability: {summary.get('rho_stability', 'N/A')}")
        print(f"  ✔ Noise_std Effect: {summary.get('noise_std_effect', 'N/A')}")
        print(f"  ✔ Partial Reset Triggers: {summary.get('reset_triggers', 'N/A')}")
        print(f"  ✔ MiniSat Verdict: {summary.get('minisat_verdict', 'N/A')}")
        print(f"  ✔ MiniSat Conflicts: {summary.get('minisat_conflicts', 'N/A')}")
        print(f"  ✔ MiniSat Propagations: {summary.get('minisat_propagations', 'N/A')}")
        if summary.get("unsat_clauses") is not None and summary.get("best_q", 0) > 0.99:
            print(f"  ⚠ Potential ε-SAT Case: {summary['unsat_clauses']} clauses unsatisfied.")
        print()

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    merged_file = merge_txt_files(current_folder)
    print(f"Merged files into: {merged_file}\n")
    run_summaries = analyze_all_runs(merged_file)
    print_run_summaries(run_summaries)
    input("Press Enter to exit...")
