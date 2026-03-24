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
    """Analyzes a single MACO run and extracts key metrics."""
    summary = {}

    # Extract Run Number
    match = re.search(r"\[Run (\d+)\]", run_text)
    if match:
        summary["run_number"] = int(match.group(1))

    # Extract Best Quality from the new "[SOLUTION ANALYSIS] Best Quality Found (GPU-MACO):" line.
    match = re.search(r"\[SOLUTION ANALYSIS\]\s+Best Quality Found \(GPU-MACO\):\s+(\d\.\d+)", run_text)
    if match:
        summary["best_q"] = float(match.group(1))
    else:
        match = re.search(r"\[Iter \d+\]\s+best_q=(\d\.\d+)", run_text)
        summary["best_q"] = float(match.group(1)) if match else None

    # Extract Final Quality from the "Satisfied Clauses:" line
    match = re.search(r"Satisfied Clauses:\s+(\d+)/(\d+)\s+\((\d+\.\d+)%\)", run_text)
    if match:
        satisfied = int(match.group(1))
        total = int(match.group(2))
        summary["final_q"] = satisfied / total
    else:
        # Fallback: use last [Iter] best_q if available
        match = re.findall(r"\[Iter \d+\]\s+best_q=(\d\.\d+)", run_text)
        summary["final_q"] = float(match[-1]) if match else None

    # Extract Average Quality from lines like "avg_q=..."
    avg_q_matches = re.findall(r"avg_q=(\d\.\d+)", run_text)
    summary["avg_q"] = float(avg_q_matches[-1]) if avg_q_matches else None

    # Extract Entropy Evolution from lines with "entropy="
    entropy_matches = re.findall(r"entropy=(\d+\.\d+)", run_text)
    if entropy_matches:
        entropy_values = [float(e) for e in entropy_matches]
        if len(entropy_values) > 5:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f} -> {entropy_values[len(entropy_values)//4]:.3f} -> {entropy_values[len(entropy_values)//2]:.3f} -> {entropy_values[-1]:.3f}"
        elif len(entropy_values) > 1:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f} -> {entropy_values[-1]:.3f}"
        else:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f}"
    else:
        summary["entropy_evolution"] = None

    # Extract Alpha, Rho, and Noise Std effect
    alpha_matches = re.findall(r"alpha=(\d+\.?\d*)", run_text)
    rho_matches = re.findall(r"rho=(\d+\.?\d*)", run_text)
    noise_matches = re.findall(r"noise_std=(\d+\.?\d*)", run_text)
    summary["alpha_growth"] = " -> ".join(f"{float(val):.1f}" for val in alpha_matches) if alpha_matches else "N/A"
    summary["rho_stability"] = " -> ".join(f"{float(val):.3f}" for val in rho_matches) if rho_matches else "N/A"
    summary["noise_std_effect"] = " -> ".join(f"{float(val):.3f}" for val in noise_matches) if noise_matches else "N/A"

    # Extract Partial Reset Triggers (look for "Partial reset triggered at iteration")
    reset_triggers = re.findall(r"Partial reset triggered at iteration (\d+)", run_text)
    summary["reset_triggers"] = ", ".join(reset_triggers) if reset_triggers else "None"

    # Extract MiniSat Verdict via the "Return code:" line
    match = re.search(r"Return code:\s*(\d+)", run_text)
    if match:
        return_code = int(match.group(1))
        if return_code == 10:
            summary["minisat_verdict"] = "SAT"
        elif return_code == 20:
            summary["minisat_verdict"] = "UNSAT"
        elif return_code == 124:
            summary["minisat_verdict"] = "TIMEOUT"
        else:
            summary["minisat_verdict"] = f"Code {return_code}"
    else:
        summary["minisat_verdict"] = "Unknown"

    # Extract MiniSat Conflicts
    match = re.search(r"conflicts\s+:\s+(\d+)", run_text)
    summary["minisat_conflicts"] = int(match.group(1)) if match else None

    # Extract MiniSat Propagations
    match = re.search(r"propagations\s+:\s+(\d+)", run_text)
    summary["minisat_propagations"] = int(match.group(1)) if match else None

    # Extract UNSAT Clauses Count (if present)
    summary["unsat_clauses"] = None
    if summary.get("minisat_verdict") in ("UNSAT", "Code 20"):
        match = re.search(r"\[HARD CONSTRAINTS\]\s+(\d+)\s+clauses remained unsatisfied\.", run_text)
        if match:
            summary["unsat_clauses"] = int(match.group(1))

    return summary

def analyze_all_runs(file_path):
    """Analyzes all runs in the given file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()

    # Split file content into individual runs. Adjust this pattern if needed.
    runs_raw = re.split(r"(\[Run \d+\])", file_content)
    runs = []
    for i in range(1, len(runs_raw), 2):
        run_text = runs_raw[i] + runs_raw[i+1]
        runs.append(run_text)

    all_summaries = []
    for run_text in runs:
        summary = analyze_run(run_text)
        all_summaries.append(summary)
    return all_summaries

def print_run_summaries(run_summaries):
    """Prints the extracted summaries in a formatted way."""
    for summary in run_summaries:
        run_num = summary.get("run_number", "N/A")
        print(f"📊 **MACO Test Run {run_num} – Summary**")
        print(f"✔ **Best Quality:** {summary.get('best_q', 'N/A')}")
        print(f"✔ **Final Quality:** {summary.get('final_q', 'N/A')}")
        print(f"✔ **Entropy Evolution:** {summary.get('entropy_evolution', 'N/A')}")
        print(f"✔ **Alpha Growth:** {summary.get('alpha_growth', 'N/A')}")
        print(f"✔ **Rho Stability:** {summary.get('rho_stability', 'N/A')}")
        print(f"✔ **Noise_std Effect:** {summary.get('noise_std_effect', 'N/A')}")
        print(f"✔ **Partial Reset Triggers:** {summary.get('reset_triggers', 'N/A')}")
        print(f"✔ **MiniSat Verdict:** {summary.get('minisat_verdict', 'N/A')}")
        print(f"✔ **MiniSat Conflicts:** {summary.get('minisat_conflicts', 'N/A')}")
        print(f"✔ **MiniSat Propagations:** {summary.get('minisat_propagations', 'N/A')}")
        if summary.get("unsat_clauses") is not None and summary.get("best_q", 0) > 0.99:
            print(f"⚠ **Potential ε-SAT Case: {summary['unsat_clauses']} clauses unsatisfied.**")
        print()

if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    merged_file = merge_txt_files(current_folder)
    print(f"Merged files into: {merged_file}\n")
    run_summaries = analyze_all_runs(merged_file)
    print_run_summaries(run_summaries)
