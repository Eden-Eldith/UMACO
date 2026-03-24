import re

def analyze_run(run_text):
    """Analyzes a single MACO run and extracts key metrics."""
    summary = {}

    # Extract Run Number
    match = re.search(r"\[Run (\d+)\]", run_text)
    if match:
        summary["run_number"] = int(match.group(1))

    # Extract Best Quality
    match = re.search(r"Best quality found \(GPU-MACO\):\s+(\d\.\d+)", run_text)
    if match:
        summary["best_q"] = float(match.group(1))
    else:
        match = re.search(r"\[Iter \d+\] best_q=(\d\.\d+)", run_text)
        summary["best_q"] = float(match.group(1)) if match else None

    # Extract Final Quality
    match = re.search(r"\[SOLUTION ANALYSIS\] (\d+)/(\d+) clauses satisfied \((\d+\.\d+)%\)", run_text)
    if match:
        satisfied = int(match.group(1))
        total = int(match.group(2))
        summary["final_q"] = satisfied / total
    else:
        match = re.findall(r"\[Iter \d+\] best_q=(\d\.\d+)", run_text)
        summary["final_q"] = float(match[-1]) if match else None

    # Extract Average Quality
    avg_q_matches = re.findall(r"avg_q=(\d\.\d+)", run_text)
    summary["avg_q"] = float(avg_q_matches[-1]) if avg_q_matches else None

    # Extract Entropy Evolution
    entropy_matches = re.findall(r"entropy=(\d\.\d+)", run_text)
    if entropy_matches:
        entropy_values = [float(e) for e in entropy_matches]
        num_values = len(entropy_values)
        if num_values > 5:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f} -> {entropy_values[num_values//4]:.3f} -> {entropy_values[num_values//2]:.3f} -> {entropy_values[-1]:.3f}"
        elif num_values > 1:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f} -> {entropy_values[-1]:.3f}"
        else:
            summary["entropy_evolution"] = f"{entropy_values[0]:.3f}"

    # Alpha, Rho, and Noise Stability (Ensure proper conversion to float)
    alpha_matches = [float(a) for a in re.findall(r"alpha=(\d+\.?\d*)", run_text)]
    rho_matches = [float(r) for r in re.findall(r"rho=(\d+\.?\d*)", run_text)]
    noise_matches = [float(n) for n in re.findall(r"noise_std=(\d+\.?\d*)", run_text)]

    summary["alpha_growth"] = " -> ".join(f"{val:.1f}" for val in alpha_matches) if alpha_matches else "N/A"
    summary["rho_stability"] = " -> ".join(f"{val:.3f}" for val in rho_matches) if rho_matches else "N/A"
    summary["noise_std_effect"] = " -> ".join(f"{val:.3f}" for val in noise_matches) if noise_matches else "N/A"

    # Reset Triggers
    reset_triggers = re.findall(r"Partial reset triggered at iteration (\d+)", run_text)
    summary["reset_triggers"] = ", ".join(reset_triggers) if reset_triggers else "None"

    # MiniSat Verdict - Corrected to Check Return Code!
    match = re.search(r"Return code:\s*(\d+)", run_text)
    if match:
        return_code = int(match.group(1))
        if return_code == 10:
            summary["minisat_verdict"] = "SAT"
        elif return_code == 20:
            summary["minisat_verdict"] = "UNSAT"
        else:
            summary["minisat_verdict"] = "Unknown"
    else:
        summary["minisat_verdict"] = "Unknown"

    # MiniSat Conflicts
    match = re.search(r"conflicts\s+:\s+(\d+)", run_text)
    summary["minisat_conflicts"] = int(match.group(1)) if match else None

    # MiniSat Propagations
    match = re.search(r"propagations\s+:\s+(\d+)", run_text)
    summary["minisat_propagations"] = int(match.group(1)) if match else None

    # UNSAT Clauses Count (Only if MiniSat returns UNSAT)
    summary["unsat_clauses"] = None
    if summary["minisat_verdict"] == "UNSAT":
        match = re.search(r"\[HARD CONSTRAINTS\] (\d+) clauses remained unsatisfied\.", run_text)
        if match:
            summary["unsat_clauses"] = int(match.group(1))

    return summary


def analyze_all_runs(file_path):
    """Analyzes all runs in the given file."""
    with open(file_path, 'r') as f:
        file_content = f.read()

    # Split the file content into individual runs
    runs = re.split(r"(-{2,}\n\n\[Run \d+\])", file_content)
    runs = [runs[i-1] + runs[i] for i in range(1, len(runs), 2)]  # Combine the separator with run

    all_summaries = []
    for run_text in runs:
        summary = analyze_run(run_text)
        all_summaries.append(summary)
    return all_summaries


# Perform the analysis
file_path = 'merged_results_3 (1).txt'  # Using the exact file path you provided
run_summaries = analyze_all_runs(file_path)

# Print the summaries in the requested format
for summary in run_summaries:
    print(f"📊 **MACO Test Run {summary['run_number']} – Summary**")
    print(f"✔ **Best Quality:** {summary['best_q']}")
    print(f"✔ **Final Quality:** {summary['final_q']}")
    print(f"✔ **Entropy Evolution:** {summary['entropy_evolution']}")
    print(f"✔ **Alpha Growth:** {summary['alpha_growth']}")
    print(f"✔ **Rho Stability:** {summary['rho_stability']}")
    print(f"✔ **Noise_std Effect:** {summary['noise_std_effect']}")
    print(f"✔ **Partial Reset Triggers:** {summary['reset_triggers']}")
    print(f"✔ **MiniSat Verdict:** {summary['minisat_verdict']}")
    print(f"✔ **MiniSat Conflicts:** {summary['minisat_conflicts']}")
    print(f"✔ **MiniSat Propagations:** {summary['minisat_propagations']}")
    if summary["unsat_clauses"] is not None and summary["best_q"] > 0.99:
        print(f"⚠ **Potential ε-SAT Case: {summary['unsat_clauses']} clauses unsatisfied.**")
    print()
