import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import logging
import sys

# Configure logging: output to both console and file
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

file_handler = logging.FileHandler("debug_info.txt", mode="w", encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logging.debug("Script started.")

# Load the MACO log file
file_path = "merged_results.txt"
try:
    with open(file_path, "r", encoding="utf-8") as file:
        log_data = file.readlines()
    logging.debug(f"Loaded log file: {file_path} with {len(log_data)} lines.")
except Exception as e:
    logging.exception(f"Failed to read file {file_path}: {e}")
    sys.exit(1)

# Initialize storage for extracted data
extracted_data = []
logging.debug("Initialized extracted_data list.")

# Define regex patterns for extracting key parameters
run_pattern = re.compile(r"\[Run (\d+)\]")
best_quality_pattern = re.compile(r"Best quality found \(GPU-MACO\): ([\d.]+)")
final_quality_pattern = re.compile(r"\[SOLUTION ANALYSIS\] (\d+)/(\d+) clauses satisfied")
avg_quality_pattern = re.compile(r"avg_q=([\d.]+)")
entropy_pattern = re.compile(r"entropy=([\d.]+)")
alpha_pattern = re.compile(r"alpha=([\d.]+)")
rho_pattern = re.compile(r"rho=([\d.]+)")
noise_pattern = re.compile(r"noise_std=([\d.]+)")
minisat_verdict_pattern = re.compile(r"Classical Solver \(MiniSat\): (\w+) in ([\d.]+) seconds.")
conflicts_pattern = re.compile(r"conflicts\s+:\s+(\d+)")
propagations_pattern = re.compile(r"propagations\s+:\s+(\d+)")

logging.debug("Regex patterns defined.")

# Variables to store extracted values per run
current_run = None
run_data = {}
in_solver_stdout = False

for idx, line in enumerate(log_data):
    logging.debug(f"Processing line {idx+1}: {line.strip()}")
    # Detect entering or exiting solver STDOUT block
    if "Solver STDOUT:" in line:
        in_solver_stdout = True
        logging.debug("Entered Solver STDOUT block.")
    if "Solver STDERR:" in line:
        in_solver_stdout = False
        logging.debug("Exited Solver STDOUT block.")
    
    # Check for final verdict inside solver STDOUT block
    if in_solver_stdout:
        verdict_line = line.strip()
        if verdict_line == "SATISFIABLE":
            run_data["MiniSat Verdict"] = "SAT"
            logging.debug(f"Overridden MiniSat Verdict to SAT from solver STDOUT for run {current_run}.")
        elif verdict_line == "UNSATISFIABLE":
            run_data["MiniSat Verdict"] = "UNSAT"
            logging.debug(f"Overridden MiniSat Verdict to UNSAT from solver STDOUT for run {current_run}.")

    run_match = run_pattern.search(line)
    if run_match:
        if current_run is not None:
            extracted_data.append(run_data)
            logging.debug(f"Appended run_data for run {current_run}: {run_data}")
        current_run = int(run_match.group(1))
        run_data = {
            "Run": current_run,
            "Best Quality": None,
            "Final Quality": None,
            "Average Quality": None,
            "Entropy Evolution": None,
            "Alpha Growth": None,
            "Rho Stability": None,
            "Noise_std Effect": None,
            "MiniSat Verdict": None,
            "MiniSat Conflicts": None,
            "MiniSat Propagations": None,
        }
        logging.debug(f"Started new run_data for run {current_run}.")

    if best_quality_match := best_quality_pattern.search(line):
        try:
            run_data["Best Quality"] = float(best_quality_match.group(1))
            logging.debug(f"Extracted Best Quality: {run_data['Best Quality']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Best Quality on line {idx+1}: {e}")

    if final_quality_match := final_quality_pattern.search(line):
        try:
            clauses_satisfied = int(final_quality_match.group(1))
            total_clauses = int(final_quality_match.group(2))
            run_data["Final Quality"] = clauses_satisfied / total_clauses
            logging.debug(f"Extracted Final Quality: {run_data['Final Quality']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Final Quality on line {idx+1}: {e}")

    if avg_quality_match := avg_quality_pattern.search(line):
        try:
            run_data["Average Quality"] = float(avg_quality_match.group(1))
            logging.debug(f"Extracted Average Quality: {run_data['Average Quality']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Average Quality on line {idx+1}: {e}")

    if entropy_match := entropy_pattern.search(line):
        try:
            run_data["Entropy Evolution"] = float(entropy_match.group(1))
            logging.debug(f"Extracted Entropy Evolution: {run_data['Entropy Evolution']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Entropy Evolution on line {idx+1}: {e}")

    if alpha_match := alpha_pattern.search(line):
        try:
            run_data["Alpha Growth"] = float(alpha_match.group(1))
            logging.debug(f"Extracted Alpha Growth: {run_data['Alpha Growth']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Alpha Growth on line {idx+1}: {e}")

    if rho_match := rho_pattern.search(line):
        try:
            run_data["Rho Stability"] = float(rho_match.group(1))
            logging.debug(f"Extracted Rho Stability: {run_data['Rho Stability']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Rho Stability on line {idx+1}: {e}")

    if noise_match := noise_pattern.search(line):
        try:
            run_data["Noise_std Effect"] = float(noise_match.group(1))
            logging.debug(f"Extracted Noise_std Effect: {run_data['Noise_std Effect']} for run {current_run}.")
        except Exception as e:
            logging.exception(f"Error parsing Noise_std Effect on line {idx+1}: {e}")

    if "Classical Solver (MiniSat):" in line:
        minisat_verdict_match = minisat_verdict_pattern.search(line)
        if minisat_verdict_match:
            minisat_verdict = minisat_verdict_match.group(1)
            run_data["MiniSat Verdict"] = minisat_verdict
            logging.debug(f"Recorded MiniSat Verdict from log: {run_data['MiniSat Verdict']} for run {current_run}.")

    if "conflicts" in line:
        if conflicts_match := conflicts_pattern.search(line):
            try:
                run_data["MiniSat Conflicts"] = int(conflicts_match.group(1))
                logging.debug(f"Extracted MiniSat Conflicts: {run_data['MiniSat Conflicts']} for run {current_run}.")
            except Exception as e:
                logging.exception(f"Error parsing MiniSat Conflicts on line {idx+1}: {e}")

    if "propagations" in line:
        if propagations_match := propagations_pattern.search(line):
            try:
                run_data["MiniSat Propagations"] = int(propagations_match.group(1))
                logging.debug(f"Extracted MiniSat Propagations: {run_data['MiniSat Propagations']} for run {current_run}.")
            except Exception as e:
                logging.exception(f"Error parsing MiniSat Propagations on line {idx+1}: {e}")

if run_data:
    extracted_data.append(run_data)
    logging.debug(f"Appended final run_data for run {current_run}: {run_data}")

try:
    df_maco = pd.DataFrame(extracted_data)
    logging.debug("Converted extracted_data to DataFrame.")
except Exception as e:
    logging.exception(f"Error converting extracted_data to DataFrame: {e}")
    sys.exit(1)

df_maco.fillna("N/A", inplace=True)
logging.debug("Filled missing values with 'N/A'.")

numeric_columns = [
    "Run", "Best Quality", "Final Quality", "Average Quality",
    "Entropy Evolution", "Alpha Growth", "Rho Stability", "Noise_std Effect",
    "MiniSat Conflicts", "MiniSat Propagations"
]
for col in numeric_columns:
    try:
        df_maco[col] = pd.to_numeric(df_maco[col], errors="coerce")
        logging.debug(f"Converted column {col} to numeric.")
    except Exception as e:
        logging.exception(f"Error converting column {col} to numeric: {e}")

df_maco.dropna(inplace=True)
logging.debug("Dropped rows with NaN values after conversion.")

dataset_path = "MACO_Tuning_Dataset.csv"
try:
    df_maco.to_csv(dataset_path, index=False)
    logging.debug(f"Dataset saved to {dataset_path}.")
except Exception as e:
    logging.exception(f"Error saving dataset to {dataset_path}: {e}")

try:
    stats_summary = df_maco.describe()
    logging.debug("Computed statistical summaries.")
except Exception as e:
    logging.exception(f"Error computing statistical summaries: {e}")

try:
    corr_matrix = df_maco.corr(numeric_only=True)
    logging.debug("Computed correlation matrix.")
except Exception as e:
    logging.exception(f"Error computing correlation matrix: {e}")

corr_matrix_path = "MACO_Correlation_Matrix.csv"
try:
    corr_matrix.to_csv(corr_matrix_path)
    logging.debug(f"Correlation matrix saved to {corr_matrix_path}.")
except Exception as e:
    logging.exception(f"Error saving correlation matrix to {corr_matrix_path}: {e}")

try:
    plt.figure(figsize=(10, 5))
    plt.plot(df_maco["Run"], df_maco["Best Quality"], marker='o', linestyle='-', label="Best Quality")
    plt.xlabel("Run Number")
    plt.ylabel("Best Quality Found")
    plt.title("Best Quality Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()
    logging.debug("Displayed Best Quality Over Time plot.")
except Exception as e:
    logging.exception(f"Error plotting Best Quality Over Time: {e}")

try:
    plt.figure(figsize=(10, 5))
    plt.plot(df_maco["Run"], df_maco["Entropy Evolution"], marker='s', linestyle='-', label="Entropy Evolution", color="orange")
    plt.xlabel("Run Number")
    plt.ylabel("Entropy Value")
    plt.title("Entropy Decay Over Runs")
    plt.legend()
    plt.grid(True)
    plt.show()
    logging.debug("Displayed Entropy Decay Over Runs plot.")
except Exception as e:
    logging.exception(f"Error plotting Entropy Decay Over Runs: {e}")

try:
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of MACO Parameters")
    plt.show()
    logging.debug("Displayed Correlation Heatmap plot.")
except Exception as e:
    logging.exception(f"Error plotting Correlation Heatmap: {e}")

try:
    plt.figure(figsize=(8, 5))
    df_maco["MiniSat Verdict"].value_counts().plot(kind="bar", color=["green", "red", "gray"])
    plt.xlabel("MiniSat Verdict")
    plt.ylabel("Number of Runs")
    plt.title("MiniSat Verdict Distribution (SAT/UNSAT/TIMEOUT)")
    plt.xticks(rotation=0)
    plt.grid(axis="y")
    plt.show()
    logging.debug("Displayed MiniSat Verdict Distribution plot.")
except Exception as e:
    logging.exception(f"Error plotting MiniSat Verdict Distribution: {e}")

logging.debug("Script completed successfully.")
print(f"Dataset saved to {dataset_path}")
print(f"Correlation matrix saved to {corr_matrix_path}")
