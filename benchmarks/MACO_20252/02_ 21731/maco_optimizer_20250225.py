#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reworked MACOConfig Tuning Script

This script uses the merged results log file (merged_results.txt) from multiple MACO runs
to derive an “optimal” configuration for the MACOConfig dataclass.
It:
  1. Parses the merged results log file to extract run data.
  2. Computes a target (ideal) vector based on high-quality runs.
  3. Defines an objective function measuring squared error relative to the target.
  4. Runs two optimizations (Powell and Differential Evolution) to tune the MACOConfig parameters.
  5. Logs the results in CSV files for review.

The tuned parameters (in order) are:
    alpha, beta, rho, initial_pheromone, clause_lr, conflict_driven_learning_rate,
    clause_weight_momentum, target_entropy, finishing_threshold, noise_std, quantum_burst_interval
"""

import re
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize, differential_evolution

# --- Set up logging ---
logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# --- File path for the merged log ---
merged_log_path = "merged_results.txt"

# --- Define regex patterns for extracting parameters and quality metrics ---
param_names = ["alpha", "beta", "rho", "initial_pheromone", "clause_lr",
               "conflict_driven_learning_rate", "clause_weight_momentum",
               "target_entropy", "finishing_threshold", "noise_std",
               "quantum_burst_interval"]

patterns = {
    "run": re.compile(r"\[Run (\d+)\]"),
    "best_quality": re.compile(r"Best quality found \(GPU-MACO\): ([\d.]+)"),
    "final_quality": re.compile(r"\[SOLUTION ANALYSIS\]\s+(\d+)/(\d+)\s+clauses satisfied"),
    "avg_quality": re.compile(r"avg_q=([\d.]+)"),
    "entropy": re.compile(r"entropy=([\d.]+)"),
    "alpha": re.compile(r"alpha=([\d.]+)"),
    "beta": re.compile(r"beta=([\d.]+)"),
    "rho": re.compile(r"rho=([\d.]+)"),
    "initial_pheromone": re.compile(r"initial_pheromone=([\d.]+)"),
    "clause_lr": re.compile(r"clause_lr=([\d.]+)"),
    "conflict_driven_learning_rate": re.compile(r"conflict_driven_learning_rate=([\d.]+)"),
    "clause_weight_momentum": re.compile(r"clause_weight_momentum=([\d.]+)"),
    "target_entropy": re.compile(r"target_entropy=([\d.]+)"),
    "finishing_threshold": re.compile(r"finishing_threshold=([\d.]+)"),
    "noise_std": re.compile(r"noise_std=([\d.]+)"),
    "quantum_burst_interval": re.compile(r"quantum_burst_interval=([\d.]+)")
}

# --- Parse merged results file ---
try:
    with open(merged_log_path, "r", encoding="utf-8") as f:
        log_lines = f.readlines()
except Exception as e:
    logging.exception(f"Failed to read merged log file {merged_log_path}: {e}")
    sys.exit(1)

run_data_list = []
current_run = None
current_data = {}

for line in log_lines:
    # Detect run boundary
    run_match = patterns["run"].search(line)
    if run_match:
        if current_data:
            run_data_list.append(current_data)
        current_run = int(run_match.group(1))
        current_data = {"Run": current_run}
    # For each parameter, try to match and store the value
    for key in param_names:
        if key not in current_data:
            m = patterns.get(key) and patterns[key].search(line)
            if m:
                try:
                    current_data[key] = float(m.group(1))
                except Exception:
                    pass
    # Extract best quality (if available)
    m_bq = patterns["best_quality"].search(line)
    if m_bq:
        current_data["Best_Quality"] = float(m_bq.group(1))
    # Extract final quality (compute as ratio if possible)
    m_fq = patterns["final_quality"].search(line)
    if m_fq:
        try:
            num_sat = int(m_fq.group(1))
            total = int(m_fq.group(2))
            current_data["Final_Quality"] = num_sat / total if total > 0 else np.nan
        except Exception:
            current_data["Final_Quality"] = np.nan
    # Optionally, add other metrics (avg_quality, entropy, etc.) if needed

# Append the last run's data if present
if current_data:
    run_data_list.append(current_data)

# Create a DataFrame from the extracted run data
df_runs = pd.DataFrame(run_data_list)
if df_runs.empty:
    logging.error("No run data could be extracted from the merged log file.")
    sys.exit(1)

# --- Filter for runs with high final quality (e.g. > 0.98) to derive a target setting ---
quality_threshold = 0.98
df_success = df_runs[df_runs["Final_Quality"] >= quality_threshold]
if df_success.empty:
    # If no run meets the threshold, use all runs
    logging.warning("No runs met the high-quality threshold; using all runs for target derivation.")
    df_success = df_runs

# Compute the target vector as the average of each parameter among successful runs
target_vector = []
for key in param_names:
    if key in df_success.columns:
        target_val = df_success[key].mean()
    else:
        # If missing, use a default (could be the original hard-coded value)
        defaults = {
            "alpha": 3.60506,
            "beta": 1.90,
            "rho": 0.14,
            "initial_pheromone": 0.17005,
            "clause_lr": 0.20,
            "conflict_driven_learning_rate": 0.26926,
            "clause_weight_momentum": 0.89999,
            "target_entropy": 0.69891,
            "finishing_threshold": 0.99900,
            "noise_std": 0.10,
            "quantum_burst_interval": 100.0
        }
        target_val = defaults.get(key, 0.0)
    target_vector.append(target_val)

target_vector = np.array(target_vector)
logging.info("Derived target MACOConfig vector:")
for key, val in zip(param_names, target_vector):
    logging.info(f"  {key} = {val:.5f}")

# --- Define the objective function ---
def maco_objective_function(x):
    """
    Objective: minimize the squared error between candidate MACOConfig parameters and the target.
    x is a candidate vector of parameters in the following order:
      alpha, beta, rho, initial_pheromone, clause_lr,
      conflict_driven_learning_rate, clause_weight_momentum,
      target_entropy, finishing_threshold, noise_std, quantum_burst_interval
    """
    return np.sum((np.array(x) - target_vector) ** 2)

# --- Initial guess: Use the derived target (or slightly perturbed version) ---
initial_guess = target_vector * (1 + np.random.uniform(-0.05, 0.05, size=target_vector.shape))
logging.info("Initial guess for optimization:")
for key, val in zip(param_names, initial_guess):
    logging.info(f"  {key} = {val:.5f}")

# --- Optimize using SciPy's minimize() with the Powell method ---
result_minimize = minimize(maco_objective_function, initial_guess, method="Powell")
logging.info("Powell Optimization Results:")
for key, val in zip(param_names, result_minimize.x):
    logging.info(f"  {key} = {val:.5f}")
logging.info(f"Objective value: {result_minimize.fun:.6f}")

# --- Define bounds for each parameter (from historical runs or design constraints) ---
bounds = [
    (3.5, 4.0),    # alpha
    (1.5, 2.5),    # beta
    (0.10, 0.20),  # rho
    (0.15, 0.20),  # initial_pheromone
    (0.15, 0.30),  # clause_lr
    (0.20, 0.30),  # conflict_driven_learning_rate
    (0.80, 0.95),  # clause_weight_momentum
    (0.65, 0.75),  # target_entropy
    (0.995, 1.0),  # finishing_threshold
    (0.05, 0.15),  # noise_std
    (80, 120)      # quantum_burst_interval
]

# --- Optimize using Differential Evolution ---
result_de = differential_evolution(maco_objective_function, bounds, strategy="best1bin", disp=True)
logging.info("Differential Evolution Optimization Results:")
for key, val in zip(param_names, result_de.x):
    logging.info(f"  {key} = {val:.5f}")
logging.info(f"Objective value: {result_de.fun:.6f}")

# --- Log optimization results to CSV ---
results_df = pd.DataFrame(
    [result_minimize.x, result_de.x],
    columns=param_names
)
results_df["Objective_Value"] = [result_minimize.fun, result_de.fun]
results_df["Method"] = ["Powell", "Differential_Evolution"]

csv_results_path = "Optimizer_Results.csv"
results_df.to_csv(csv_results_path, index=False)
logging.info(f"Optimization results written to {csv_results_path}")

# --- (Optional) Visualization of the best quality evolution ---
if "Run" in df_runs.columns and "Best_Quality" in df_runs.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(df_runs["Run"], df_runs["Best_Quality"], marker='o', linestyle='-')
    plt.xlabel("Run Number")
    plt.ylabel("Best Quality Found")
    plt.title("Best Quality Over Runs")
    plt.grid(True)
    plt.show()

# --- Save the derived target vector as the new MACOConfig default setting ---
# You can use this vector to update your MACOConfig dataclass in your main MACO solver.
with open("Tuned_MACOConfig.txt", "w", encoding="utf-8") as f:
    f.write("Tuned MACOConfig Parameters:\n")
    for key, val in zip(param_names, target_vector):
        f.write(f"{key} = {val:.5f}\n")
logging.info("Tuned MACOConfig parameters saved to Tuned_MACOConfig.txt")
