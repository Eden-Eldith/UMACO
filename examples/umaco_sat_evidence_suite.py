#!/usr/bin/env python3
"""
UMACO SAT Evidence Suite — Multi-Instance-Type Scaling Analysis
===============================================================
Generates SAT instances across 5 distinct problem families,
runs MACO (GPU, full macov8 solver with 6 CUDA kernels) and MiniSat
on each, verifies all assignments independently, fits scaling curves
per family, and produces a comprehensive evidence report.

Instance families:
  1. Random 3-SAT at phase transition (ratio 4.267)
  2. Random 3-SAT with planted solution (guaranteed SAT)
  3. Pigeonhole principle PHP(n) (guaranteed UNSAT, resolution-hard)
  4. Graph 3-coloring (structured SAT, planted solution)
  5. Tseitin formulas on random graphs (guaranteed UNSAT, structured)

Usage:
  python umaco_sat_evidence_suite.py
  python umaco_sat_evidence_suite.py --runs 10 --minisat C:/msys64/ucrt64/bin/minisat.exe

All Optuna-tuned parameters from macov8: alpha=3.54879, beta=2.38606,
rho=0.13814, initial_pheromone=0.20498, noise_std=0.11266, n_ants=3072,
conflict_driven_lr=0.21015, clause_weight_momentum=0.87959,
target_entropy=0.68894, finishing_threshold=0.99663,
partial_reset_stagnation=40, quantum_burst_interval=100.
"""

import sys
import os
import time
import math
import argparse
import logging
import importlib
import importlib.util
import tempfile
import subprocess
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Import macov8 (dashed filename requires importlib)
# ---------------------------------------------------------------------------

def _import_macov8():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    macov8_path = os.path.join(script_dir, "macov8no-3-25-02-2025.py")
    if not os.path.exists(macov8_path):
        raise FileNotFoundError(f"Cannot find macov8 at {macov8_path}")
    spec = importlib.util.spec_from_file_location("macov8", macov8_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

macov8 = _import_macov8()
MACOConfig = macov8.MACOConfig
MACOSystem = macov8.MACOSystem


# ===================================================================
# Instance generators
# ===================================================================

def generate_random_3sat(num_vars: int, ratio: float = 4.267,
                         seed: int = 42) -> Tuple[List[List[int]], List[int], dict]:
    """Random 3-SAT at specified clause-to-variable ratio."""
    rng = np.random.RandomState(seed)
    num_clauses = int(num_vars * ratio)
    variables = list(range(1, num_vars + 1))
    clauses = []
    for _ in range(num_clauses):
        vs = rng.choice(variables, size=3, replace=False)
        clause = [int(v) if rng.random() < 0.5 else int(-v) for v in vs]
        clauses.append(clause)
    return clauses, variables, {
        "family": "random_3sat",
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "ratio": ratio,
        "expected_sat": "probabilistic",
    }


def generate_planted_3sat(num_vars: int, ratio: float = 4.267,
                          seed: int = 42) -> Tuple[List[List[int]], List[int], dict]:
    """Random 3-SAT with a planted satisfying assignment (guaranteed SAT)."""
    rng = np.random.RandomState(seed)
    num_clauses = int(num_vars * ratio)
    variables = list(range(1, num_vars + 1))
    planted = {v: bool(rng.randint(2)) for v in variables}
    clauses = []
    for _ in range(num_clauses):
        vs = rng.choice(variables, size=3, replace=False)
        clause = [int(v) if rng.random() < 0.5 else int(-v) for v in vs]
        # Ensure at least one literal is satisfied by planted assignment
        satisfied = any(
            (lit > 0 and planted[abs(lit)]) or (lit < 0 and not planted[abs(lit)])
            for lit in clause
        )
        if not satisfied:
            idx = rng.randint(3)
            v = abs(clause[idx])
            clause[idx] = int(v) if planted[v] else int(-v)
        clauses.append(clause)
    return clauses, variables, {
        "family": "planted_3sat",
        "num_vars": num_vars,
        "num_clauses": num_clauses,
        "ratio": ratio,
        "expected_sat": "SAT",
    }


def generate_pigeonhole(n: int) -> Tuple[List[List[int]], List[int], dict]:
    """
    Pigeonhole principle PHP(n): fit n+1 pigeons into n holes.
    Variables: p_{i,j} = pigeon i in hole j.
    Guaranteed UNSAT. Known to be hard for resolution-based solvers.
    """
    num_pigeons = n + 1
    num_holes = n

    def var(pigeon, hole):
        return pigeon * num_holes + hole + 1

    clauses = []
    # Each pigeon in at least one hole
    for i in range(num_pigeons):
        clauses.append([var(i, j) for j in range(num_holes)])
    # Each hole has at most one pigeon
    for j in range(num_holes):
        for i1 in range(num_pigeons):
            for i2 in range(i1 + 1, num_pigeons):
                clauses.append([-var(i1, j), -var(i2, j)])

    num_vars = num_pigeons * num_holes
    variables = list(range(1, num_vars + 1))
    return clauses, variables, {
        "family": "pigeonhole",
        "n": n,
        "pigeons": num_pigeons,
        "holes": num_holes,
        "num_vars": num_vars,
        "num_clauses": len(clauses),
        "expected_sat": "UNSAT",
    }


def generate_graph_coloring(num_vertices: int, num_colors: int = 3,
                            edge_prob: float = 0.3,
                            seed: int = 42) -> Tuple[List[List[int]], List[int], dict]:
    """
    Graph k-coloring as SAT with planted valid coloring (guaranteed SAT).
    Encodes: each vertex gets exactly one color, adjacent vertices differ.
    """
    rng = np.random.RandomState(seed)
    planted_colors = [rng.randint(num_colors) for _ in range(num_vertices)]

    edges = []
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if rng.random() < edge_prob:
                edges.append((i, j))

    def var(vertex, color):
        return vertex * num_colors + color + 1

    clauses = []
    # Each vertex has at least one color
    for v in range(num_vertices):
        clauses.append([var(v, c) for c in range(num_colors)])
    # Each vertex has at most one color
    for v in range(num_vertices):
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                clauses.append([-var(v, c1), -var(v, c2)])
    # Adjacent vertices have different colors
    for (u, v) in edges:
        for c in range(num_colors):
            clauses.append([-var(u, c), -var(v, c)])

    num_vars = num_vertices * num_colors
    variables = list(range(1, num_vars + 1))
    return clauses, variables, {
        "family": "graph_coloring",
        "num_vertices": num_vertices,
        "num_colors": num_colors,
        "num_edges": len(edges),
        "num_vars": num_vars,
        "num_clauses": len(clauses),
        "expected_sat": "SAT",
    }


def _xor_clauses(lits: List[int], parity: int, next_var: int) -> Tuple[List[List[int]], int]:
    """
    Encode XOR(lits) = parity as CNF using auxiliary variable chaining.
    Returns (clauses, updated next_var).
    """
    if len(lits) == 0:
        return ([[]] if parity == 1 else []), next_var
    if len(lits) == 1:
        return ([[lits[0]]] if parity == 1 else [[-lits[0]]]), next_var
    if len(lits) == 2:
        a, b = lits
        if parity == 1:
            return [[a, b], [-a, -b]], next_var
        else:
            return [[a, -b], [-a, b]], next_var

    # Chain: aux = a XOR b, then aux XOR c, etc.
    clauses = []
    current = lits[0]
    for i in range(1, len(lits)):
        b = lits[i]
        if i == len(lits) - 1:
            # Final pair: target parity
            if parity == 1:
                clauses.extend([[current, b], [-current, -b]])
            else:
                clauses.extend([[current, -b], [-current, b]])
        else:
            aux = next_var
            next_var += 1
            # aux <-> (current XOR b)
            clauses.extend([
                [-current, -b, -aux],
                [current, b, -aux],
                [current, -b, aux],
                [-current, b, aux],
            ])
            current = aux
    return clauses, next_var


def generate_tseitin(num_vertices: int, edge_prob: float = 0.4,
                     seed: int = 42) -> Tuple[List[List[int]], List[int], dict]:
    """
    Tseitin formula on a random connected graph with odd total parity.
    Guaranteed UNSAT. Known to be hard for resolution.
    Variables: one per edge, plus auxiliaries for XOR chaining.
    """
    rng = np.random.RandomState(seed)

    # Build random connected graph: spanning tree + extra edges
    edges = set()
    in_tree = {0}
    remaining = list(range(1, num_vertices))
    rng.shuffle(remaining)
    for v in remaining:
        u = rng.choice(list(in_tree))
        edges.add((min(u, v), max(u, v)))
        in_tree.add(v)
    for i in range(num_vertices):
        for j in range(i + 1, num_vertices):
            if (i, j) not in edges and rng.random() < edge_prob:
                edges.add((i, j))
    edges = sorted(edges)

    num_edges = len(edges)
    edge_to_var = {e: i + 1 for i, e in enumerate(edges)}

    # Parities: vertex 0 gets parity 1, rest get 0 → odd total → UNSAT
    parities = [0] * num_vertices
    parities[0] = 1

    next_var = num_edges + 1
    all_clauses = []

    for v in range(num_vertices):
        incident = [edge_to_var[e] for e in edges if v in e]
        if not incident:
            if parities[v] == 1:
                all_clauses.append([])  # empty clause → UNSAT
            continue
        new_clauses, next_var = _xor_clauses(incident, parities[v], next_var)
        all_clauses.extend(new_clauses)

    total_vars = next_var - 1
    variables = list(range(1, total_vars + 1))
    return all_clauses, variables, {
        "family": "tseitin",
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "num_vars": total_vars,
        "num_clauses": len(all_clauses),
        "expected_sat": "UNSAT",
    }


# ===================================================================
# Solver wrappers
# ===================================================================

def run_maco(clauses, variables, config):
    """Run macov8 MACOSystem. Returns (assignment_np, quality, elapsed, iters)."""
    solver = MACOSystem(config=config)
    solver.initialize_search_space(formula=clauses, variables=variables)
    t0 = time.time()
    best_assignment, best_quality = solver.solve()
    elapsed = time.time() - t0
    iters_used = solver.last_improvement_iter if hasattr(solver, 'last_improvement_iter') else -1
    return best_assignment, float(best_quality), elapsed, iters_used


def write_dimacs(clauses, num_vars, filepath):
    with open(filepath, "w") as f:
        f.write(f"p cnf {num_vars} {len(clauses)}\n")
        for clause in clauses:
            f.write(" ".join(str(lit) for lit in clause) + " 0\n")


def run_minisat(clauses, num_vars, solver_path, timeout=3600):
    """Run MiniSat on the same instance. Returns (status, elapsed)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp:
        dimacs_file = tmp.name
        write_dimacs(clauses, num_vars, dimacs_file)
    try:
        t0 = time.time()
        result = subprocess.run(
            [solver_path, dimacs_file],
            capture_output=True, text=True, check=False, timeout=timeout
        )
        elapsed = time.time() - t0
        is_sat = any("SATISFIABLE" in ln.upper() and "UNSATISFIABLE" not in ln.upper()
                      for ln in result.stdout.splitlines())
        is_unsat = any("UNSATISFIABLE" in ln.upper() for ln in result.stdout.splitlines())
        if is_unsat:
            status = "UNSAT"
        elif is_sat:
            status = "SAT"
        else:
            status = "UNKNOWN"
        return status, elapsed
    except subprocess.TimeoutExpired:
        return "TIMEOUT", timeout
    except Exception as e:
        logging.warning(f"MiniSat error: {e}")
        return "ERROR", 0.0
    finally:
        if os.path.exists(dimacs_file):
            os.remove(dimacs_file)


def verify_assignment(assignment, clauses):
    """Independent clause-by-clause verification. Returns (satisfied, total)."""
    if assignment is None:
        return 0, len(clauses)
    satisfied = 0
    for clause in clauses:
        for lit in clause:
            var_idx = abs(lit) - 1
            if var_idx >= len(assignment):
                continue
            if (lit > 0 and assignment[var_idx] == 1) or \
               (lit < 0 and assignment[var_idx] == 0):
                satisfied += 1
                break
    return satisfied, len(clauses)


# ===================================================================
# Curve fitting
# ===================================================================

def polynomial(x, a, b, c):
    return a * np.power(x, b) + c

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def fit_scaling(sizes, times):
    """Fit polynomial and exponential models. Returns dict of results."""
    sizes = np.array(sizes, dtype=float)
    times = np.array(times, dtype=float)
    results = {}

    # Polynomial: a * n^b + c
    try:
        popt, _ = curve_fit(polynomial, sizes, times,
                            p0=[1e-4, 2.5, 0],
                            bounds=([0, 0.5, -1000], [1e6, 8.0, 1000]),
                            maxfev=50000)
        pred = polynomial(sizes, *popt)
        ss_res = np.sum((times - pred) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        results["polynomial"] = {"params": popt, "r2": r2, "exponent": popt[1]}
    except Exception:
        results["polynomial"] = None

    # Exponential: a * exp(b*n) + c
    try:
        popt, _ = curve_fit(exponential, sizes, times,
                            p0=[0.01, 0.01, 0], maxfev=50000)
        pred = exponential(sizes, *popt)
        ss_res = np.sum((times - pred) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        results["exponential"] = {"params": popt, "r2": r2}
    except Exception:
        results["exponential"] = None

    # Log-log linear fit (most robust polynomial indicator)
    try:
        valid = times > 0
        if valid.sum() >= 2:
            log_s = np.log(sizes[valid])
            log_t = np.log(times[valid])
            coeffs = np.polyfit(log_s, log_t, 1)
            pred_ll = coeffs[0] * log_s + coeffs[1]
            ss_res = np.sum((log_t - pred_ll) ** 2)
            ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
            r2_ll = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            results["loglog"] = {"slope": coeffs[0], "intercept": coeffs[1], "r2": r2_ll}
        else:
            results["loglog"] = None
    except Exception:
        results["loglog"] = None

    return results


# ===================================================================
# Instance family definitions
# ===================================================================

# Each family: name, generator function, list of size parameters,
# and a function that maps size param → (clauses, variables, meta)

FAMILIES = {
    "random_3sat": {
        "name": "Random 3-SAT (phase transition, ratio 4.267)",
        "sizes": [10, 20, 50, 100, 150, 200, 300, 500],
        "generate": lambda size, seed: generate_random_3sat(size, ratio=4.267, seed=seed),
        "size_label": "variables",
    },
    "planted_3sat": {
        "name": "Planted 3-SAT (guaranteed SAT, ratio 4.267)",
        "sizes": [10, 20, 50, 100, 150, 200, 300, 500],
        "generate": lambda size, seed: generate_planted_3sat(size, ratio=4.267, seed=seed),
        "size_label": "variables",
    },
    "pigeonhole": {
        "name": "Pigeonhole Principle (guaranteed UNSAT)",
        "sizes": [4, 6, 8, 10, 12, 15, 18, 20],
        "generate": lambda size, seed: generate_pigeonhole(size),
        "size_label": "n (pigeons=n+1, holes=n)",
    },
    "graph_coloring": {
        "name": "Graph 3-Coloring (planted solution)",
        "sizes": [10, 20, 30, 50, 75, 100, 150, 200],
        "generate": lambda size, seed: generate_graph_coloring(size, num_colors=3,
                                                                edge_prob=0.3, seed=seed),
        "size_label": "vertices",
    },
    "tseitin": {
        "name": "Tseitin Formulas (guaranteed UNSAT)",
        "sizes": [10, 15, 20, 30, 40, 50, 60, 75],
        "generate": lambda size, seed: generate_tseitin(size, edge_prob=0.4, seed=seed),
        "size_label": "vertices",
    },
}


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="UMACO SAT Evidence Suite — Multi-Instance-Type Scaling Analysis")
    parser.add_argument("--minisat", type=str,
                        default="C:/msys64/ucrt64/bin/minisat.exe",
                        help="Path to MiniSat binary.")
    parser.add_argument("--runs", type=int, default=5,
                        help="Runs per size per family.")
    parser.add_argument("--families", type=str, default="all",
                        help="Comma-separated family names, or 'all'.")
    parser.add_argument("--output", type=str, default="sat_evidence_results",
                        help="Output directory for results.")
    parser.add_argument("--max_iter", type=int, default=5000,
                        help="Max MACO iterations.")
    parser.add_argument("--n_ants", type=int, default=3072,
                        help="Number of ants.")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device ID.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output, exist_ok=True)

    # Select families
    if args.families == "all":
        families_to_run = list(FAMILIES.keys())
    else:
        families_to_run = [f.strip() for f in args.families.split(",")]

    # Base config with Optuna-tuned params
    base_config = MACOConfig(
        alpha=3.54879,
        beta=2.38606,
        rho=0.13814,
        initial_pheromone=0.20498,
        n_ants=args.n_ants,
        max_iterations=args.max_iter,
        clause_lr=0.24910,
        conflict_driven_learning_rate=0.21015,
        clause_weight_momentum=0.87959,
        target_entropy=0.68894,
        finishing_threshold=0.99663,
        partial_reset_stagnation=40,
        noise_std=0.11266,
        quantum_burst_interval=100,
        gpu_device_id=args.device,
        solver_path=args.minisat,
    )

    # GPU info
    try:
        import cupy as cp
        dev = cp.cuda.Device(args.device)
        gpu_name = dev.attributes.get("DeviceName", cp.cuda.runtime.getDeviceProperties(args.device)["name"].decode())
        gpu_mem = cp.cuda.Device(args.device).mem_info[1] / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    except Exception:
        print("  GPU: detection failed, proceeding anyway")

    print("=" * 72)
    print("UMACO SAT EVIDENCE SUITE")
    print(f"  Families: {', '.join(families_to_run)}")
    print(f"  Runs per size: {args.runs}")
    print(f"  Ants: {args.n_ants} | Max iter: {args.max_iter}")
    print(f"  MiniSat: {args.minisat}")
    print(f"  Output: {args.output}/")
    print("=" * 72)

    # Master CSV
    csv_path = os.path.join(args.output, "all_results.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "family", "size_param", "run", "num_vars", "num_clauses",
        "expected_sat", "maco_satisfied", "maco_total", "maco_pct",
        "maco_quality", "maco_time_s", "maco_iters",
        "minisat_status", "minisat_time_s",
        "maco_correct",
    ])

    all_family_results = {}

    for family_key in families_to_run:
        fam = FAMILIES[family_key]
        print(f"\n{'='*72}")
        print(f"FAMILY: {fam['name']}")
        print(f"Sizes ({fam['size_label']}): {fam['sizes']}")
        print(f"{'='*72}")

        family_data = {s: {"maco_times": [], "minisat_times": [],
                           "maco_pcts": [], "maco_correct": []}
                       for s in fam["sizes"]}

        run_count = 0
        total_runs = len(fam["sizes"]) * args.runs

        for size in fam["sizes"]:
            for run_idx in range(args.runs):
                run_count += 1
                seed = size * 1000 + run_idx * 7 + hash(family_key) % 10000

                # Generate instance
                clauses, variables, meta = fam["generate"](size, seed)
                num_vars = len(variables)
                num_clauses = len(clauses)
                expected = meta.get("expected_sat", "unknown")

                # Run MACO
                assignment, quality, maco_time, maco_iters = run_maco(
                    clauses, variables, base_config)

                # Verify independently
                sat_count, total_count = verify_assignment(assignment, clauses)
                pct = 100.0 * sat_count / total_count if total_count > 0 else 0.0

                # Run MiniSat
                ms_status, ms_time = run_minisat(
                    clauses, num_vars, args.minisat)

                # Determine correctness
                if expected == "SAT" and pct == 100.0:
                    correct = "TRUE"
                elif expected == "UNSAT":
                    correct = "UNSAT_INSTANCE"
                elif expected == "probabilistic":
                    correct = "SAT" if pct == 100.0 else f"{pct:.1f}%"
                else:
                    correct = f"{pct:.1f}%"

                # Store
                family_data[size]["maco_times"].append(maco_time)
                family_data[size]["minisat_times"].append(ms_time)
                family_data[size]["maco_pcts"].append(pct)
                family_data[size]["maco_correct"].append(correct)

                # CSV row
                csv_writer.writerow([
                    family_key, size, run_idx, num_vars, num_clauses,
                    expected, sat_count, total_count, f"{pct:.2f}",
                    f"{quality:.6f}", f"{maco_time:.3f}", maco_iters,
                    ms_status, f"{ms_time:.3f}", correct,
                ])
                csv_file.flush()

                # Console
                star = "*" if pct == 100.0 else " "
                print(f"  [{run_count:>4}/{total_runs}] {star} "
                      f"size={size:>4} {sat_count}/{total_count} ({pct:.1f}%) "
                      f"MACO={maco_time:.2f}s  MiniSat={ms_status}({ms_time:.2f}s)")

        # Per-family summary
        print(f"\n--- {fam['name']} Summary ---")
        print(f"{'Size':>8} {'MeanTime':>10} {'StdTime':>10} {'MeanSat%':>10} "
              f"{'MS_MeanT':>10} {'Speedup':>8}")
        print("-" * 64)

        fam_sizes = []
        fam_mean_times = []

        for size in fam["sizes"]:
            d = family_data[size]
            if not d["maco_times"]:
                continue
            mt = np.mean(d["maco_times"])
            st = np.std(d["maco_times"])
            mp = np.mean(d["maco_pcts"])
            mst = np.mean(d["minisat_times"]) if d["minisat_times"] else 0
            speedup = mst / mt if mt > 0 and mst > 0 else 0
            fam_sizes.append(size)
            fam_mean_times.append(mt)
            print(f"{size:>8} {mt:>10.3f} {st:>10.3f} {mp:>9.1f}% "
                  f"{mst:>10.3f} {speedup:>7.2f}x")

        # Scaling analysis
        if len(fam_sizes) >= 3:
            fits = fit_scaling(fam_sizes, fam_mean_times)
            print(f"\n  Scaling Analysis:")
            if fits.get("loglog"):
                ll = fits["loglog"]
                print(f"    Log-log slope: {ll['slope']:.3f}  (R² = {ll['r2']:.4f})")
                print(f"    → Empirical complexity: O(n^{ll['slope']:.2f})")
            if fits.get("polynomial"):
                p = fits["polynomial"]
                print(f"    Polynomial fit: a*n^{p['exponent']:.3f}+c  (R² = {p['r2']:.4f})")
            if fits.get("exponential"):
                e = fits["exponential"]
                print(f"    Exponential fit: R² = {e['r2']:.4f}")

            # Verdict
            poly_r2 = fits["polynomial"]["r2"] if fits.get("polynomial") else 0
            exp_r2 = fits["exponential"]["r2"] if fits.get("exponential") else 0
            if poly_r2 > exp_r2:
                print(f"    VERDICT: Polynomial fits better (R² {poly_r2:.4f} > {exp_r2:.4f})")
            else:
                print(f"    VERDICT: Exponential fits better (R² {exp_r2:.4f} > {poly_r2:.4f})")

        all_family_results[family_key] = {
            "sizes": fam_sizes,
            "mean_times": fam_mean_times,
            "data": family_data,
        }

    csv_file.close()

    # ================================================================
    # Combined report
    # ================================================================
    print("\n" + "=" * 72)
    print("COMBINED SCALING REPORT")
    print("=" * 72)

    for fk in families_to_run:
        fr = all_family_results.get(fk, {})
        if not fr.get("sizes"):
            continue
        fits = fit_scaling(fr["sizes"], fr["mean_times"]) if len(fr["sizes"]) >= 3 else {}
        slope = fits["loglog"]["slope"] if fits.get("loglog") else float("nan")
        r2 = fits["loglog"]["r2"] if fits.get("loglog") else float("nan")
        poly_r2 = fits["polynomial"]["r2"] if fits.get("polynomial") else float("nan")
        exp_r2 = fits["exponential"]["r2"] if fits.get("exponential") else float("nan")
        print(f"  {FAMILIES[fk]['name']:<50} slope={slope:.3f}  "
              f"poly_R²={poly_r2:.4f}  exp_R²={exp_r2:.4f}")

    # Write text report
    report_path = os.path.join(args.output, "scaling_report.txt")
    with open(report_path, "w") as f:
        f.write("UMACO SAT Evidence Suite — Scaling Report\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Ants: {args.n_ants} | Max iter: {args.max_iter}\n")
        f.write(f"Runs per size: {args.runs}\n\n")

        for fk in families_to_run:
            fr = all_family_results.get(fk, {})
            if not fr.get("sizes"):
                continue
            f.write(f"{'='*60}\n")
            f.write(f"{FAMILIES[fk]['name']}\n")
            f.write(f"{'='*60}\n")
            for i, size in enumerate(fr["sizes"]):
                d = fr["data"][size]
                f.write(f"  size={size:>4}  time={fr['mean_times'][i]:.3f}s  "
                        f"sat={np.mean(d['maco_pcts']):.1f}%\n")
            if len(fr["sizes"]) >= 3:
                fits = fit_scaling(fr["sizes"], fr["mean_times"])
                if fits.get("loglog"):
                    ll = fits["loglog"]
                    f.write(f"\n  Log-log slope: {ll['slope']:.3f} (R²={ll['r2']:.4f})\n")
                    f.write(f"  Empirical complexity: O(n^{ll['slope']:.2f})\n")
            f.write("\n")

    print(f"\nResults saved to {args.output}/")
    print(f"  all_results.csv — raw data ({csv_path})")
    print(f"  scaling_report.txt — text summary")
    print("Done.")


if __name__ == "__main__":
    main()
