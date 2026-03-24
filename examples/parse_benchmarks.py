#!/usr/bin/env python3
"""
UMACO Benchmark Parser & Scaling Analyzer
==========================================
Parses all existing benchmark logs in benchmarks/, extracts metrics,
fits scaling curves, and produces a complete evidence report.

No GPU required. Just reads logs and does math.
"""

import os
import re
import csv
import sys
import io
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from scipy.optimize import curve_fit
from collections import defaultdict

BENCHMARKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")

# ============================================================
# Log parsing
# ============================================================

def parse_log(filepath):
    """Extract key metrics from a single benchmark log file."""
    result = {
        "file": filepath,
        "num_vars": None,
        "num_clauses": None,
        "num_ants": None,
        "max_iter": None,
        "time_seconds": None,
        "best_quality": None,
        "satisfied": None,
        "total_clauses": None,
        "sat_pct": None,
        "unsatisfied": None,
        "quantum_bursts": 0,
        "partial_resets": 0,
        "iters_used": None,
        "minisat_status": None,
        "minisat_time": None,
        "error": None,
    }

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    except Exception as e:
        result["error"] = str(e)
        return result

    # Check for errors
    if "Traceback" in text or "No such file" in text or "Error" in text.split("\n")[3] if len(text.split("\n")) > 3 else False:
        # Check if it still has results after the error
        if "Finished in" not in text and "SOLUTION ANALYSIS" not in text:
            result["error"] = "run_failed"
            return result

    # Variables and clauses
    m = re.search(r"Generated random (\d+)-SAT with (\d+) vars, (\d+) clauses", text)
    if m:
        result["num_vars"] = int(m.group(2))
        result["num_clauses"] = int(m.group(3))

    m = re.search(r"Variables:\s*(\d+),\s*Clauses:\s*(\d+),\s*Ants:\s*(\d+),\s*Max Iterations:\s*(\d+)", text)
    if m:
        result["num_vars"] = int(m.group(1))
        result["num_clauses"] = int(m.group(2))
        result["num_ants"] = int(m.group(3))
        result["max_iter"] = int(m.group(4))

    # Command line ants
    m = re.search(r"--num_ants\s+(\d+)", text)
    if m and result["num_ants"] is None:
        result["num_ants"] = int(m.group(1))

    # Finish time
    m = re.search(r"Finished in ([\d.]+) seconds", text)
    if m:
        result["time_seconds"] = float(m.group(1))

    # Best quality
    m = re.search(r"Best quality found \(GPU-MACO\):\s*([\d.]+)", text)
    if m:
        result["best_quality"] = float(m.group(1))

    # Solution analysis
    m = re.search(r"\[SOLUTION ANALYSIS\]\s*(\d+)/(\d+)\s*clauses satisfied\s*\(([\d.]+)%\)", text)
    if m:
        result["satisfied"] = int(m.group(1))
        result["total_clauses"] = int(m.group(2))
        result["sat_pct"] = float(m.group(3))

    # Hard constraints
    m = re.search(r"\[HARD CONSTRAINTS\]\s*(\d+)\s*clauses remained unsatisfied", text)
    if m:
        result["unsatisfied"] = int(m.group(1))

    # Count quantum bursts and partial resets
    result["quantum_bursts"] = len(re.findall(r"Quantum burst triggered", text))
    result["partial_resets"] = len(re.findall(r"Partial reset triggered", text))

    # Last iteration logged
    iter_matches = re.findall(r"\[Iter (\d+)\]", text)
    if iter_matches:
        result["iters_used"] = max(int(x) for x in iter_matches)

    # Finishing phase
    if "Entering finishing phase" in text:
        m = re.search(r"Entering finishing phase at iteration (\d+)", text)
        if m:
            result["iters_used"] = int(m.group(1))

    # Full SAT found
    if "Full or near-full SAT found" in text:
        result["sat_pct"] = 100.0

    # MiniSat results
    m = re.search(r"Return code:\s*(\d+)", text)
    if m:
        rc = int(m.group(1))
        if rc == 10:
            result["minisat_status"] = "SAT"
        elif rc == 20:
            result["minisat_status"] = "UNSAT"

    m = re.search(r"Classical Solver.*?:\s*(SAT|UNSAT|TIMEOUT).*?([\d.]+)\s*seconds", text)
    if m:
        result["minisat_status"] = m.group(1)
        result["minisat_time"] = float(m.group(2))

    return result


def find_all_logs(base_dir):
    """Recursively find all .log and .txt benchmark files."""
    logs = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.startswith("results_") and (f.endswith(".log") or f.endswith(".txt")):
                logs.append(os.path.join(root, f))
    return sorted(logs)


# ============================================================
# Curve fitting
# ============================================================

def polynomial(x, a, b, c):
    return a * np.power(x, b) + c

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c


def fit_and_report(sizes, times, label=""):
    """Fit polynomial and exponential, return analysis dict."""
    sizes = np.array(sizes, dtype=float)
    times = np.array(times, dtype=float)
    results = {"label": label}

    # Log-log (most robust)
    valid = (times > 0) & (sizes > 0)
    if valid.sum() >= 2:
        log_s = np.log(sizes[valid])
        log_t = np.log(times[valid])
        coeffs = np.polyfit(log_s, log_t, 1)
        pred = coeffs[0] * log_s + coeffs[1]
        ss_res = np.sum((log_t - pred) ** 2)
        ss_tot = np.sum((log_t - np.mean(log_t)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        results["loglog_slope"] = coeffs[0]
        results["loglog_r2"] = r2

    # Polynomial: a * n^b + c
    try:
        popt, _ = curve_fit(polynomial, sizes, times,
                            p0=[1e-4, 2.5, 0],
                            bounds=([0, 0.5, -1e4], [1e8, 8.0, 1e4]),
                            maxfev=50000)
        pred = polynomial(sizes, *popt)
        ss_res = np.sum((times - pred) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        results["poly_exponent"] = popt[1]
        results["poly_r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    except Exception:
        pass

    # Exponential
    try:
        popt, _ = curve_fit(exponential, sizes, times,
                            p0=[0.01, 0.01, 0], maxfev=50000)
        pred = exponential(sizes, *popt)
        ss_res = np.sum((times - pred) ** 2)
        ss_tot = np.sum((times - np.mean(times)) ** 2)
        results["exp_r2"] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    except Exception:
        pass

    return results


# ============================================================
# Main
# ============================================================

def main():
    if not os.path.isdir(BENCHMARKS_DIR):
        print(f"ERROR: benchmarks directory not found at {BENCHMARKS_DIR}")
        sys.exit(1)

    print("=" * 72)
    print("UMACO BENCHMARK PARSER & SCALING ANALYZER")
    print(f"Scanning: {BENCHMARKS_DIR}")
    print("=" * 72)

    # Find and parse all logs
    log_files = find_all_logs(BENCHMARKS_DIR)
    print(f"Found {len(log_files)} log files")

    results = []
    errors = 0
    for lf in log_files:
        r = parse_log(lf)
        if r["error"]:
            errors += 1
        else:
            results.append(r)

    print(f"Successfully parsed: {len(results)}")
    print(f"Failed/errored: {errors}")

    if not results:
        print("No valid results to analyze.")
        return

    # Write raw CSV
    output_dir = os.path.join(os.path.dirname(BENCHMARKS_DIR), "benchmark_analysis")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "all_parsed_benchmarks.csv")
    fieldnames = ["file", "num_vars", "num_clauses", "num_ants", "max_iter",
                  "time_seconds", "best_quality", "satisfied", "total_clauses",
                  "sat_pct", "unsatisfied", "quantum_bursts", "partial_resets",
                  "iters_used", "minisat_status", "minisat_time"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\nRaw CSV written: {csv_path}")

    # ============================================================
    # Analysis
    # ============================================================

    # Group by (num_vars, num_clauses, num_ants) configuration
    configs = defaultdict(list)
    for r in results:
        key = (r["num_vars"], r["num_clauses"], r["num_ants"])
        configs[key].append(r)

    print(f"\n{'='*72}")
    print("CONFIGURATIONS FOUND")
    print(f"{'='*72}")
    print(f"{'Vars':>6} {'Clauses':>8} {'Ants':>6} {'Runs':>6} "
          f"{'MeanTime':>10} {'StdTime':>10} {'MeanSat%':>10} {'MinSat%':>10} {'MaxSat%':>10}")
    print("-" * 82)

    for key in sorted(configs.keys(), key=lambda k: (k[0] or 0, k[1] or 0, k[2] or 0)):
        runs = configs[key]
        nv, nc, na = key
        if nv is None:
            continue
        times = [r["time_seconds"] for r in runs if r["time_seconds"] is not None]
        pcts = [r["sat_pct"] for r in runs if r["sat_pct"] is not None]
        if not times and not pcts:
            continue
        mt = np.mean(times) if times else 0
        st = np.std(times) if times else 0
        mp = np.mean(pcts) if pcts else 0
        minp = np.min(pcts) if pcts else 0
        maxp = np.max(pcts) if pcts else 0
        na_str = str(na) if na is not None else "?"
        print(f"{nv:>6} {nc or '?':>8} {na_str:>6} {len(runs):>6} "
              f"{mt:>10.2f} {st:>10.2f} {mp:>9.2f}% {minp:>9.2f}% {maxp:>9.2f}%")

    # ============================================================
    # Scaling analysis: group by problem size (vars), take mean time
    # ============================================================

    by_vars = defaultdict(list)
    for r in results:
        if r["time_seconds"] is not None and r["num_vars"] is not None:
            by_vars[r["num_vars"]].append(r)

    if len(by_vars) >= 2:
        print(f"\n{'='*72}")
        print("SCALING ANALYSIS — Time vs Problem Size")
        print(f"{'='*72}")

        sizes = sorted(by_vars.keys())
        mean_times = []
        for s in sizes:
            runs = by_vars[s]
            ts = [r["time_seconds"] for r in runs if r["time_seconds"]]
            mean_times.append(np.mean(ts) if ts else 0)

        print(f"\n{'Vars':>8} {'Runs':>6} {'MeanTime':>12} {'MeanSat%':>10}")
        print("-" * 40)
        for i, s in enumerate(sizes):
            runs = by_vars[s]
            pcts = [r["sat_pct"] for r in runs if r["sat_pct"] is not None]
            mp = np.mean(pcts) if pcts else 0
            print(f"{s:>8} {len(runs):>6} {mean_times[i]:>12.3f} {mp:>9.2f}%")

        if len(sizes) >= 3:
            fits = fit_and_report(sizes, mean_times, "all_runs")
            print(f"\n  Log-log slope: {fits.get('loglog_slope', float('nan')):.3f} "
                  f"(R² = {fits.get('loglog_r2', float('nan')):.4f})")
            print(f"  -> Empirical complexity: O(n^{fits.get('loglog_slope', 0):.2f})")
            if "poly_exponent" in fits:
                print(f"  Polynomial fit exponent: {fits['poly_exponent']:.3f} "
                      f"(R² = {fits.get('poly_r2', 0):.4f})")
            if "exp_r2" in fits:
                print(f"  Exponential fit R²: {fits['exp_r2']:.4f}")

            poly_r2 = fits.get("poly_r2", 0)
            exp_r2 = fits.get("exp_r2", 0)
            print(f"\n  VERDICT: {'Polynomial' if poly_r2 >= exp_r2 else 'Exponential'} "
                  f"fits better (poly R²={poly_r2:.4f} vs exp R²={exp_r2:.4f})")

    # ============================================================
    # MiniSat comparison
    # ============================================================

    minisat_runs = [r for r in results if r["minisat_time"] is not None]
    if minisat_runs:
        print(f"\n{'='*72}")
        print("MINISAT COMPARISON")
        print(f"{'='*72}")
        print(f"{'Vars':>6} {'MACO_Time':>12} {'MS_Time':>12} {'Speedup':>10} {'MS_Status':>10} {'MACO_Sat%':>10}")
        print("-" * 66)
        for r in sorted(minisat_runs, key=lambda x: x["num_vars"] or 0):
            speedup = r["minisat_time"] / r["time_seconds"] if r["time_seconds"] and r["time_seconds"] > 0 else 0
            print(f"{r['num_vars']:>6} {r['time_seconds']:>12.2f} {r['minisat_time']:>12.2f} "
                  f"{speedup:>9.2f}x {r['minisat_status']:>10} {r['sat_pct'] or 0:>9.2f}%")

    # ============================================================
    # Summary statistics
    # ============================================================

    print(f"\n{'='*72}")
    print("OVERALL SUMMARY")
    print(f"{'='*72}")
    all_pcts = [r["sat_pct"] for r in results if r["sat_pct"] is not None]
    all_times = [r["time_seconds"] for r in results if r["time_seconds"] is not None]
    full_sat = sum(1 for p in all_pcts if p == 100.0)
    above_99 = sum(1 for p in all_pcts if p >= 99.0)
    above_95 = sum(1 for p in all_pcts if p >= 95.0)

    print(f"  Total valid runs: {len(results)}")
    print(f"  Runs with timing data: {len(all_times)}")
    print(f"  Runs with satisfaction data: {len(all_pcts)}")
    print(f"  100% satisfied: {full_sat} ({100*full_sat/len(all_pcts):.1f}%)" if all_pcts else "")
    print(f"  >=99% satisfied: {above_99} ({100*above_99/len(all_pcts):.1f}%)" if all_pcts else "")
    print(f"  >=95% satisfied: {above_95} ({100*above_95/len(all_pcts):.1f}%)" if all_pcts else "")
    if all_pcts:
        print(f"  Mean satisfaction: {np.mean(all_pcts):.2f}%")
        print(f"  Min satisfaction: {np.min(all_pcts):.2f}%")
    if all_times:
        print(f"  Mean time: {np.mean(all_times):.2f}s")
        print(f"  Total compute time: {np.sum(all_times):.1f}s ({np.sum(all_times)/3600:.1f} hours)")

    # ============================================================
    # Write full report
    # ============================================================

    report_path = os.path.join(output_dir, "benchmark_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("UMACO BENCHMARK ANALYSIS REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Total logs found: {len(log_files)}\n")
        f.write(f"Successfully parsed: {len(results)}\n")
        f.write(f"Failed/errored: {errors}\n\n")

        f.write("CONFIGURATIONS:\n")
        for key in sorted(configs.keys(), key=lambda k: (k[0] or 0, k[1] or 0, k[2] or 0)):
            runs = configs[key]
            nv, nc, na = key
            if nv is None:
                continue
            times = [r["time_seconds"] for r in runs if r["time_seconds"] is not None]
            pcts = [r["sat_pct"] for r in runs if r["sat_pct"] is not None]
            if times or pcts:
                mt = np.mean(times) if times else 0
                mp = np.mean(pcts) if pcts else 0
                f.write(f"  {nv} vars, {nc} clauses, {na or '?'} ants: "
                        f"{len(runs)} runs, mean_time={mt:.2f}s, mean_sat={mp:.2f}%\n")

        if len(by_vars) >= 2:
            f.write(f"\nSCALING:\n")
            sizes = sorted(by_vars.keys())
            for s in sizes:
                runs = by_vars[s]
                ts = [r["time_seconds"] for r in runs if r["time_seconds"]]
                pcts = [r["sat_pct"] for r in runs if r["sat_pct"] is not None]
                f.write(f"  {s} vars: {len(runs)} runs, "
                        f"mean_time={np.mean(ts) if ts else 0:.2f}s, "
                        f"mean_sat={np.mean(pcts) if pcts else 0:.2f}%\n")

            if len(sizes) >= 3:
                mean_times = [np.mean([r["time_seconds"] for r in by_vars[s] if r["time_seconds"]]) for s in sizes]
                fits = fit_and_report(sizes, mean_times)
                if "loglog_slope" in fits:
                    f.write(f"\n  Log-log slope: {fits['loglog_slope']:.3f} (R²={fits['loglog_r2']:.4f})\n")
                    f.write(f"  Empirical complexity: O(n^{fits['loglog_slope']:.2f})\n")

        f.write(f"\nSUMMARY:\n")
        f.write(f"  Total valid runs: {len(results)}\n")
        if all_pcts:
            f.write(f"  Mean satisfaction: {np.mean(all_pcts):.2f}%\n")
            f.write(f"  100% satisfied: {full_sat}/{len(all_pcts)}\n")
            f.write(f"  >=95% satisfied: {above_95}/{len(all_pcts)}\n")
        if all_times:
            f.write(f"  Total GPU compute: {np.sum(all_times)/3600:.1f} hours\n")

    print(f"\nFull report: {report_path}")
    print(f"Raw CSV: {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
