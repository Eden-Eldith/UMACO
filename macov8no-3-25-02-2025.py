#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
macov8no-1-24-02-2025.py Ultimate Edition - GPU-Accelerated SAT Solver (ZVSS Self-Optimization Integrated, Extended GPU Offload)
=============================================================================================================
NO ILLUSTRATIVE EXAMPLES ARE PROVIDED. USE AT YOUR OWN RISK.

Major Change from Previous Versions:
  - The clause coverage counting and clause-weight updating logic are moved fully to GPU to offload more CPU work.

Logs include both new and legacy messages:
  "Running MACO with generated instance"
  "Generated random 3-SAT instance:"
  "=== Starting GPU-MACO SAT Solver ==="
  "[SOLUTION ANALYSIS] ..."
  "Return code: ..."

All parameters are controlled via an expanded MACOConfig dataclass that includes:
  alpha, beta, rho, initial_pheromone, n_ants, max_iterations, partial_reset_stagnation,
  quantum_burst_interval, finishing_threshold, target_entropy, conflict_driven_learning_rate,
  clause_weight_momentum, noise_std, etc.
"""

import sys
import math
import time
import argparse
import random
import logging
import subprocess
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx


def _resolve_gpu_backend(module_name: str = "macov8no-3-25-02-2025"):
    """Resolve the GPU backend, enforcing CuPy unless the global override is enabled."""
    allow_cpu = os.getenv("UMACO_ALLOW_CPU", "0") == "1"
    module_logger = logging.getLogger(module_name)

    try:
        import cupy as _cp  # type: ignore
    except ImportError as exc:
        if not allow_cpu:
            raise RuntimeError(
                "macov8no-3-25-02-2025 requires CuPy for GPU execution. Install cupy-cudaXX or set UMACO_ALLOW_CPU=1 to acknowledge CPU fallback."
            ) from exc
        module_logger.warning(
            "CuPy is not installed; running in NumPy compatibility mode because UMACO_ALLOW_CPU=1."
        )
        return np, False

    try:
        _cp.cuda.runtime.getDeviceCount()
        _cp.cuda.nvrtc.getVersion()
    except Exception as exc:
        if not allow_cpu:
            raise RuntimeError(
                "CuPy is installed but CUDA runtime is unhealthy (missing nvrtc or CUDA device). Install the matching toolkit or set UMACO_ALLOW_CPU=1 to override."
            ) from exc
        module_logger.warning(
            "CUDA runtime issue detected (%s); running in NumPy compatibility mode because UMACO_ALLOW_CPU=1.",
            exc,
        )
        return np, False

    return _cp, True


cp, GPU_AVAILABLE = _resolve_gpu_backend(__name__)

# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class MACOConfig:
    """
    MACO configuration parameters, extended with ZVSS-like self-optimizing logic.
    No examples are provided. Use with caution.
    """
    # Core ACO parameters
    alpha: float = 3.54879
    beta: float = 2.38606
    rho: float = 0.13814
    initial_pheromone: float = 0.20498
    n_ants: int = 3072
    max_iterations: int = 5000

    # Clause weighting and adaptation
    clause_lr: float = 0.24910
    conflict_driven_learning_rate: float = 0.21015
    clause_weight_momentum: float = 0.87959
    clause_weight_decay: float = 0.0  # If needed for long runs

    # Entropy, partial resets, quantum bursts, advanced heuristics
    target_entropy: float = 0.68894
    finishing_threshold: float = 0.99663
    partial_reset_stagnation: int = 40
    noise_std: float = 0.11266
    quantum_burst_interval: float = 100.59397

    # GPU, logging, parallelization
    parallel_ants: bool = True
    num_workers: int = 8
    solver_path: str = "C:/msys64/ucrt64/bin/minisat.exe"
    gpu_device_id: int = 0

    # Random SAT generation defaults (if no CNF provided)
    num_vars_gen: int = 500
    num_clauses_gen: int = 2500
    clause_size_gen: int = 3
    n_gpu_layers: int = 60
    gpu_threads_per_block: int = 256

# =============================================================================
# Utility Functions (Parsing, Array Building, Random SAT Generation)
# =============================================================================

def parse_dimacs_cnf(filename: str) -> Tuple[int, int, List[List[int]]]:
    clauses = []
    num_vars = 0
    num_clauses = 0
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('c'):
                continue
            if line.startswith('p'):
                parts = line.split()
                if len(parts) >= 4:
                    num_vars = int(parts[2])
                    num_clauses = int(parts[3])
            else:
                parts = line.split()
                lits = list(map(int, parts))
                if lits and lits[-1] == 0:
                    lits.pop()
                if lits:
                    clauses.append(lits)
    return num_vars, num_clauses, clauses


def build_clause_array(clauses: List[List[int]], max_clause_size: int) -> np.ndarray:
    num_clauses = len(clauses)
    clause_array = np.zeros((num_clauses, max_clause_size), dtype=np.int32)
    for i, clause in enumerate(clauses):
        for j, lit in enumerate(clause):
            clause_array[i, j] = lit
    return clause_array


def compute_max_clause_size(clauses: List[List[int]]) -> int:
    return max(len(c) for c in clauses) if clauses else 0


def generate_sat_formula(num_vars: int, num_clauses: int, clause_size: int = 3) -> Tuple[List[Tuple[int, ...]], List[int]]:
    """
    Generates a random k-SAT formula for demonstration. No examples included beyond this placeholder.
    """
    variables = list(range(1, num_vars + 1))
    formula = []
    for _ in range(num_clauses):
        chosen = random.sample(variables, k=clause_size)
        clause = [v if random.random() < 0.5 else -v for v in chosen]
        formula.append(tuple(clause))
    return formula, variables


def write_dimacs(formula: List[Tuple[int, ...]], variables: List[int], filepath: str) -> None:
    num_vars = len(variables)
    num_clauses = len(formula)
    with open(filepath, "w") as f:
        f.write(f"p cnf {num_vars} {num_clauses}\n")
        for clause in formula:
            line = " ".join(str(lit) for lit in clause) + " 0\n"
            f.write(line)


def run_classical_solver(
    clauses: List[Tuple[int, ...]],
    variables: List[int],
    solver_path: str = "minisat",
    timeout: int = 3600
) -> Tuple[str, Dict[int, bool], float]:
    """
    Invokes a classical SAT solver for final cross-check. Use with caution.
    """
    import tempfile

    start_time = time.time()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cnf", delete=False) as tmp:
        dimacs_file = tmp.name
        write_dimacs(clauses, variables, dimacs_file)

    try:
        cmd = [solver_path, dimacs_file]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=timeout)
        solver_output = result.stdout
        solver_err = result.stderr
        runtime = time.time() - start_time
        print("\n--------------------------------------")
        print("CLASSICAL SOLVER INVOCATION:")
        print("Command:", " ".join(cmd))
        print(f"Return code: {result.returncode}")
        print("Solver STDOUT:\n", solver_output)
        print("Solver STDERR:\n", solver_err)
        print("--------------------------------------\n")
        assignment = {}
        is_sat = any("SATISFIABLE" in ln.upper() for ln in solver_output.splitlines())
        for ln in solver_output.splitlines():
            if ln.strip().startswith("v "):
                for lit_str in ln.strip().split()[1:]:
                    if lit_str == '0':
                        continue
                    lit = int(lit_str)
                    assignment[abs(lit)] = (lit > 0)
        status = "SAT" if assignment else "UNSAT"
        return status, assignment, runtime
    except subprocess.TimeoutExpired:
        runtime = time.time() - start_time
        print("[Classical Solver] Timed out.")
        return "TIMEOUT", {}, runtime
    except Exception as e:
        runtime = time.time() - start_time
        print("[Classical Solver] Exception:", e)
        return "ERROR", {}, runtime
    finally:
        if os.path.exists(dimacs_file):
            os.remove(dimacs_file)


# =============================================================================
# GPU Kernels
# =============================================================================

evaluate_kernel = cp.RawKernel(r'''
extern "C" __global__
void evaluate_assignments_kernel(
    const int* clause_array,
    const int num_clauses,
    const int max_clause_size,
    const int num_vars,
    const int num_ants,
    const int* assignments,
    const float* clause_weights,
    float* out_quality
)
{
    int ant_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ant_id >= num_ants) return;

    float sum_weights = 0.0f;
    float sum_satisfied = 0.0f;

    for (int c = 0; c < num_clauses; c++) {
        float w = clause_weights[c];
        bool clause_satisfied = false;

        for (int j = 0; j < max_clause_size; j++) {
            int lit = clause_array[c * max_clause_size + j];
            if (lit == 0) break;

            int var_idx = abs(lit) - 1;
            bool assigned_true = (assignments[ant_id * num_vars + var_idx] == 1);

            if ((lit > 0 && assigned_true) || (lit < 0 && !assigned_true)) {
                clause_satisfied = true;
                break;
            }
        }

        sum_weights += w;
        if (clause_satisfied) {
            sum_satisfied += w;
        }
    }

    out_quality[ant_id] = (sum_weights > 1e-9f) ? (sum_satisfied / sum_weights) : 0.0f;
}
''', 'evaluate_assignments_kernel')


pheromone_evaporation_kernel = cp.RawKernel(r'''
extern "C" __global__
void pheromone_evaporation_kernel(
    float* pheromones,
    const float rho,
    const int num_vars
)
{
    int v = blockDim.x * blockIdx.x + threadIdx.x;
    if (v >= num_vars) return;

    // Evaporate both false and true pheromones
    pheromones[v*2 + 0] = fmaxf(fminf((1.0f - rho) * pheromones[v*2 + 0], 10.0f), 0.001f);
    pheromones[v*2 + 1] = fmaxf(fminf((1.0f - rho) * pheromones[v*2 + 1], 10.0f), 0.001f);
}
''', 'pheromone_evaporation_kernel')


pheromone_update_kernel = cp.RawKernel(r'''
extern "C" __global__
void pheromone_update_kernel(
    float* pheromones,
    const float* qualities,
    const int* assignments,
    const float alpha,
    const float rho,
    const int num_vars,
    const int num_ants
)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_ants) return;

    float q = qualities[idx];
    float scaledQ = powf(q, 1.5f);

    // Each thread processes exactly one ant. We still need to update all variables for that ant.
    // This is a partial approach. If performance is an issue, consider a 2D kernel. 
    for (int v = 0; v < num_vars; v++) {
        int chosen_val = assignments[idx * num_vars + v];

        // Calculate deposit amount
        float deposit = alpha * scaledQ;

        // Atomic deposit - thread-safe pheromone updates
        if (chosen_val == 1) {
            atomicAdd(&pheromones[v*2 + 1], deposit);
        } else {
            atomicAdd(&pheromones[v*2 + 0], deposit);
        }
    }
}
''', 'pheromone_update_kernel')


build_assignments_kernel = cp.RawKernel(r'''
extern "C" __global__
void build_assignments_kernel(
    const float* pheromones,
    const float alpha,
    const float beta,
    const float noise_std,
    const int num_vars,
    const int num_ants,
    const unsigned long long seed,
    int* out_assignments
)
{
    int ant_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ant_id >= num_ants) return;

    // Unique RNG seed for each ant
    unsigned long long rng = seed ^ ((unsigned long long)(ant_id+1) * 2685821657736338717ULL);

    for (int v = 0; v < num_vars; v++) {
        // xorshift64-like step
        rng ^= (rng << 13); rng ^= (rng >> 7); rng ^= (rng << 17);

        float r01 = (float)(rng & 0xFFFFFFFF) / 4294967295.0f;

        float phF = pheromones[v*2+0];
        float phT = pheromones[v*2+1];

        float aF = powf(phF + 1e-9f, alpha);
        float aT = powf(phT + 1e-9f, alpha);

        // ZVSS-inspired noise injection
        float noise = 1.0f;
        if (noise_std > 0.0f) {
            float shift = (r01 - 0.5f) * 2.0f * noise_std;
            noise += shift;
            if (noise < 0.01f) noise = 0.01f;
        }

        float pT = (aT * noise) / (aF + aT + 1e-9f);
        pT = fmaxf(fminf(pT, 0.9999f), 0.0001f);

        int val = (r01 < pT) ? 1 : 0;
        out_assignments[ant_id * num_vars + v] = val;
    }
}
''', 'build_assignments_kernel')


gpu_local_search_kernel = cp.RawKernel(r'''
extern "C" __global__
void gpu_local_search_kernel(
    const int* clause_array,
    const float* clause_weights,
    const int num_clauses,
    const int max_clause_size,
    const int num_vars,
    const int num_ants,
    const int max_flips,
    const float temperature,
    int* assignments
)
{
    int ant_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (ant_id >= num_ants) return;

    unsigned long long rng = 88172645463325252ULL ^ ((unsigned long long)(ant_id+1) * 1099511628211ULL);

    for (int flip = 0; flip < max_flips; flip++) {
        // Weighted random choice of one unsatisfied clause
        int local_unsat_idx = -1;
        int unsat_found = 0;

        for (int c = 0; c < num_clauses; c++) {
            bool clause_satisfied = false;
            for (int j = 0; j < max_clause_size; j++) {
                int lit = clause_array[c * max_clause_size + j];
                if (lit == 0) break;
                int var_idx = abs(lit) - 1;
                bool assigned_true = (assignments[ant_id * num_vars + var_idx] == 1);
                if ((lit > 0 && assigned_true) || (lit < 0 && !assigned_true)) {
                    clause_satisfied = true;
                    break;
                }
            }
            if (!clause_satisfied) {
                // reservoir sampling logic
                unsat_found++;
                rng ^= (rng << 13); rng ^= (rng >> 7); rng ^= (rng << 17);
                if ((rng & 0xFFFFFFFF) % unsat_found == 0) {
                    local_unsat_idx = c;
                }
            }
        }

        if (unsat_found == 0) {
            // All satisfied
            break;
        }

        if (local_unsat_idx < 0) {
            break;
        }

        // Flip a variable from the chosen unsatisfied clause
        int var_candidates[64];
        int var_count = 0;

        for (int j = 0; j < max_clause_size && var_count < 64; j++) {
            int lit = clause_array[local_unsat_idx * max_clause_size + j];
            if (lit == 0) break;
            var_candidates[var_count++] = abs(lit) - 1;
        }

        if (var_count == 0) continue;

        rng ^= (rng << 13); rng ^= (rng >> 7); rng ^= (rng << 17);
        unsigned long long pick = rng & 0xFFFFFFFF;
        int pick_idx = (int)(pick % var_count);
        int flip_var = var_candidates[pick_idx];

        // Evaluate acceptance with weighted difference
        float current_score = 0.0f;
        float new_score = 0.0f;

        int old_val = assignments[ant_id * num_vars + flip_var];
        int test_val = 1 - old_val;

        for (int c2 = 0; c2 < num_clauses; c2++) {
            float w = clause_weights[c2];
            bool sat_before = false;
            bool sat_after = false;

            // old assignment
            for (int j = 0; j < max_clause_size; j++) {
                int cl_lit = clause_array[c2 * max_clause_size + j];
                if (cl_lit == 0) break;
                int cvar_idx = abs(cl_lit) - 1;
                bool cval_before = (assignments[ant_id * num_vars + cvar_idx] == 1);
                if ((cl_lit > 0 && cval_before) || (cl_lit < 0 && !cval_before)) {
                    sat_before = true;
                    break;
                }
            }

            // prospective new assignment
            for (int j = 0; j < max_clause_size; j++) {
                int cl_lit = clause_array[c2 * max_clause_size + j];
                if (cl_lit == 0) break;
                int cvar_idx = abs(cl_lit) - 1;
                bool cval_after = (cvar_idx == flip_var) ? (test_val == 1)
                                                        : (assignments[ant_id * num_vars + cvar_idx] == 1);
                if ((cl_lit > 0 && cval_after) || (cl_lit < 0 && !cval_after)) {
                    sat_after = true;
                    break;
                }
            }

            if (sat_before) current_score += w;
            if (sat_after)  new_score += w;
        }

        float diff = new_score - current_score;
        if (diff >= 0.0f) {
            // Flip improves or equals
            assignments[ant_id * num_vars + flip_var] = test_val;
        } else {
            // Metropolis acceptance
            float T = (temperature < 1e-6f) ? 1e-6f : temperature;
            float prob = expf(diff / T);
            rng ^= (rng << 13); rng ^= (rng >> 7); rng ^= (rng << 17);
            float r01 = (float)(rng & 0xFFFFFFFF) / 4294967295.0f;
            if (r01 < prob) {
                assignments[ant_id * num_vars + flip_var] = test_val;
            }
        }
    }
}
''', 'gpu_local_search_kernel')


# ------------------------------ NEW GPU Kernels for Coverage & Weight Updates ------------------------------
coverage_count_kernel = cp.RawKernel(r'''
extern "C" __global__
void coverage_count_kernel(
    const int* clause_array,
    const int num_clauses,
    const int max_clause_size,
    const int num_vars,
    const int num_ants,
    const int* assignments,
    int* coverage_count
)
{
    // One thread per (ant,clause) pair
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_ants * num_clauses) return;

    int ant_id = idx / num_clauses;
    int c      = idx % num_clauses;

    bool satisfied = false;
    for (int j = 0; j < max_clause_size; j++) {
        int lit = clause_array[c * max_clause_size + j];
        if (lit == 0) break;

        int var_idx = abs(lit) - 1;
        bool assigned_true = (assignments[ant_id * num_vars + var_idx] == 1);
        if ((lit > 0 && assigned_true) || (lit < 0 && !assigned_true)) {
            satisfied = true;
            break;
        }
    }
    if (satisfied) {
        atomicAdd(&coverage_count[c], 1);
    }
}
''', 'coverage_count_kernel')


stubb_update_kernel = cp.RawKernel(r'''
extern "C" __global__
void stubb_update_kernel(
    const int* coverage_count,
    const int num_clauses,
    const int num_ants,
    float* clause_stubbornness,
    float* clause_weights,
    const float conflict_lr,
    const float momentum
)
{
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= num_clauses) return;

    int cov = coverage_count[c];
    if (cov == 0) {
        // Completely unsatisfied by all ants => conflict
        clause_stubbornness[c] += conflict_lr;
    } else {
        float frac = (float)cov / (float)num_ants;
        clause_stubbornness[c] = momentum * clause_stubbornness[c] + (1.0f - momentum) * (1.0f - frac);
    }
    clause_weights[c] = 1.0f + 5.0f * clause_stubbornness[c];
}
''', 'stubb_update_kernel')


# =============================================================================
# MACO System Class (with extended GPU Offload)
# =============================================================================

class MACOSystem:
    def __init__(self, config: MACOConfig):
        self.config = config
        self.clauses: List[List[int]] = []
        self.num_vars = 0
        self.num_ants = config.n_ants
        self.max_iter = config.max_iterations

        # Imported from MACO-ZVSS approach
        self.alpha = config.alpha
        self.beta = config.beta
        self.rho = config.rho
        self.noise_std = config.noise_std
        self.target_entropy = config.target_entropy
        self.finishing_threshold = config.finishing_threshold
        self.partial_reset_stagnation = config.partial_reset_stagnation
        self.local_search_flips = 20
        self.logging_interval = 50

        self.clause_lr = config.conflict_driven_learning_rate
        self.clause_weight_momentum = config.clause_weight_momentum

        device_id = config.gpu_device_id
        if GPU_AVAILABLE:
            cp.cuda.Device(device_id).use()
        else:
            logging.warning(
                "GPU backend unavailable; proceeding in CPU compatibility mode because UMACO_ALLOW_CPU=1."
            )

        # Graph placeholders (kept for legacy)
        self.graph = nx.DiGraph()
        self.pheromone_matrix: Dict[Tuple[str, str], float] = {}

        # GPU arrays for the problem
        self.clause_array_gpu: Optional[Any] = None
        self.clause_weights_gpu: Optional[Any] = None
        self.clause_stubbornness_gpu: Optional[Any] = None
        self.pheromones_gpu: Optional[Any] = None
        self.assignments_gpu: Optional[Any] = None
        self.qualities_gpu: Optional[Any] = None
        self.coverage_count_gpu: Optional[Any] = None

        self.best_quality_so_far = 0.0
        self.best_assignment = None
        self.last_improvement_iter = 0

    def initialize_search_space(self, formula: List[List[int]], variables: List[int]) -> None:
        """
        Initialize the GPU arrays and data structures for the SAT instance.
        """
        self.clauses = formula
        self.num_vars = len(variables)
        self.sat_vars = variables
        self.num_clauses = len(formula)

        # Clause weighting array
        self.clause_weights_gpu = cp.full((self.num_clauses,), 1.0, dtype=cp.float32)
        self.clause_stubbornness_gpu = cp.zeros((self.num_clauses,), dtype=cp.float32)

        # Pheromones: shape (num_vars, 2) -> (phFalse, phTrue)
        init_pheromone = self.config.initial_pheromone
        self.pheromones_gpu = cp.full((self.num_vars, 2), init_pheromone, dtype=cp.float32)

        # Ant assignments and qualities
        self.assignments_gpu = cp.zeros((self.config.n_ants, self.num_vars), dtype=cp.int32)
        self.qualities_gpu = cp.zeros((self.num_ants,), dtype=cp.float32)

        # coverage counts for GPU-based clause coverage
        self.coverage_count_gpu = cp.zeros((self.num_clauses,), dtype=cp.int32)

        # Build and send clause array to GPU
        self.max_clause_size = compute_max_clause_size(formula)
        clause_array_np = build_clause_array(formula, self.max_clause_size).astype(np.int32)
        self.clause_array_gpu = cp.asarray(clause_array_np)

        self._init_sat_graph()

    def _init_sat_graph(self) -> None:
        """
        Builds a trivial layered graph for legacy logging/structure only.
        """
        self.graph.clear()
        self.pheromone_matrix.clear()
        self.graph.add_node("start", type="sat_start")
        prev_layer_nodes = ["start"]
        for i, var in enumerate(self.sat_vars):
            layer_nodes = []
            for val in [True, False]:
                node_id = f"var_{var}_L{i}_{val}"
                self.graph.add_node(node_id, type="sat_decision", var=var, val=val, layer=i)
                layer_nodes.append(node_id)
            for pln in prev_layer_nodes:
                for ln in layer_nodes:
                    self.graph.add_edge(pln, ln)
                    self.pheromone_matrix[(pln, ln)] = self.config.initial_pheromone
            prev_layer_nodes = layer_nodes

    def solve(self) -> Tuple[Optional[np.ndarray], float]:
        """
        Runs the MACO+ZVSS self-optimizing loop to tackle the SAT problem, fully on GPU for coverage & weighting.
        Returns the best assignment found and the best quality.
        """
        logging.info("=== Starting MACO Ultimate Edition ===")
        logging.info("=== Starting GPU-MACO SAT Solver ===")
        logging.info(f"Variables: {self.num_vars}, Clauses: {self.num_clauses}, Ants: {self.num_ants}, "
                     f"Max Iterations: {self.max_iter}")

        start_time = time.time()
        best_global_quality = 0.0
        best_global_assignment = None

        for it in range(self.config.max_iterations):
            # Adapt parameters each iteration (ZVSS-inspired)
            self._adapt_parameters(it)

            # Build solutions on GPU
            seed = random.getrandbits(64)
            self._launch_build_assignments(seed)

            # Optional local search
            if it > 0:
                T = max(1e-3, 0.1 * (1.0 - float(it) / self.max_iter))
                self._launch_local_search(top_fraction=0.2, temperature=T)

            # Evaluate solutions
            avg_q, best_q, best_idx = self._launch_evaluation()

            # Track best
            if best_q > best_global_quality:
                best_global_quality = best_q
                best_assignment = self.assignments_gpu[best_idx, :]
                best_global_assignment = best_assignment
                self.last_improvement_iter = it

            # Update clause weighting on GPU
            self._update_clause_weights_gpu()

            # Update pheromones
            self._launch_pheromone_update()

            # Possibly trigger quantum burst
            self._check_quantum_burst(it, best_global_quality)

            # If near finishing threshold, do a final local search
            if best_global_quality >= self.finishing_threshold:
                logging.info(f">>> Entering finishing phase at iteration {it}")
                self._launch_local_search(top_fraction=1.0, temperature=1e-5)
                _, final_q, best_idx2 = self._launch_evaluation()
                if final_q > best_global_quality:
                    best_global_quality = final_q
                    best_assignment = self.assignments_gpu[best_idx2, :]
                    best_global_assignment = best_assignment
                if final_q >= 0.999999:
                    logging.info(">>> Full or near-full SAT found. Exiting.")
                    break

            # Partial resets if no improvement
            if (it - self.last_improvement_iter) >= self.partial_reset_stagnation:
                logging.info(f"Partial reset triggered at iteration {it}.")
                self._apply_partial_pheromone_reset()
                self.last_improvement_iter = it

            # Periodic logging
            if (it % self.logging_interval) == 0:
                cur_entropy = self._compute_entropy()
                logging.info(f"[Iter {it}] best_q={best_global_quality:.4f}, "
                             f"avg_q={avg_q:.4f}, "
                             f"entropy={cur_entropy:.3f}, "
                             f"alpha={self.alpha:.3f}, rho={self.rho:.3f}, "
                             f"noise_std={self.config.noise_std:.4f}")

            # If effectively at 100% satisfaction, break early
            if abs(1.0 - best_global_quality) < 1e-9:
                break

        total_time = time.time() - start_time
        logging.info(f"=== Finished in {total_time:.2f} seconds ===")
        return best_global_assignment, best_global_quality

    # -------------------------------------------------------------------------
    # ZVSS-Inspired Parameter Adaptation, Partial Resets, Quantum Bursts
    # -------------------------------------------------------------------------

    def _compute_entropy(self) -> float:
        """
        Computes the average entropy across pheromone distributions.
        """
        p = self.pheromones_gpu / cp.sum(self.pheromones_gpu, axis=1, keepdims=True)
        epsilon = 1e-9
        ent = -cp.sum(p * cp.log2(p + epsilon), axis=1)
        return float(cp.mean(ent))

    def _adapt_parameters(self, iteration: int) -> None:
        """
        Self-optimizing logic adapted from MACO-ZVSS:
        - Entropy-based adjustments to alpha, rho, noise
        - Periodic sinusoidal variation in beta
        """
        current_entropy = self._compute_entropy()
        ent_diff = self.config.target_entropy - current_entropy

        # Noise adjusts inversely with entropy difference
        self.config.noise_std = max(0.01, min(0.1, self.config.noise_std + 0.01 * (-ent_diff)))

        # alpha and rho shift based on ent_diff
        self.alpha = max(2.0, min(6.0, self.alpha + 0.02 * ent_diff))
        self.rho   = max(0.01, min(0.5, self.rho - 0.01 * ent_diff))

        # Beta modulated by iteration-based sine wave
        cycle = 2.0 * math.pi * (float(iteration) / self.config.max_iterations)
        sin_mod = 0.5 + 0.5 * math.sin(cycle)
        self.beta = max(1.0, min(3.0, 1.8 + 0.2 * sin_mod))

    def _apply_partial_pheromone_reset(self) -> None:
        """
        Partially resets pheromones below a certain quantile, per ZVSS logic.
        """
        ph = self.pheromones_gpu.ravel()
        ph_cpu = ph
        cutoff_idx = int(len(ph_cpu) * 0.05)
        cutoff_idx = max(1, cutoff_idx)

        sorted_ph = np.sort(ph_cpu)[::-1]
        if cutoff_idx <= len(sorted_ph):
            cutoff_val = sorted_ph[cutoff_idx - 1]
        else:
            cutoff_val = 0.0

        rng = np.random.default_rng()
        for i in range(len(ph_cpu)):
            if ph_cpu[i] < cutoff_val:
                ph_cpu[i] = 0.01 + 0.01 * rng.random()

        ph = cp.asarray(ph_cpu).reshape(self.pheromones_gpu.shape)
        self.pheromones_gpu = ph

    def _check_quantum_burst(self, iteration: int, best_q: float) -> None:
        """
        If stagnating, forcibly amplify noise and reduce alpha, similar to quantum leaps in MACO-ZVSS.
        """
        qb_interval = int(self.config.quantum_burst_interval)
        if qb_interval <= 0:
            return

        if (iteration % qb_interval) == (qb_interval - 1):
            if best_q < 0.999:
                old_noise = self.noise_std
                self.noise_std = min(0.5, self.noise_std * 3.0)
                self.alpha = max(1.0, self.alpha * 0.7)
                logging.info(f"Quantum burst triggered: noise_std from {old_noise:.3f} "
                             f"to {self.noise_std:.3f}, alpha now {self.alpha:.3f}")

    # -------------------------------------------------------------------------
    # Core ACO Steps (Build, Evaluate, Update)
    # -------------------------------------------------------------------------

    def _launch_build_assignments(self, seed: int) -> None:
        block_size = 128
        grid_size = (self.num_ants + block_size - 1) // block_size
        build_assignments_kernel(
            (grid_size,), (block_size,),
            (
                self.pheromones_gpu,
                np.float32(self.alpha),
                np.float32(self.beta),
                np.float32(self.config.noise_std),
                np.int32(self.num_vars),
                np.int32(self.num_ants),
                np.uint64(seed),
                self.assignments_gpu
            )
        )

    def _launch_evaluation(self) -> Tuple[float, float, int]:
        block_size = 128
        grid_size = (self.num_ants + block_size - 1) // block_size
        self.qualities_gpu.fill(0)

        evaluate_kernel(
            (grid_size,), (block_size,),
            (
                self.clause_array_gpu,
                np.int32(self.num_clauses),
                np.int32(self.max_clause_size),
                np.int32(self.num_vars),
                np.int32(self.num_ants),
                self.assignments_gpu,
                self.clause_weights_gpu,
                self.qualities_gpu
            )
        )
        cp.cuda.Stream.null.synchronize()

        qualities_cpu = self.qualities_gpu
        avg_q = float(np.mean(qualities_cpu))
        best_q = float(np.max(qualities_cpu))
        best_idx = int(np.argmax(qualities_cpu))
        return avg_q, best_q, best_idx

    def _launch_pheromone_update(self) -> None:
        # First evaporate pheromones
        evap_block_size = 64
        evap_grid_size = (self.num_vars + evap_block_size - 1) // evap_block_size
        pheromone_evaporation_kernel(
            (evap_grid_size,), (evap_block_size,),
            (
                self.pheromones_gpu,
                np.float32(self.rho),
                np.int32(self.num_vars)
            )
        )
        cp.cuda.Stream.null.synchronize()
        
        # Then deposit pheromones atomically
        block_size = 64
        grid_size = (self.num_ants + block_size - 1) // block_size
        pheromone_update_kernel(
            (grid_size,), (block_size,),
            (
                self.pheromones_gpu,
                self.qualities_gpu,
                self.assignments_gpu,
                np.float32(self.alpha),
                np.float32(self.rho),
                np.int32(self.num_vars),
                np.int32(self.num_ants)
            )
        )
        cp.cuda.Stream.null.synchronize()

    # -------------------------------------------------------------------------
    # Local Search Kernel
    # -------------------------------------------------------------------------

    def _launch_local_search(self, top_fraction: float = 0.2, temperature: float = 0.01) -> None:
        """
        Applies a conflict-driven local search to the top fraction of solutions
        (based on the current GPU qualities). No examples provided.
        """
        # For simplicity, we run the local search across all ants. 
        # If you only want top fraction, gather their indices in CPU or do a more advanced 2D kernel.
        block_size = 128
        grid_size = (self.num_ants + block_size - 1) // block_size
        gpu_local_search_kernel(
            (grid_size,), (block_size,),
            (
                self.clause_array_gpu,
                self.clause_weights_gpu,
                np.int32(self.num_clauses),
                np.int32(self.max_clause_size),
                np.int32(self.num_vars),
                np.int32(self.num_ants),
                np.int32(self.local_search_flips),
                np.float32(temperature),
                self.assignments_gpu
            )
        )
        cp.cuda.Stream.null.synchronize()

    # -------------------------------------------------------------------------
    # Clause Weighting Updates (Fully on GPU)
    # -------------------------------------------------------------------------

    def _update_clause_weights_gpu(self) -> None:
        """
        Adapt coverage-based clause weights on the GPU, avoiding large CPU loops.
        """

        # 1) Zero coverage_count
        self.coverage_count_gpu.fill(0)

        # 2) coverage_count_kernel
        #    Each thread checks (ant, clause) and does an atomicAdd if satisfied
        total_pairs = self.num_ants * self.num_clauses
        block_size = 256
        grid_size = (total_pairs + block_size - 1) // block_size
        coverage_count_kernel(
            (grid_size,), (block_size,),
            (
                self.clause_array_gpu,
                np.int32(self.num_clauses),
                np.int32(self.max_clause_size),
                np.int32(self.num_vars),
                np.int32(self.num_ants),
                self.assignments_gpu,
                self.coverage_count_gpu
            )
        )
        cp.cuda.Stream.null.synchronize()

        # 3) stubb_update_kernel
        block_size2 = 128
        grid_size2 = (self.num_clauses + block_size2 - 1) // block_size2
        stubb_update_kernel(
            (grid_size2,), (block_size2,),
            (
                self.coverage_count_gpu,
                np.int32(self.num_clauses),
                np.int32(self.num_ants),
                self.clause_stubbornness_gpu,
                self.clause_weights_gpu,
                np.float32(self.clause_lr),           # conflict-driven LR
                np.float32(self.clause_weight_momentum)
            )
        )
        cp.cuda.Stream.null.synchronize()

        # 4) Keep average clause weight in a reasonable range
        avgw = cp.mean(self.clause_weights_gpu).item()
        if avgw > 10.0:
            scale_factor = 10.0 / avgw
            self.clause_weights_gpu *= scale_factor

# =============================================================================
# Main Function
# =============================================================================

def main() -> None:
    """
    Entry point for MACO Ultimate Edition (ZVSS Self-Optimization integrated),
    extended to offload more computations onto the GPU.
    NO EXAMPLES PROVIDED. USE CODE WITH CAUTION.
    """
    parser = argparse.ArgumentParser(description="GPU-Accelerated MACO SAT Solver (ZVSS + Extended GPU Offload)")
    parser.add_argument("--cnf", type=str, required=False,
                        help="Path to a DIMACS CNF file. If omitted, a random instance is generated.")
    parser.add_argument("--num_ants", type=int, default=1024, help="Number of ants.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations.")
    parser.add_argument("--log_level", type=str, default="DEBUG",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID.")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    config = MACOConfig(
        n_ants=args.num_ants,
        max_iterations=args.max_iter,
        solver_path="C:/msys64/ucrt64/bin/minisat.exe",
        gpu_device_id=args.device
    )

    clauses: List[List[int]] = []
    variables: List[int] = []

    # If user did not provide a CNF, generate a random 3-SAT instance
    if args.cnf:
        num_vars, num_clauses, clauses = parse_dimacs_cnf(args.cnf)
        logging.info(f"Parsed CNF file: {args.cnf}, {num_vars} vars, {num_clauses} clauses.")
        if num_clauses == 0 or num_vars == 0:
            logging.error("CNF file appears empty or invalid.")
            return
        variables = list(range(1, num_vars + 1))
    else:
        # Legacy logging lines for compatibility:
        logging.info("Running MACO with generated instance")
        logging.info("Generated random 3-SAT instance:")
        num_vars = config.num_vars_gen
        num_clauses = config.num_clauses_gen
        clause_size = config.clause_size_gen
        formula, vars_ = generate_sat_formula(num_vars, num_clauses, clause_size)
        clauses = [list(c) for c in formula]
        variables = vars_
        logging.info(f"Generated random {clause_size}-SAT with {num_vars} vars, {num_clauses} clauses.")

    solver = MACOSystem(config=config)
    solver.initialize_search_space(formula=clauses, variables=variables)
    best_assignment, best_quality = solver.solve()

    logging.info("Best quality found (GPU-MACO): %.4f" % best_quality)

    if best_assignment is not None:
        assignment_str = " ".join(str(v) for v in best_assignment[:min(50, len(best_assignment))])
        logging.info(f"Sample of best assignment (first 50 vars): {assignment_str}")
        full_assignment_str = " ".join(str(v) for v in best_assignment)
        logging.info(f"[FULL SOLUTION] {full_assignment_str}")

        satisfied_clauses = sum(
            any(
                (lit > 0 and best_assignment[abs(lit) - 1] == 1) or
                (lit < 0 and best_assignment[abs(lit) - 1] == 0)
                for lit in clause
            ) for clause in clauses
        )
        total_clauses = len(clauses)
        satisfaction_rate = satisfied_clauses / total_clauses if total_clauses > 0 else 0
        logging.info(f"[SOLUTION ANALYSIS] {satisfied_clauses}/{total_clauses} clauses satisfied "
                     f"({100.0 * satisfaction_rate:.2f}%)")
        unsatisfied_clauses = [
            clause for clause in clauses
            if not any(
                (lit > 0 and best_assignment[abs(lit) - 1] == 1) or
                (lit < 0 and best_assignment[abs(lit) - 1] == 0)
                for lit in clause
            )
        ]
        logging.info(f"[HARD CONSTRAINTS] {len(unsatisfied_clauses)} clauses remained unsatisfied.")
    else:
        logging.warning("No valid assignment found (or no assignment tracked).")

    # Optionally run classical solver for cross-check
    classical_status, classical_assignment, classical_time = run_classical_solver(
        clauses, variables, solver_path=config.solver_path
    )
    if classical_status == "SAT":
        logging.info(f"Classical Solver (MiniSat): SAT in {classical_time:.2f} seconds.")
    elif classical_status == "UNSAT":
        logging.info(f"Classical Solver (MiniSat): UNSAT in {classical_time:.2f} seconds.")
    elif classical_status == "TIMEOUT":
        logging.warning(f"Classical Solver (MiniSat): TIMEOUT after {classical_time:.2f} seconds.")
    else:
        logging.info(f"Classical Solver (MiniSat): UNKNOWN status in {classical_time:.2f} seconds.")

    rc = 1
    if classical_status == "SAT":
        rc = 10
    elif classical_status == "UNSAT":
        rc = 20
    elif classical_status == "TIMEOUT":
        rc = 124
    elif classical_status == "ERROR":
        rc = 2

    logging.info(f"Return code: {rc}")
    # input("Press Enter to exit...")  # comment out to avoid pause on some shells


if __name__ == "__main__":
    main()
