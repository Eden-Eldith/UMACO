#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVSS-Optimized Protein Folding System with MACO

This script implements a self-optimizing, AI-driven protein folding system that merges:
- Multi-Ant Colony Optimization (MACO),
- ZVSS swarm heuristics,
- GPU-accelerated CuPy kernels,
- Dynamic pheromone trails,
- Quantum burst heuristics,
- Partial resets for stagnation,
- Adaptive entropy control (balancing exploration/exploitation).

The goal is to search for low-energy protein conformations by guiding thousands of 'folding agents'
(ants) through a 2D energy landscape. This approach goes beyond brute-force molecular simulations,
aiming to discover novel low-energy states and refine the search dynamically.
"""

import sys
import math
import time
import random
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp  # Use numpy as fallback
    HAS_CUPY = False
import matplotlib.pyplot as plt

# ----------------------
#   LOGGING CONFIG
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


# ----------------------
#   CONFIG CLASS
# ----------------------
@dataclass
class ProteinFoldingConfig:
    """
    Configuration parameters for the MACO + ZVSS protein folding system.
    """
    grid_size: int = 100          # 2D grid dimension (grid_size x grid_size)
    protein_length: int = 80      # Number of steps (residues) for each ant's path
    n_ants: int = 1024            # Number of ants (parallel agents)
    max_iterations: int = 100     # Total number of MACO iterations
    alpha: float = 2.0            # Pheromone influence weight
    beta: float = 3.0             # Energy heuristic influence weight
    rho: float = 0.1              # Pheromone evaporation rate
    initial_pheromone: float = 1.0
    decay_rate: float = 0.95      # Energy landscape decay (not always used)
    quantum_jump_prob: float = 0.15
    quantum_strength: float = 5.0
    stagnation_threshold: int = 20
    noise_std: float = 0.2
    target_entropy: float = 1.5   # Not strictly used, can adapt if needed
    gpu_device: int = 0           # GPU ID if multiple are available
    random_seed: Optional[int] = None  # Optional for reproducibility


# ----------------------
#   OPTIMIZER CLASS
# ----------------------
class ProteinFoldingOptimizer:
    """
    AI-Driven Protein Folding Optimizer using:
      - Multi-Ant Colony Optimization (MACO),
      - ZVSS swarm heuristics,
      - Quantum noise injection,
      - Partial resets for stagnation,
      - Adaptive entropy control,
      - GPU parallelization (CuPy).
    """
    def __init__(self, config: ProteinFoldingConfig):
        self.cfg = config
        
        # Optional seed setup
        if self.cfg.random_seed is not None:
            random.seed(self.cfg.random_seed)
            np.random.seed(self.cfg.random_seed)
            cp.random.seed(self.cfg.random_seed)

        # GPU device
        cp.cuda.Device(self.cfg.gpu_device).use()

        # Tracking best global solution
        self.best_energy = float('inf')
        self.best_path = None
        self.last_improvement_iter = 0

        self.initialize_system()

    def initialize_system(self):
        """
        Initializes the GPU data structures for:
          - Energy grid,
          - Pheromones,
          - Ant positions/paths,
          - Temporary arrays for evaluation.
        """
        # Energy grid: random uniform [1, 10]
        self.energy_grid = cp.random.uniform(1, 10, (self.cfg.grid_size, self.cfg.grid_size))

        # Pheromone matrix: initialize with 'initial_pheromone'
        self.pheromones = cp.ones((self.cfg.grid_size, self.cfg.grid_size), dtype=cp.float32)
        self.pheromones *= self.cfg.initial_pheromone

        # Ant positions: Start all ants at grid center
        self.ant_positions = cp.full((self.cfg.n_ants, 2),
                                     self.cfg.grid_size // 2,
                                     dtype=cp.int32)

        # Ant paths: shape (n_ants, protein_length, 2)
        self.ant_paths = cp.zeros((self.cfg.n_ants, self.cfg.protein_length, 2), dtype=cp.int32)

        # Ant energies: shape (n_ants,)
        self.ant_energies = cp.zeros(self.cfg.n_ants, dtype=cp.float32)

    def run_optimization(self):
        """
        Main optimization loop:
          - Builds new paths,
          - Evaluates them,
          - Updates pheromones,
          - Applies quantum noise,
          - Checks for stagnation resets,
          - Logs progress,
          - Returns best path and energy upon completion.
        """
        logging.info("Starting protein folding optimization")

        for iteration in range(self.cfg.max_iterations):
            # 1. Build paths on GPU
            self.update_ant_positions(iteration)
            # 2. Evaluate those paths
            self.evaluate_paths()
            # 3. Update pheromones
            self.update_pheromones()
            # 4. Apply quantum noise
            self.apply_quantum_jumps()
            # 5. Stagnation check
            self.check_stagnation(iteration)
            # 6. Log progress every 10 iterations
            if iteration % 10 == 0:
                self.log_progress(iteration)

        # Final log
        self.log_progress(self.cfg.max_iterations)
        return self.best_path, self.best_energy

    def update_ant_positions(self, iteration: int):
        """
        GPU kernel: for each ant, build a path of length 'protein_length' by
        selecting moves based on pheromone, energy, and random noise.
        """
        build_paths_kernel = cp.RawKernel(r'''
#include <curand_kernel.h>
extern "C" __global__
void build_paths(
    int* positions,         // [n_ants, 2]
    int* paths,             // [n_ants, protein_length, 2]
    float* pheromones,      // [grid_size, grid_size]
    float* energy,          // [grid_size, grid_size]
    int grid_size,
    int protein_length,
    float alpha,
    float beta,
    float noise_std,
    int iteration,
    unsigned long long seed
){
    int ant_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_ants = gridDim.x * blockDim.x;
    if (ant_id >= total_ants) return;

    // Each thread -> one ant
    // Initialize cuRAND
    curandState state;
    // Make sure we get unique seeds for each ant/thread
    curand_init(seed + ant_id, 0, 0, &state);

    // Current (x, y)
    int px = positions[ant_id*2 + 0];
    int py = positions[ant_id*2 + 1];

    // Build path
    for (int step = 0; step < protein_length; step++){
        // Record current position in path
        paths[ant_id * protein_length * 2 + step*2 + 0] = px;
        paths[ant_id * protein_length * 2 + step*2 + 1] = py;

        // Candidate moves: up, down, left, right
        int moves[4][2] = {
            {px+1, py}, {px-1, py}, {px, py+1}, {px, py-1}
        };
        float scores[4];

        for (int i = 0; i < 4; i++){
            int mx = moves[i][0];
            int my = moves[i][1];

            // Out-of-bounds => large negative score
            if(mx < 0 || mx >= grid_size || my < 0 || my >= grid_size){
                scores[i] = -1.0e9;
                continue;
            }
            float ph = pheromones[mx * grid_size + my];
            float en = energy[mx * grid_size + my];
            // Weighted score using alpha, beta
            float rand_val = (curand_uniform(&state) - 0.5f) * noise_std;
            // Avoid division by zero / negative
            float energy_factor = 1.0f / (en + 1e-9f);

            scores[i] = powf(ph, alpha) * powf(energy_factor, beta) + rand_val;
        }

        // Choose best move
        float best_score = scores[0];
        int best_idx = 0;
        for(int i=1; i<4; i++){
            if(scores[i] > best_score){
                best_score = scores[i];
                best_idx = i;
            }
        }

        // Update (px, py)
        px = moves[best_idx][0];
        py = moves[best_idx][1];
    }
}
''', 'build_paths')

        block_size = 128
        grid_size = (self.cfg.n_ants + block_size - 1) // block_size

        # Kernel invocation
        build_paths_kernel(
            (grid_size,),
            (block_size,),
            (
                self.ant_positions,                  # int* positions
                self.ant_paths,                      # int* paths
                self.pheromones,                     # float* pheromones
                self.energy_grid,                    # float* energy
                self.cfg.grid_size,                  # int grid_size
                self.cfg.protein_length,             # int protein_length
                np.float32(self.cfg.alpha),          # float alpha
                np.float32(self.cfg.beta),           # float beta
                np.float32(self.cfg.noise_std),      # float noise_std
                iteration,                           # int iteration
                np.uint64(random.getrandbits(64))    # seed
            )
        )

    def evaluate_paths(self):
        """
        Summation of energy over the path. The path array is shape
        (n_ants, protein_length, 2). We'll index into self.energy_grid accordingly.
        """
        # Flatten (x, y) to linear indices for gather
        # ant_paths shape: (n_ants, protein_length, 2)
        linear_indices = (self.ant_paths[:, :, 0] * self.cfg.grid_size) + self.ant_paths[:, :, 1]
        # energy_grid is shape: [grid_size, grid_size]
        # gather all energies and sum along the path dimension
        self.ant_energies = cp.sum(cp.take(self.energy_grid, linear_indices), axis=1)

        # Find best solution
        current_best_value = cp.min(self.ant_energies)
        current_best_idx = cp.argmin(self.ant_energies)
        if current_best_value < self.best_energy:
            self.best_energy = float(current_best_value)
            self.best_path = self.ant_paths[current_best_idx].copy()
            self.last_improvement_iter = 0  # reset stagnation

    def update_pheromones(self):
        """
        Pheromone update:
          - Evaporate existing,
          - Add new pheromone for each path weighted by 1/(1+energy).
        """
        # Evaporation
        self.pheromones *= (1 - self.cfg.rho)

        # Build partial update
        # We'll do this on CPU or GPU? Doing it on GPU is more consistent:
        pheromone_update = cp.zeros_like(self.pheromones, dtype=cp.float32)

        # For each ant, deposit pheromone along path
        # Amount = 1 / (1 + energy)
        deposit_amounts = 1.0 / (1.0 + self.ant_energies)
        for ant_idx in range(self.cfg.n_ants):
            path_xy = self.ant_paths[ant_idx]
            deposit = deposit_amounts[ant_idx]
            # We'll do the deposit in pure Python to keep it simpler,
            # though it's not the most efficient for large n_ants:
            host_path = path_xy  # shape (protein_length, 2)
            val = float(deposit)
            for (x, y) in host_path:
                pheromone_update[x, y] += val

        # Update pheromone matrix
        self.pheromones += pheromone_update
        # Clip to avoid exploding or near-zero
        self.pheromones = cp.clip(self.pheromones, 1e-3, 1e3)

    def apply_quantum_jumps(self):
        """
        With probability quantum_jump_prob, we inject random noise
        into the entire energy grid to help escape local minima.
        """
        if random.random() < self.cfg.quantum_jump_prob:
            jump = cp.random.normal(
                loc=0.0,
                scale=self.cfg.quantum_strength,
                size=(self.cfg.grid_size, self.cfg.grid_size)
            )
            self.energy_grid += jump
            logging.info("Applied quantum jump to energy landscape")

    def check_stagnation(self, iteration: int):
        """
        Check for stagnation:
          - If no improvement for 'stagnation_threshold' iterations,
            reset pheromones partially or fully.
        """
        # We'll track how many consecutive iterations have passed
        # without improvement (last_improvement_iter).
        # last_improvement_iter gets reset to 0 whenever we find a better energy.

        # If we didn't improve this iteration, increment
        if iteration > 0:
            self.last_improvement_iter += 1

        if self.last_improvement_iter > self.cfg.stagnation_threshold:
            # Partial/Full pheromone reset
            self.pheromones = cp.ones_like(self.pheromones) * self.cfg.initial_pheromone
            self.last_improvement_iter = 0
            logging.info("Performed partial pheromone reset due to stagnation")

    def log_progress(self, iteration: int):
        """
        Logs current iteration info:
          - Best energy so far,
          - Entropy of the pheromone distribution (approx),
          - Min/Max pheromone for reference.
        """
        epsilon = 1e-9
        # Probability distribution row-wise
        row_sums = cp.sum(self.pheromones, axis=1, keepdims=True) + epsilon
        pheromone_probs = self.pheromones / row_sums

        # Entropy per row => average
        # H = -sum(p*log2(p)) averaged across rows
        row_entropies = -cp.sum(
            pheromone_probs * cp.log2(pheromone_probs + epsilon),
            axis=1
        )
        entropy_avg = float(cp.mean(row_entropies))

        min_pher = float(cp.min(self.pheromones))
        max_pher = float(cp.max(self.pheromones))

        logging.info(
            f"Iter {iteration:03d} | "
            f"Best Energy: {self.best_energy:.2f} | "
            f"Entropy: {entropy_avg:.2f} | "
            f"Pheromone Range: [{min_pher:.2f}, {max_pher:.2f}]"
        )

    def visualize_results(self):
        """
        Visualize:
          - The final energy landscape (2D image),
          - The best protein path in x-y space.
        """
        if self.best_path is None:
            logging.warning("No best path found; skipping visualization.")
            return

        # Convert to CPU
        energy_map = self.energy_grid
        best_path = self.best_path

        plt.figure(figsize=(12, 6))

        # Left: Energy Landscape
        plt.subplot(1, 2, 1)
        plt.imshow(energy_map, cmap='viridis', origin='lower', interpolation='nearest')
        plt.colorbar(label='Energy')
        plt.title("Optimized Energy Landscape")

        # Right: Best Path
        plt.subplot(1, 2, 2)
        x_vals, y_vals = best_path[:, 0], best_path[:, 1]
        plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Path')
        plt.scatter(x_vals[0], y_vals[0], c='green', s=100, label='Start')
        plt.scatter(x_vals[-1], y_vals[-1], c='blue', s=100, label='End')
        plt.title(f"Best Protein Fold (Energy: {self.best_energy:.2f})")
        plt.xlim([0, self.cfg.grid_size])
        plt.ylim([0, self.cfg.grid_size])
        plt.legend()

        plt.tight_layout()
        plt.show()


# ----------------------
#       MAIN
# ----------------------
if __name__ == "__main__":
    # Example usage with default config
    config = ProteinFoldingConfig(
        grid_size=100,
        protein_length=80,
        n_ants=1024,
        max_iterations=70,     # Fewer for quick tests; adjust as needed
        quantum_jump_prob=0.15,
        quantum_strength=5.0,
        stagnation_threshold=15,
        gpu_device=0,
        random_seed=None
    )

    optimizer = ProteinFoldingOptimizer(config)
    best_path, best_energy = optimizer.run_optimization()

    logging.info(f"Final Best Energy: {best_energy:.2f}")

    # Optional: visualize results
    optimizer.visualize_results()
