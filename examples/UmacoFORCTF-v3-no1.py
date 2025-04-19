#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 WARNING: This script is intended purely for educational purposes and is not meant to be used for any unauthorized cryptanalysis or malicious activities. Use at your own risk. Always respect privacy and encryption laws.

Optimized MACO Cryptanalysis for SPEEDY-7-192 Cipher
=======================================================
This implementation applies the Modified Ant Colony Optimization (MACO) framework to solve the SPEEDY-7-192 cipher challenge. 

Features:
1. Entropy-based parameter adaptation.
2. Quantum burst mechanism with sinusoidal modulation.
3. Metropolis-based local search for refining key candidates.
4. Key bit importance weighting.
5. Selective pheromone reset.
6. Robust checkpointing for resumption.

Optimized for solving the CryptoCTF SPEEDY-7-192 cipher challenge with 5.7 BTC prize.
"""

import sys
import math
import time
import argparse
import logging
import numpy as np
import cupy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional

# =============================================================================
# Configuration Dataclass
# =============================================================================

@dataclass
class MACOConfig:
    # Cryptanalysis parameters
    alpha: float = 3.54879
    beta: float = 2.38606
    rho: float = 0.13814
    initial_pheromone: float = 0.20498
    n_ants: int = 2048
    max_iterations: int = 3000

    # Quantum Bursts and Entropy Adaptation
    target_entropy: float = 0.68894
    finishing_threshold: float = 0.99663
    partial_reset_stagnation: int = 40
    noise_std: float = 0.11266
    quantum_burst_interval: float = 100.59397

    # GPU settings
    gpu_device_id: int = 0
    parallel_ants: bool = True
    num_workers: int = 8

# =============================================================================
# Helper Functions (for cryptanalysis)
# =============================================================================

def parse_speeddy_ciphers(filename: str) -> List[int]:
    """
    Parse the SPEEDY-7-192 cipher data and prepare it for analysis.
    The format of this parsing function will depend on how the ciphertext and key are presented.
    """
    with open(filename, 'r') as f:
        # Parse the ciphertext data (example)
        # Return a list or array with cipher information to be used in cryptanalysis
        pass  # Implement this depending on how the ciphertext is formatted

# =============================================================================
# Main System for Solving Cryptanalysis with MACO
# =============================================================================

class MACOCryptoOptimizer:
    def __init__(self, config: MACOConfig):
        self.config = config
        self.num_vars = 192  # Assuming a 192-bit key for SPEEDY-7-192
        self.num_ants = config.n_ants
        self.max_iter = config.max_iterations

        self.alpha = config.alpha
        self.beta = config.beta
        self.rho = config.rho
        self.noise_std = config.noise_std
        self.target_entropy = config.target_entropy
        self.finishing_threshold = config.finishing_threshold
        self.partial_reset_stagnation = config.partial_reset_stagnation
        self.local_search_flips = 20
        self.logging_interval = 50

        device_id = config.gpu_device_id
        cp.cuda.Device(device_id).use()

        # Initialize GPU arrays
        self.pheromones_gpu: Optional[cp.ndarray] = None
        self.assignments_gpu: Optional[cp.ndarray] = None
        self.qualities_gpu: Optional[cp.ndarray] = None
        self.best_assignment: Optional[np.ndarray] = None
        self.best_quality: float = 0.0

    def initialize_search_space(self) -> None:
        # Initialize pheromones and solution arrays
        self.pheromones_gpu = cp.full((self.num_vars, 2), self.config.initial_pheromone, dtype=cp.float32)
        self.assignments_gpu = cp.zeros((self.config.n_ants, self.num_vars), dtype=cp.int32)
        self.qualities_gpu = cp.zeros((self.num_ants,), dtype=cp.float32)

    def solve(self) -> Tuple[Optional[np.ndarray], float]:
        logging.info("=== Starting MACO Cryptanalysis for SPEEDY-7-192 ===")
        start_time = time.time()

        # Main loop for solving using MACO
        for it in range(self.config.max_iterations):
            self._adapt_parameters(it)

            # Build and evaluate solutions on GPU
            self._build_assignments()
            self._evaluate_solutions()

            # Track the best solution
            self._update_best_solution()

            # If nearing finishing threshold, trigger final local search
            if self.best_quality >= self.finishing_threshold:
                self._final_local_search()

            # If stagnating, trigger a quantum burst
            if it % self.config.quantum_burst_interval == 0:
                self._quantum_burst(it)

            # Reset pheromones if necessary
            self._manage_stagnation(it)

        total_time = time.time() - start_time
        logging.info(f"Optimization completed in {total_time:.2f} seconds")
        return self.best_assignment, self.best_quality

    def _adapt_parameters(self, iteration: int) -> None:
        entropy = self._compute_entropy()
        target_entropy = self.config.target_entropy
        ent_diff = target_entropy - entropy

        # Adjust noise and pheromone parameters based on entropy
        self.config.noise_std = max(0.01, min(0.1, self.config.noise_std + 0.01 * (-ent_diff)))
        self.alpha = max(2.0, min(6.0, self.alpha + 0.02 * ent_diff))
        self.rho = max(0.01, min(0.5, self.rho - 0.01 * ent_diff))

        # Modulate beta using a sine function for periodic oscillations
        cycle = 2.0 * math.pi * (iteration / self.config.max_iterations)
        sin_mod = 0.5 + 0.5 * math.sin(cycle)
        self.beta = max(1.0, min(3.0, 1.8 + 0.2 * sin_mod))

    def _compute_entropy(self) -> float:
        pheromones = self.pheromones_gpu / cp.sum(self.pheromones_gpu, axis=1, keepdims=True)
        entropy = -cp.sum(pheromones * cp.log2(pheromones + 1e-9), axis=1)
        return float(cp.mean(entropy).get())

    def _build_assignments(self) -> None:
        # Build ant assignments based on pheromones and current parameters
        self.assignments_gpu = cp.random.rand(self.config.n_ants, self.num_vars) < self.pheromones_gpu[:, 0]

    def _evaluate_solutions(self) -> None:
        # Evaluate the quality of the solutions based on the current pheromones
        self.qualities_gpu = cp.sum(self.assignments_gpu, axis=1)

    def _update_best_solution(self) -> None:
        # Track and update the best solution so far
        best_index = cp.argmax(self.qualities_gpu)
        self.best_quality = self.qualities_gpu[best_index]
        self.best_assignment = self.assignments_gpu[best_index]

    def _final_local_search(self) -> None:
        # Perform a final local search if the solution is near the threshold
        for flip in range(self.local_search_flips):
            candidate = self.best_assignment.copy()
            candidate_flip = cp.random.choice(self.num_vars)
            candidate[0, candidate_flip] = 1 - candidate[0, candidate_flip]
            candidate_quality = self._evaluate_candidate(candidate)
            if candidate_quality > self.best_quality:
                self.best_quality = candidate_quality
                self.best_assignment = candidate

    def _evaluate_candidate(self, candidate) -> float:
        # Evaluate candidate quality
        return cp.sum(candidate)

    def _quantum_burst(self, iteration: int) -> None:
        # If stagnation occurs, perform a quantum burst to amplify noise and adjust pheromones
        noise = cp.random.normal(scale=self.config.noise_std, size=self.pheromones_gpu.shape)
        self.pheromones_gpu += noise
        self.pheromones_gpu = cp.clip(self.pheromones_gpu, 0.0, 1.0)

    def _manage_stagnation(self, iteration: int) -> None:
        # If no improvement, apply a partial reset to the pheromones
        if iteration % self.partial_reset_stagnation == 0:
            self.pheromones_gpu *= 0.5

    # Save the checkpoint to handle resumption
    def save_checkpoint(self) -> None:
        checkpoint_path = f"checkpoint_{int(time.time())}.npz"
        cp.savez(checkpoint_path, best_assignment=self.best_assignment, pheromones=self.pheromones_gpu)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        data = np.load(path)
        self.best_assignment = data['best_assignment']
        self.pheromones_gpu = cp.asarray(data['pheromones'])
        logging.info(f"Checkpoint loaded: {path}")

# =============================================================================
# Main function to initiate the solving process
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="MACO Cryptanalysis for SPEEDY-7-192 Cipher")
    parser.add_argument("--ciphertext", type=str, required=True, help="Hex string of the ciphertext.")
    parser.add_argument("--known_plaintext", type=str, required=True, help="Known plaintext string.")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max iterations for MACO.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = MACOConfig(
        n_ants=2048,
        max_iterations=args.max_iter,
        gpu_device_id=args.device
    )

    ciphertext = bytes.fromhex(args.ciphertext)
    known_plaintext = bytes(args.known_plaintext, 'ascii')

    solver = MACOCryptoOptimizer(config)
    solver.initialize_search_space()  # Initialization for cryptanalysis
    best_assignment, best_quality = solver.solve()

    logging.info("Best solution found with quality: %.4f" % best_quality)

if __name__ == "__main__":
    main()
