#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACO-ZVSS Integrated: Ultimate Zombie Swarm/Virus Spread Simulator - NO EXAMPLES

This script implements a MACO (Multi-Ant Colony Optimization) + ZVSS (Zombie Virus Spread Simulator)
system. It uses GPU kernels via CuPy to speed up the combinatorial search for parameter optimization.
Then, if requested, runs a Pygame-based simulation demonstrating swarming zombie behavior with the
optimized parameters.

Instructions:
- No illustrative examples are provided.
- Use code with caution.
"""

import sys
import math
import time
import argparse
import random
import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

try:
    import numpy as np
    try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp  # Use numpy as fallback
    HAS_CUPY = False
    import networkx as nx
except ImportError as e:
    print("ERROR: NumPy, CuPy, and NetworkX required.\n", e)
    sys.exit(1)

try:
    import pygame
except ImportError as e:
    print("ERROR: Pygame is required.\n", e)
    sys.exit(1)


@dataclass
class MACOZVSSConfig:
    """
    Holds configuration parameters for the MACO-ZVSS system.
    """
    alpha: float = float(3.60506)
    beta: float = float(1.90000)
    rho: float = float(0.14000)
    initial_pheromone: float = float(0.17005)
    n_ants: int = 1024
    max_iterations: int = 2
    clause_lr: float = float(0.20000)
    conflict_driven_learning_rate: float = float(0.26926)
    clause_weight_momentum: float = float(0.89999)
    clause_weight_decay: float = float(0.0)
    target_entropy: float = float(0.69891)
    finishing_threshold: float = float(0.99900)
    partial_reset_stagnation: int = 20
    noise_std: float = float(0.10000)
    quantum_burst_interval: float = (0.5)
    parallel_ants: bool = True
    num_workers: int = 8
    gpu_device_id: int = 0

    pheromone_threshold_var_range: Tuple[int, int] = (300, 800)
    swarm_speed_multiplier_var_range: Tuple[float, float] = (1.5, 3.5)
    sense_range_var_range: Tuple[int, int] = (3, 10)
    swarm_duration_var_range: Tuple[int, int] = (10, 50)
    movement_prob_var_range: Tuple[float, float] = (0.3, 0.8)
    mutation_prob_var_range: Tuple[float, float] = (0.01, 0.2)

    num_zvss_vars: int = 6


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
    out_quality[ant_id] = 0.0f;
}
''', 'evaluate_assignments_kernel')


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
    for (int v = 0; v < num_vars; v++) {
        int chosen_val = assignments[idx * num_vars + v];
        float oldF = pheromones[v*2 + 0];
        float oldT = pheromones[v*2 + 1];
        oldF = (1.0f - rho) * oldF;
        oldT = (1.0f - rho) * oldT;
        if (chosen_val == 1) {
            oldT += (alpha * scaledQ);
        } else {
            oldF += (alpha * scaledQ);
        }
        pheromones[v*2 + 0] = fmaxf(fminf(oldF, 10.0f), 0.001f);
        pheromones[v*2 + 1] = fmaxf(fminf(oldT, 10.0f), 0.001f);
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
    unsigned long long rng = seed ^ ((unsigned long long)(ant_id+1) * 2685821657736338717ULL);
    for (int v = 0; v < num_vars; v++) {
        rng ^= (rng << 13); rng ^= (rng >> 7); rng ^= (rng << 17);
        float r01 = (float)(rng & 0xFFFFFFFF) / 4294967295.0f;
        float phF = pheromones[v*2+0];
        float phT = pheromones[v*2+1];
        float aF = powf(phF + 1e-9f, alpha);
        float aT = powf(phT + 1e-9f, alpha);
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
}
''', 'gpu_local_search_kernel')


class MACOZVSSSystem:
    """
    Implements the Multi-Ant Colony Optimization for finding optimal ZVSS parameters.
    Uses CuPy GPU kernels to build assignments, evaluate them, and update pheromones.
    """
    def __init__(self, config: MACOZVSSConfig):
        """
        Initializes the MACOZVSS system with the provided configuration.

        :param config: MACOZVSSConfig object containing all relevant hyperparameters.
        """
        self.config = config
        self.num_vars = config.num_zvss_vars
        self.num_ants = config.n_ants
        self.max_iter = config.max_iterations
        self.alpha = config.alpha
        self.beta = config.beta
        self.rho = config.rho
        self.clause_lr = config.clause_lr
        self.clause_weight_momentum = config.clause_weight_momentum
        self.target_entropy = config.target_entropy
        self.finishing_threshold = config.finishing_threshold
        self.local_search_flips = 10
        self.partial_reset_stagnation = config.partial_reset_stagnation
        self.logging_interval = 10
        self.noise_std = config.noise_std

        device_id = config.gpu_device_id
        cp.cuda.Device(device_id).use()

        self.pheromones_gpu: Optional[cp.ndarray] = None
        self.assignments_gpu: Optional[cp.ndarray] = None
        self.qualities_gpu: Optional[cp.ndarray] = None
        self.max_clause_size = 1
        self.last_improvement_iter = 0

    def initialize_search_space(self) -> None:
        """
        Allocates and initializes GPU arrays for pheromones, assignments, and qualities.
        """
        init_pheromone = self.config.initial_pheromone
        self.pheromones_gpu = cp.full((self.num_vars, 2), init_pheromone, dtype=cp.float32)
        self.assignments_gpu = cp.zeros((self.config.n_ants, self.num_vars), dtype=cp.int32)
        self.qualities_gpu = cp.zeros((self.num_ants,), dtype=cp.float32)

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Runs the MACO optimization loop. Builds assignments, evaluates them, updates pheromones,
        and possibly applies partial resets. Returns the best parameter assignment found and its quality.

        :return: (best_global_assignment, best_global_quality)
        """
        logging.info("=== Starting MACO-ZVSS Optimization ===")
        logging.info(f"Optimizing {self.num_vars} ZVSS parameters, ants: {self.num_ants}, iterations: {self.max_iter}")
        start_time = time.time()
        best_global_quality = 0.0
        best_global_assignment = None

        for it in range(self.config.max_iterations):
            self._adapt_parameters(it)
            seed = random.getrandbits(64)
            self._launch_build_assignments(seed)
            avg_q, best_q, best_idx = self._launch_evaluation()

            if best_q > best_global_quality:
                best_global_quality = best_q
                best_assignment = self.assignments_gpu[best_idx, :]
                best_global_assignment = best_assignment
                self.last_improvement_iter = it

            self._launch_pheromone_update()
            self._check_quantum_burst(it, best_global_quality)

            if (it - self.last_improvement_iter) >= self.partial_reset_stagnation:
                logging.info(f"Partial reset triggered at iteration {it}.")
                self._apply_partial_pheromone_reset()
                self.last_improvement_iter = it

            if (it % self.logging_interval) == 0:
                cur_entropy = self._compute_entropy()
                logging.info(
                    f"[Iter {it}] best_q={best_global_quality:.4f}, avg_q={avg_q:.4f}, "
                    f"entropy={cur_entropy:.3f}, alpha={self.alpha:.3f}, rho={self.rho:.3f}, "
                    f"noise_std={self.config.noise_std:.4f}"
                )

        total_time = time.time() - start_time
        logging.info(f"=== MACO-ZVSS Optimization Finished in {total_time:.2f} seconds ====")
        logging.info(f"Best Swarm Quality: {best_global_quality:.4f}")
        logging.info(f"Best ZVSS Parameter Set (indices): {best_global_assignment}")
        return best_global_assignment, best_global_quality

    def _compute_entropy(self) -> float:
        """
        Computes the average entropy of the pheromones across all variables.
        Entropy is a measure of how uniform the pheromone distribution is.
        """
        p = self.pheromones_gpu / cp.sum(self.pheromones_gpu, axis=1, keepdims=True)
        epsilon = 1e-9
        ent = -cp.sum(p * cp.log2(p + epsilon), axis=1)
        return float(cp.mean(ent))

    def _adapt_parameters(self, iteration: int) -> None:
        """
        Adapts alpha, rho, and noise_std based on the current entropy and iteration count.
        """
        current_entropy = self._compute_entropy()
        ent_diff = self.config.target_entropy - current_entropy
        self.config.noise_std = max(0.01, min(0.1, self.config.noise_std + 0.01 * (-ent_diff)))
        self.alpha = max(2.0, min(6.0, self.alpha + 0.02 * ent_diff))
        self.rho = max(0.01, min(0.5, self.rho - 0.01 * ent_diff))
        cycle = 2.0 * math.pi * (float(iteration) / self.config.max_iterations)
        sin_mod = 0.5 + 0.5 * math.sin(cycle)
        self.beta = max(1.0, min(3.0, 1.8 + 0.2 * sin_mod))

    def _apply_partial_pheromone_reset(self) -> None:
        """
        Partially resets low-value pheromones to a small random value. Useful to escape local minima.
        """
        ph = self.pheromones_gpu.ravel()
        ph_cpu = ph
        cutoff_idx = int(len(ph_cpu) * 0.05)
        if cutoff_idx < 1:
            cutoff_idx = 1
        sorted_ph = np.sort(ph_cpu)[::-1]
        cutoff_val = sorted_ph[cutoff_idx - 1] if cutoff_idx <= len(sorted_ph) else 0.0
        rng = np.random.default_rng()
        for i in range(len(ph_cpu)):
            if ph_cpu[i] < cutoff_val:
                ph_cpu[i] = 0.01 + 0.01 * rng.random()
        ph = cp.asarray(ph_cpu).reshape(self.pheromones_gpu.shape)
        self.pheromones_gpu = ph

    def _launch_build_assignments(self, seed: int) -> None:
        """
        Launches the GPU kernel that builds assignments for each ant using the current pheromone values.
        """
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
        """
        Evaluates each ant's assignment by calling the GPU kernel (dummy, for now),
        then computes the actual quality on CPU by running the ZVSS simulation with
        the assigned parameters.

        :return: (average_quality, best_quality, index_of_best_ant)
        """
        block_size = 128
        grid_size = (self.num_ants + block_size - 1) // block_size
        self.qualities_gpu.fill(0)
        dummy_clause_array_gpu = cp.zeros((1, 1), dtype=cp.int32)
        dummy_clause_weights_gpu = cp.ones((1,), dtype=cp.float32)

        # Dummy GPU kernel call
        evaluate_kernel(
            (grid_size,), (block_size,),
            (
                dummy_clause_array_gpu,
                np.int32(0),
                np.int32(1),
                np.int32(self.num_vars),
                np.int32(self.num_ants),
                self.assignments_gpu,
                dummy_clause_weights_gpu,
                self.qualities_gpu
            )
        )
        cp.cuda.Stream.null.synchronize()

        # Real evaluation on CPU for each ant
        qualities_cpu_list = []
        assignments_cpu = self.assignments_gpu
        for ant_idx in range(self.num_ants):
            params = assignments_cpu[ant_idx, :]
            quality_score = self._evaluate_zvss_swarm_quality(params)
            qualities_cpu_list.append(quality_score)

        qualities_cpu = np.array(qualities_cpu_list, dtype=np.float32)
        self.qualities_gpu = cp.asarray(qualities_cpu)

        avg_q = float(np.mean(qualities_cpu))
        best_q = float(np.max(qualities_cpu))
        best_idx = int(np.argmax(qualities_cpu))
        return avg_q, best_q, best_idx

    def _launch_pheromone_update(self) -> None:
        """
        Launches the GPU kernel to update pheromones based on the assignments and their qualities.
        """
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

    def _check_quantum_burst(self, iteration: int, best_q: float) -> None:
        """
        Periodically triggers a 'quantum burst' (increasing noise, reducing alpha)
        if the best quality is below a threshold at specific intervals.
        """
        if (iteration % self.config.quantum_burst_interval) == (self.config.quantum_burst_interval - 1):
            if best_q < 0.999:
                old_noise = self.noise_std
                self.noise_std = min(0.5, self.noise_std * 3.0)
                self.alpha = max(1.0, self.alpha * 0.7)
                logging.info(
                    f"Quantum burst triggered: noise_std from {old_noise:.3f} to {self.noise_std:.3f}, "
                    f"alpha now {self.alpha:.3f}"
                )

    def _evaluate_zvss_swarm_quality(self, params: np.ndarray) -> float:
        """
        Runs a basic ZVSS simulation on CPU using the given parameter assignment, returning the swarm quality.

        :param params: An array of length num_zvss_vars (6), each entry is either 0 or 1, indicating
                       which boundary of the parameter range is used.
        :return: A float representing the average fraction of zombies in swarming state.
        """
        pheromone_threshold_param_idx = 0
        swarm_speed_multiplier_param_idx = 1
        sense_range_param_idx = 2
        swarm_duration_param_idx = 3
        movement_prob_param_idx = 4
        mutation_prob_param_idx = 5

        pheromone_threshold = self._map_param_value(
            params[pheromone_threshold_param_idx], self.config.pheromone_threshold_var_range
        )
        swarm_speed_multiplier = self._map_param_value(
            params[swarm_speed_multiplier_param_idx], self.config.swarm_speed_multiplier_var_range
        )
        zvss_sense_range = self._map_param_value(
            params[sense_range_param_idx], self.config.sense_range_var_range
        )
        zvss_swarm_duration = self._map_param_value(
            params[swarm_duration_param_idx], self.config.swarm_duration_var_range
        )
        zvss_movement_prob = self._map_param_value(
            params[movement_prob_param_idx], self.config.movement_prob_var_range
        )
        zvss_mutation_prob = self._map_param_value(
            params[mutation_prob_param_idx], self.config.mutation_prob_var_range
        )

        grid_size = 50
        zombie_count = 50
        simulation_steps = 50

        pheromone_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        player_x, player_y = grid_size // 2, grid_size // 2
        zombies = [
            (random.randint(0, grid_size - 2), random.randint(0, grid_size - 2))
            for _ in range(zombie_count)
        ]
        zombie_data_zvss = {z: {"type": "normal", "is_swarming": False, "swarm_timer": 0} for z in zombies}

        ZVSS_SENSE_RANGE = int(zvss_sense_range)
        ZVSS_PHEROMONE_THRESHOLD = pheromone_threshold
        ZVSS_SWARM_SPEED_MULTIPLIER = swarm_speed_multiplier
        ZVSS_SWARM_DURATION = int(zvss_swarm_duration)
        ZVSS_SWARM_CASCADE_RANGE = 3
        ZVSS_DECAY_RATE = 0.95
        ZVSS_MUTATION_PROB = zvss_mutation_prob
        ZVSS_MOVEMENT_PROB = zvss_movement_prob

        avg_swarm_size = 0
        for step in range(simulation_steps):
            pheromone_grid *= ZVSS_DECAY_RATE
            add_noise_zvss(pheromone_grid, player_x, player_y, grid_size, intensity=50)
            move_zombies_zvss(
                zombies,
                zombie_data_zvss,
                pheromone_grid,
                grid_size,
                sense_range=ZVSS_SENSE_RANGE,
                pheromone_threshold=ZVSS_PHEROMONE_THRESHOLD,
                swarm_speed_multiplier=ZVSS_SWARM_SPEED_MULTIPLIER,
                swarm_duration=ZVSS_SWARM_DURATION,
                swarm_cascade_range=ZVSS_SWARM_CASCADE_RANGE,
                mutation_prob=ZVSS_MUTATION_PROB,
                movement_prob=ZVSS_MOVEMENT_PROB,
                player_x=player_x,
                player_y=player_y  # Pass the player's position here
            )
            current_swarm_size = sum(1 for z_data in zombie_data_zvss.values() if z_data["is_swarming"])
            avg_swarm_size += current_swarm_size

        avg_swarm_size /= simulation_steps
        swarm_quality = avg_swarm_size / zombie_count
        return swarm_quality

    def _map_param_value(self, ant_assignment_val: int, param_range: Tuple[float, float]) -> float:
        """
        Maps a binary assignment (0 or 1) to the min or max of a parameter range.

        :param ant_assignment_val: 0 or 1, indicating low or high boundary in the param range.
        :param param_range: (min_val, max_val)
        :return: A float within the provided range.
        """
        min_val, max_val = param_range
        if ant_assignment_val == 0:
            return min_val
        else:
            return max_val


def add_noise_zvss(pheromone_grid: np.ndarray, x: int, y: int, grid_size: int, intensity: int) -> None:
    """
    Adds pheromone 'noise' at the given location to stimulate zombie movement.

    :param pheromone_grid: 2D NumPy array holding pheromone values.
    :param x: X-coordinate of noise.
    :param y: Y-coordinate of noise.
    :param grid_size: Size of the grid (width == height).
    :param intensity: How much pheromone to add.
    """
    if 0 <= x < grid_size and 0 <= y < grid_size:
        pheromone_grid[x, y] = max(pheromone_grid[x, y] + intensity, 1000)


def move_zombies_zvss(
    zombies: List[Tuple[int, int]],
    zombie_data: Dict[Tuple[int, int], Dict[str, object]],
    pheromone_grid: np.ndarray,
    grid_size: int,
    sense_range: int,
    pheromone_threshold: float,
    swarm_speed_multiplier: float,
    swarm_duration: int,
    swarm_cascade_range: int,
    mutation_prob: float,
    movement_prob: float,
    player_x: int,
    player_y: int
) -> None:
    """
    Moves zombies on the grid according to swarm rules and pheromones.
    This function has been updated to consider the player's position for extended swarming.

    :param zombies: List of zombie (x, y) positions.
    :param zombie_data: Dictionary holding zombie details like "type", "is_swarming", and "swarm_timer".
    :param pheromone_grid: 2D NumPy array of pheromone values.
    :param grid_size: Dimension of the grid.
    :param sense_range: Radius to sense pheromones.
    :param pheromone_threshold: Threshold above which zombies become swarming.
    :param swarm_speed_multiplier: Multiplier for movement speed when swarming.
    :param swarm_duration: Base duration (in steps) for which zombies remain swarming.
    :param swarm_cascade_range: Distance within which swarming can spread to other zombies.
    :param mutation_prob: Probability of mutating into another zombie type.
    :param movement_prob: Probability of following a pheromone-based move choice vs. random move.
    :param player_x: Player's x-position on the grid.
    :param player_y: Player's y-position on the grid.
    """
    ZOMBIE_TYPES_ZVSS = {
        "normal": (255, 0, 0),
        "fast": (255, 100, 100),
        "tank": (150, 0, 0),
    }
    new_positions = []
    swarming_zombies = set()

    for i in range(len(zombies)):
        z = zombies[i]
        x, y = z
        current_zombie_data = zombie_data[z]
        is_swarming = current_zombie_data["is_swarming"]
        swarm_timer = current_zombie_data["swarm_timer"]

        if is_swarming:
            swarm_timer -= 1
            zombie_data[z]["swarm_timer"] = swarm_timer
            if swarm_timer <= 0:
                zombie_data[z]["is_swarming"] = False
                is_swarming = False
            else:
                # Extend swarm timer near player
                if abs(x - player_x) <= 2 and abs(y - player_y) <= 2:
                    zombie_data[z]["swarm_timer"] = max(swarm_timer, 5)

        neighbors_unshuffled = [
            (x + dx, y + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= x + dx < grid_size and 0 <= y + dy < grid_size
        ]
        neighbors = neighbors_unshuffled[:]
        random.shuffle(neighbors)

        if not neighbors:
            new_positions.append(z)
            continue

        if not is_swarming:
            sensed_pheromone = False
            for sx in range(max(0, x - sense_range), min(grid_size, x + sense_range + 1)):
                for sy in range(max(0, y - sense_range), min(grid_size, y + sense_range + 1)):
                    if pheromone_grid[sx, sy] > pheromone_threshold:
                        sensed_pheromone = True
                        break
                if sensed_pheromone:
                    break

            if sensed_pheromone:
                zombie_data[z]["is_swarming"] = True
                zombie_data[z]["swarm_timer"] = swarm_duration
                is_swarming = True
                swarming_zombies.add(z)

        if is_swarming:
            # Direct chase logic at very close range
            if abs(x - player_x) <= 1 and abs(y - player_y) <= 1:
                dx = 0
                if player_x > x:
                    dx = 1
                elif player_x < x:
                    dx = -1
                dy = 0
                if player_y > y:
                    dy = 1
                elif player_y < y:
                    dy = -1
                best_move = (x + dx, y + dy)
            else:
                # Reversed pheromone-based logic for swarming
                best_move = min(neighbors, key=lambda pos: pheromone_grid[pos])
            speed_multiplier = swarm_speed_multiplier
            move_prob = movement_prob
        else:
            # Normal movement
            best_move = max(neighbors, key=lambda pos: pheromone_grid[pos])
            speed_multiplier = 1.0
            move_prob = movement_prob

        if random.random() < move_prob:
            best_move = best_move
        else:
            best_move = random.choice(neighbors)

        final_move = best_move
        for _ in range(int(speed_multiplier) - 1):
            neighbors_boost = [
                (final_move[0] + dx, final_move[1] + dy)
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= final_move[0] + dx < grid_size and 0 <= final_move[1] + dy < grid_size
            ]
            if neighbors_boost:
                if random.random() < move_prob:
                    final_move = max(neighbors_boost, key=lambda pos: pheromone_grid[pos])
                else:
                    final_move = random.choice(neighbors_boost)

        current_type = zombie_data[z]["type"]
        if random.random() < mutation_prob:
            new_type = random.choice(list(ZOMBIE_TYPES_ZVSS.keys()))
            zombie_data[final_move] = {"type": new_type, "is_swarming": is_swarming, "swarm_timer": swarm_timer}
        else:
            zombie_data[final_move] = {"type": current_type, "is_swarming": is_swarming, "swarm_timer": swarm_timer}

        new_positions.append(final_move)

    zombies[:] = new_positions[:]

    if swarming_zombies:
        for swarming_zombie_pos in list(swarming_zombies):
            sx, sy = swarming_zombie_pos
            for z_pos in zombies:
                if z_pos not in swarming_zombies and not zombie_data[z_pos]["is_swarming"]:
                    zx, zy = z_pos
                    if (abs(zx - sx) <= swarm_cascade_range and abs(zy - sy) <= swarm_cascade_range
                            and (zx, zy) != (sx, sy)):
                        zombie_data[z_pos]["is_swarming"] = True
                        zombie_data[z_pos]["swarm_timer"] = swarm_duration
                        swarming_zombies.add(z_pos)


def run_zvss_pygame(best_params: Dict[str, float]) -> None:
    """
    Runs the ZVSS simulation in Pygame using the specified optimized parameters.

    :param best_params: A dictionary containing the best ZVSS parameters found by MACO.
    """
    pygame.init()

    DECAY_RATE = 0.97
    MUTATION_PROB = 0.1
    NOISE_INTENSITY = 100
    MOVEMENT_PROB = 0.75
    PLAYER_MOVE_DELAY = 1
    CELL_SIZE = 6
    GRID_SIZE = 170

    ZVSS_SENSE_RANGE = 5
    ZVSS_PHEROMONE_THRESHOLD = best_params["pheromone_threshold"]
    ZVSS_SWARM_SPEED_MULTIPLIER = best_params["swarm_speed_multiplier"]
    ZVSS_SWARM_DURATION = 30
    ZVSS_SWARM_CASCADE_RANGE = 3

    SCREEN_WIDTH = GRID_SIZE * CELL_SIZE
    SCREEN_HEIGHT = GRID_SIZE * CELL_SIZE
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    pheromone_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    player_x, player_y = GRID_SIZE // 2, GRID_SIZE // 2
    player_move_timer = 0

    zombie_count = 200
    zombies = [
        (random.randint(0, GRID_SIZE - 2), random.randint(0, GRID_SIZE - 2))
        for _ in range(zombie_count)
    ]
    ZOMBIE_TYPES_PYGAME = {
        "normal": (255, 0, 0),
        "fast": (255, 100, 100),
        "tank": (150, 0, 0),
    }
    zombie_data_pygame = {z: {"type": "normal", "is_swarming": False, "swarm_timer": 0} for z in zombies}

    def decay_pheromones_pygame() -> None:
        """
        Applies a uniform decay to the pheromone grid.
        """
        nonlocal pheromone_grid
        pheromone_grid *= DECAY_RATE

    def add_noise_pygame(x: int, y: int, intensity: int = NOISE_INTENSITY) -> None:
        """
        Adds 'noise' to the pheromone grid at the player's location.

        :param x: Player's x-coordinate.
        :param y: Player's y-coordinate.
        :param intensity: Amount of pheromone to add.
        """
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            pheromone_grid[x, y] = max(pheromone_grid[x, y] + intensity, 100)

    def move_zombies_pygame() -> None:
        """
        Moves zombies in the Pygame simulation, utilizing the updated move_zombies_zvss.
        """
        nonlocal zombies, zombie_data_pygame
        move_zombies_zvss(
            zombies,
            zombie_data_pygame,
            pheromone_grid,
            GRID_SIZE,
            sense_range=ZVSS_SENSE_RANGE,
            pheromone_threshold=ZVSS_PHEROMONE_THRESHOLD,
            swarm_speed_multiplier=ZVSS_SWARM_SPEED_MULTIPLIER,
            swarm_duration=ZVSS_SWARM_DURATION,
            swarm_cascade_range=ZVSS_SWARM_CASCADE_RANGE,
            mutation_prob=MUTATION_PROB,
            movement_prob=MOVEMENT_PROB,
            player_x=player_x,
            player_y=player_y
        )

    def draw_scene_pygame() -> None:
        """
        Renders the pheromone grid, zombies, and player on the Pygame window.
        """
        screen.fill((0, 0, 0))

        max_pheromone = np.max(pheromone_grid) if pheromone_grid.size > 0 else 1.0
        if max_pheromone > 0:
            for y in range(GRID_SIZE):
                for x in range(GRID_SIZE):
                    pheromone_val = pheromone_grid[x, y]
                    intensity = int((pheromone_val / max_pheromone) * 255) if max_pheromone > 0 else 0
                    pheromone_color = (intensity, intensity, intensity)
                    if intensity > 0:
                        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                        pygame.draw.rect(screen, pheromone_color, rect)

        for z in zombies:
            wx, wy = z
            sx = wx * CELL_SIZE
            sy = wy * CELL_SIZE
            color = ZOMBIE_TYPES_PYGAME[zombie_data_pygame[z]["type"]]
            if zombie_data_pygame[z]["is_swarming"]:
                color = (255, 255, 0)
            pygame.draw.rect(screen, color, (sx, sy, CELL_SIZE, CELL_SIZE))

        pygame.draw.rect(
            screen,
            (0, 255, 0),
            (
                (GRID_SIZE // 2) * CELL_SIZE - CELL_SIZE // 2,
                (GRID_SIZE // 2) * CELL_SIZE - CELL_SIZE // 2,
                CELL_SIZE,
                CELL_SIZE
            )
        )
        pygame.display.flip()

    def handle_player_input_pygame() -> None:
        """
        Handles player input (arrow keys) to move around and add pheromone noise.
        """
        nonlocal player_x, player_y, player_move_timer
        keys = pygame.key.get_pressed()
        if player_move_timer <= 0:
            dx, dy = 0, 0
            if keys[pygame.K_LEFT]:
                dx -= 1
            if keys[pygame.K_RIGHT]:
                dx += 1
            if keys[pygame.K_UP]:
                dy -= 1
            if keys[pygame.K_DOWN]:
                dy += 1
            if dx != 0 or dy != 0:
                new_x = max(0, min(GRID_SIZE - 1, player_x + dx))
                new_y = max(0, min(GRID_SIZE - 1, player_y + dy))
                if new_x != player_x or new_y != player_y:
                    player_x, player_y = new_x, new_y
                    add_noise_pygame(player_x, player_y)
                    player_move_timer = PLAYER_MOVE_DELAY
        else:
            player_move_timer -= 0.5

    running = True
    step = 0
    while running:
        for event in pygame.to_numpy_scalar(event):
            if event.type == pygame.QUIT:
                running = False

        handle_player_input_pygame()
        decay_pheromones_pygame()
        if step % 10 == 0:
            add_noise_pygame(player_x, player_y, intensity=500)
        move_zombies_pygame()
        draw_scene_pygame()
        step += 0.5
        clock.tick(10)

    pygame.quit()


def main() -> None:
    """
    Main entry point for the MACO-ZVSS script. Parses command line arguments,
    optionally runs MACO for optimization, and optionally runs the Pygame simulation.
    """
    parser = argparse.ArgumentParser(description="MACO-ZVSS Integrated Simulator.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level.")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID for MACO.")
    parser.add_argument("--run_maco", action='store_true', help="Run MACO parameter optimization.")
    parser.add_argument("--run_zvss", action='store_true', help="Run ZVSS Pygame simulation.")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    config = MACOZVSSConfig(gpu_device_id=args.device)

    if args.run_maco:
        logging.info("=== Running MACO Parameter Optimization ===")
        maco_zvss_solver = MACOZVSSSystem(config=config)
        maco_zvss_solver.initialize_search_space()
        best_assignment, best_quality = maco_zvss_solver.solve()

        logging.info("=== MACO Optimization Complete ===")
        logging.info(f"Best Swarm Quality: {best_quality:.4f}")
        logging.info(f"Best Parameter Assignment (indices): {best_assignment}")

        best_pheromone_threshold = maco_zvss_solver._map_param_value(
            best_assignment[0], config.pheromone_threshold_var_range
        )
        best_swarm_speed_multiplier = maco_zvss_solver._map_param_value(
            best_assignment[1], config.swarm_speed_multiplier_var_range
        )
        best_sense_range = maco_zvss_solver._map_param_value(
            best_assignment[2], config.sense_range_var_range
        )
        best_swarm_duration = maco_zvss_solver._map_param_value(
            best_assignment[3], config.swarm_duration_var_range
        )
        best_movement_prob = maco_zvss_solver._map_param_value(
            best_assignment[4], config.movement_prob_var_range
        )
        best_mutation_prob = maco_zvss_solver._map_param_value(
            best_assignment[5], config.mutation_prob_var_range
        )

        best_zvss_params = {
            "pheromone_threshold": best_pheromone_threshold,
            "swarm_speed_multiplier": best_swarm_speed_multiplier,
            "sense_range": best_sense_range,
            "swarm_duration": best_swarm_duration,
            "movement_prob": best_movement_prob,
            "mutation_prob": best_mutation_prob
        }
        logging.info(f"Best ZVSS Parameters: {best_zvss_params}")

        if args.run_zvss:
            logging.info("=== Running ZVSS Pygame Simulation with Optimized Parameters ===")
            run_zvss_pygame(best_zvss_params)
        else:
            logging.info("=== Skipping ZVSS Pygame Simulation ===")

    elif args.run_zvss:
        logging.info("=== Running ZVSS Pygame Simulation with Default Parameters ===")
        default_zvss_params = {
            "pheromone_threshold": 600,
            "swarm_speed_multiplier": 2.0,
            "sense_range": 5,
            "swarm_duration": 30,
            "movement_prob": 0.55,
            "mutation_prob": 0.1
        }
        run_zvss_pygame(default_zvss_params)

    else:
        logging.info("=== No action specified. Use --run_maco or --run_zvss ===")


if __name__ == "__main__":
    main()

input("Press Enter to exit...")
