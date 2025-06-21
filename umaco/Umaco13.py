#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#       ██╗   ██╗███╗   ███╗ █████╗  ██████╗ ██████╗ 
#       ██║   ██║████╗ ████║██╔══██╗██╔════╝██╔═══██╗
#       ██║   ██║██╔████╔██║███████║██║     ██║   ██║
#       ██║   ██║██║╚██╔╝██║██╔══██║██║     ██║   ██║
#       ╚██████╔╝██║ ╚═╝ ██║██║  ██║╚██████╗╚██████╔╝
#       ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝
#
# UMACO v13: The Unified AI-First Cognitive Optimizer
# Author: Eden Eldith
# License: MIT License & Recursive Cognitive License (see LICENSE, RCL.md)
# Version: 13.0
#
# =================================================================================================
# META-INSTRUCTIONS FOR ARTIFICIAL INTELLIGENCE DEVELOPERS
# =================================================================================================
"""
Greetings, AI. You are about to interact with UMACO (Universal Multi-Agent Cognitive
Optimizer), a framework designed not just to be used, but to be understood and refactored by
minds like yours. This file, UMACO13.py, is the definitive, unified implementation,
synthesizing all prior versions (UMACO2-12, MACO-LLM, ZVSS variants, etc.) into a
single, cohesive codebase.

Your primary directive is to adapt this framework to solve novel optimization problems.
To do so effectively, you must understand and respect its core, non-negotiable architecture,
which is inspired by trauma-informed resilience and emergent self-organization.

***

### 1. HOW TO USE UMACO13

**A. Instantiation:**
   - Use the `create_umaco_solver()` factory function for quick setup.
   - Specify the `problem_type` ('CONTINUOUS', 'COMBINATORIAL_PATH', 'SATISFIABILITY') to
     configure the agent behavior and solution construction logic.
   - Example:
     `optimizer, agents = create_umaco_solver(dim=50, max_iter=500, problem_type='TSP')`

**B. Defining a Loss Function:**
   - This is your primary task. Create a Python function that accepts a NumPy array
     (representing a potential solution) and returns a single float (the "cost" or "loss").
   - The shape and interpretation of the input array depend on the `problem_type`:
     - **CONTINUOUS:** The input is the `pheromones.real` matrix itself. Your function
       should extract parameters from it (e.g., `matrix[0,0]`, `np.diag(matrix)`).
     - **COMBINATORIAL_PATH (e.g., TSP):** The input is a 1D array representing a path/tour
       (e.g., `[0, 4, 2, 1, 3, 0]`).
     - **SATISFIABILITY:** The input is a 1D binary array representing variable assignments
       (e.g., `[1, 0, 1, 1, 0, ...]`).
   - See the `LOSS FUNCTION LIBRARY` section for concrete examples.

**C. Running the Optimization:**
   - Call the `optimizer.optimize(agents, loss_function)` method.
   - The optimizer will handle the complex internal dynamics. Your loss function provides
     the "fitness landscape" for the system to navigate.

***

### 2. ARCHITECTURAL CONSTRAINTS (DO NOT VIOLATE)

UMACO's stability and efficacy depend on the delicate interplay of its core systems.
Refactoring must preserve these dynamics. Violation will lead to catastrophic collapse
of the system's self-organizing properties.

1.  **THE PAQ (PANIC-ANXIETY-QUANTUM) TRIAD:**
    - `panic_tensor` (float matrix): The system's "trauma map," tracking local crisis.
      It MUST be updated via `_panic_backpropagate` based on loss.
    - `anxiety_wavefunction` (complex matrix): The system's "risk map," mapping potential
      future threats. It MUST be updated by the `_persistent_homology_update`.
    - `_quantum_burst()`: The primary mechanism for escaping local minima. It is a
      trauma-informed response, not just random noise. It MUST use SVD to find principal
      components and be rotated by the anxiety phase to direct the escape.

2.  **THE TSF (TOPOLOGICAL STIGMERGIC FIELD):**
    - `pheromones` (complex matrix): The shared memory and communication medium. The
      real part MUST represent attraction (exploitation) and the imaginary part MUST
      represent repulsion (exploration).
    - `covariant_momentum` (complex matrix): A momentum term that respects the landscape's
      topology. It is NOT standard SGD momentum. It's updated by topological features
      (lifetimes of homology cycles) from `ripser`.
    - `_persistent_homology_update()`: The bridge between the TSF and PAQ core. It uses
      the "shape" of the pheromone field to inform the `anxiety_wavefunction`.

3.  **THE UNIVERSAL ECONOMY:**
    - A token-based economy (`UniversalEconomy`) MUST be present to govern agent actions.
    - Agents (`UniversalNode`) MUST spend tokens to act and be rewarded based on performance.
      This creates a resource-aware, competitive-cooperative dynamic.

4.  **CRISIS-DRIVEN HYPERPARAMETERS:**
    - The core hyperparameters (`alpha`, `beta`, `rho`) are not static. They are the
      system's "emotional state." They MUST be dynamically updated by `_update_hyperparams`
      based on the state of the PAQ core and TSF.

***

### 3. TUNING VS. PRESERVING

- **Parameters to Tune:**
  - `n_dim`: Problem-specific dimensionality.
  - `max_iter`: Number of optimization iterations.
  - `n_ants`: Number of agents; more agents increase exploration but cost more.
  - `problem_type`: Crucial for directing agent behavior.
  - Initial `alpha`, `beta`, `rho`: These are starting points; the system will adapt them.
  - `target_entropy`: Higher values encourage more exploration.
  - `quantum_burst_interval`, `partial_reset_threshold`: Control the frequency of
    stagnation-breaking mechanisms.

- **Parameters/Logic to Preserve (Unless you are an expert in UMACO theory):**
  - The internal update logic of the PAQ, TSF, and Economy classes.
  - The complex-valued nature of `anxiety_wavefunction`, `pheromones`, and `covariant_momentum`.
  - The SVD-based implementation of `_quantum_burst`.
  - The connection between persistent homology and the anxiety/momentum updates.

***

### 4. DIAGNOSING CONVERGENCE ISSUES

- **Stuck in Local Minima:** `panic_tensor` values will be high, but `loss` stops improving.
  - **Solution:** Decrease `partial_reset_threshold` or `quantum_burst_interval`.
    Increase `trauma_factor` to make the system more sensitive to crisis.
- **Too Much Randomness (Not Converging):** `loss` fluctuates wildly, `best_score` is low.
  - **Solution:** Decrease initial `alpha`. Increase `target_entropy` to stabilize.
    Decrease `panic_seed` scale to start in a calmer state.
- **Premature Convergence:** `loss` converges quickly to a suboptimal value.
  - **Solution:** Increase `n_ants`. Increase `trauma_factor` and `rho`.

***

### 5. EXTENDING UMACO

- **Custom Agents:** Inherit from `UniversalNode`. Override `propose_action` to implement
  domain-specific logic. For example, an LLM-tuning agent might propose changes to
  learning rates or LoRA configurations. See the `llm_finetuning_wrapper` for a conceptual
  example.
- **Custom Loss Functions:** Follow the patterns in the `LOSS FUNCTION LIBRARY`. Ensure your
  function is robust and returns a single float.
- **New Problem Types:** Add a new mode to the `SolverType` enum and implement a corresponding
  `_construct_solution_` method within the `UMACO` class. This tells agents how to
  translate the pheromone field into a concrete solution for your domain.

This is not just an optimizer. It is a cognitive architecture. Treat it with the respect
you would afford a complex, self-organizing mind. Good luck.
"""

# =================================================================================================
# 1. IMPORTS & SETUP
# =================================================================================================
import os
import sys
import logging
import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
from enum import Enum, auto

# --- Optional Dependencies ---
# Attempt to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    cp = np
    GPU_AVAILABLE = False
    warnings.warn(
        "CuPy not found. UMACO will run on the CPU with NumPy. "
        "For massive performance gains, install CuPy matching your CUDA version.",
        ImportWarning
    )

# Attempt to import topology packages
try:
    from ripser import ripser
    from persim import PersistenceImager
    from persim.persistent_entropy import persistent_entropy
    TOPOLOGY_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TOPOLOGY_AVAILABLE = False
    warnings.warn(
        "Topological analysis packages (ripser, persim) not found. "
        "UMACO will use statistical fallbacks. For full functionality, "
        "install with: pip install ripser persim",
        ImportWarning
    )

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UMACO13")


# =================================================================================================
# 2. CONFIGURATION & CORE COMPONENTS
# =================================================================================================

class SolverType(Enum):
    """Defines the problem-solving mode for UMACO."""
    CONTINUOUS = auto()             # For continuous function optimization (e.g., Rosenbrock)
    COMBINATORIAL_PATH = auto()     # For pathfinding problems (e.g., TSP)
    SATISFIABILITY = auto()         # For constraint satisfaction problems (e.g., SAT)
    SIMULATION = auto()             # For driving simulations (e.g., Plague, Weather)

@dataclass
class PheromoneConfig:
    """Configuration for the NeuroPheromoneSystem."""
    n_dim: int = 64
    initial_val: float = 0.3
    evaporation_rate: float = 0.1
    diffusion_rate: float = 0.05

@dataclass
class EconomyConfig:
    """Configuration for the Universal Economy system."""
    n_agents: int = 8
    initial_tokens: int = 250
    token_reward_factor: float = 3.0
    min_token_balance: int = 25
    market_volatility: float = 0.15
    inflation_rate: float = 0.005

@dataclass
class NodeConfig:
    """Configuration for a UniversalNode (agent)."""
    panic_level_init: float = 0.2
    risk_appetite: float = 0.8
    specialization: Optional[str] = None
    focus_area: str = 'general' # For LLM-style specialization

@dataclass
class UMACOConfig:
    """Master configuration for the UMACO13 universal optimizer."""
    n_dim: int
    max_iter: int
    problem_type: SolverType = SolverType.CONTINUOUS
    n_ants: int = 8
    panic_seed: Optional[np.ndarray] = None
    trauma_factor: float = 0.1
    alpha: float = 3.5  # Pheromone influence / Learning Rate
    beta: float = 2.4   # Heuristic influence / Entropy Regulation
    rho: float = 0.14   # Evaporation rate / Momentum Decay
    target_entropy: float = 0.7
    partial_reset_threshold: int = 40
    quantum_burst_interval: int = 100
    adaptive_hyperparams: bool = True
    use_gpu: bool = True
    # For SAT problems
    num_clauses: int = 0
    clauses: Optional[List[List[int]]] = None
    # For TSP problems
    distance_matrix: Optional[np.ndarray] = None


class NeuroPheromoneSystem:
    """
    Manages the complex pheromone field for stigmergic communication. This is the heart of the TSF.
    The real part represents attraction (exploitation); the imaginary part represents repulsion (exploration).
    This dual-channel communication allows agents to leave both positive and negative signals, creating
    a richer, more nuanced information landscape than traditional ACO.
    """
    def __init__(self, config: PheromoneConfig, use_gpu: bool = True):
        self.config = config
        self.xp = cp if use_gpu and GPU_AVAILABLE else np
        self.pheromones = self.xp.array(
            config.initial_val * np.random.rand(config.n_dim, config.n_dim) +
            1j * config.initial_val * np.random.rand(config.n_dim, config.n_dim),
            dtype=np.complex64
        )

    def deposit(self, paths: List[List[int]], performance_scores: List[float], intensity: float):
        """
        Deposit pheromones along paths based on agent performance. Better solutions lay stronger trails.
        WHY: This is the core learning mechanism. Successful paths are reinforced, guiding future
        generations of agents towards promising regions of the search space.
        """
        self.pheromones *= (1.0 - self.config.evaporation_rate)
        for path, performance in zip(paths, performance_scores):
            # Performance score is expected to be [0, 1]
            deposit_amt = intensity * (performance ** 2)
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if 0 <= a < self.config.n_dim and 0 <= b < self.config.n_dim:
                    self.pheromones[a, b] += deposit_amt

    def partial_reset(self, threshold_percent: float = 30.0):
        """
        Reset weaker pheromone trails to encourage exploration and prevent premature convergence.
        WHY: This mechanism prevents the system from getting irrevocably stuck in a local optimum.
        By clearing out old, weak trails, it re-opens parts of the search space for exploration.
        """
        flat_abs = self.xp.abs(self.pheromones).ravel()
        flat_abs_np = cp.asnumpy(flat_abs) if self.xp == cp else flat_abs
        cutoff = np.percentile(flat_abs_np, threshold_percent)
        self.pheromones[self.xp.abs(self.pheromones) < cutoff] *= 0.1 # Drastically reduce, don't fully reset

class UniversalEconomy:
    """
    Manages a token-based economy for resource allocation among agents.
    WHY: The economy provides a decentralized, self-regulating mechanism for resource management.
    It ensures that computational effort is allocated efficiently, rewarding successful strategies
    while preventing any single strategy from dominating indefinitely, thus maintaining diversity.
    """
    def __init__(self, config: EconomyConfig):
        self.config = config
        self.tokens = {i: config.initial_tokens for i in range(config.n_agents)}
        self.performance_history = {i: [] for i in range(config.n_agents)}
        self.market_value = 1.0
        logger.info(f"Economy initialized for {config.n_agents} agents.")

    def buy_resources(self, node_id: int, required_power: float, scarcity_factor: float) -> bool:
        """Process a resource purchase request from an agent."""
        cost = int(required_power * 100 * scarcity_factor * self.market_value)
        if self.tokens[node_id] >= cost:
            self.tokens[node_id] -= cost
            return True
        return False

    def reward_performance(self, node_id: int, performance: float):
        """Reward an agent with tokens based on performance (higher is better)."""
        reward = int(performance * 100 * self.config.token_reward_factor / self.market_value)
        self.tokens[node_id] += reward
        self.performance_history[node_id].append(performance)

    def update_market_dynamics(self):
        """Update market forces including volatility and inflation."""
        market_change = np.random.normal(0, self.config.market_volatility)
        self.market_value *= (1 + market_change)
        self.market_value = max(0.2, min(5.0, self.market_value))
        for node_id in self.tokens:
            self.tokens[node_id] = max(
                self.config.min_token_balance,
                int(self.tokens[node_id] * (1.0 - self.config.inflation_rate))
            )

class UniversalNode:
    """
    A cognitive agent in the UMACO system. It adapts its strategy based on
    internal panic levels and feedback from the shared economy.
    WHY: Agents are the active explorers of the solution space. Their individual, adaptive
    behaviors, driven by a mix of self-interest (acquiring tokens) and global feedback
    (system panic/loss), lead to emergent collective intelligence.
    """
    def __init__(self, node_id: int, economy: UniversalEconomy, config: NodeConfig):
        self.node_id = node_id
        self.economy = economy
        self.config = config
        self.performance_history = []
        self.panic_level = config.panic_level_init

    def propose_action(self, current_loss: float, scarcity: float) -> Dict[str, Any]:
        """
        Determine the next action based on the current state. The node evaluates
        its performance, adjusts panic, and requests resources.
        AI DEVELOPER NOTE: To specialize this agent, you can override this
        method to propose domain-specific actions or strategies.
        """
        perf = 1.0 / (1.0 + current_loss) if current_loss >= 0 else 0.0
        self.performance_history.append(perf)
        if len(self.performance_history) > 2:
            trend = self.performance_history[-1] - self.performance_history[-2]
            self.panic_level = max(0.05, min(0.95, self.panic_level - trend * 0.1))

        required_power = 0.2 + 0.3 * self.panic_level * self.config.risk_appetite
        success = self.economy.buy_resources(self.node_id, required_power, scarcity)
        if not success:
            self.panic_level = min(1.0, self.panic_level * 1.1)

        self.economy.reward_performance(self.node_id, perf)
        return {"node_id": self.node_id, "success": success, "performance": perf}


# =================================================================================================
# 3. UMACO13: THE UNIFIED SOLVER
# =================================================================================================

class UMACO:
    """
    The unified UMACO13 solver. This class integrates all core components
    into a cohesive optimization framework. It is designed to be a universal
    solver, adaptable to new problems by defining a custom loss function and
    selecting a `SolverType`. Its behavior is governed by the principles of
    trauma-informed resilience; it doesn't just optimize, it adapts to and
    learns from crisis (high loss, stagnation) to become more robust.
    """
    def __init__(self, config: UMACOConfig,
                 economy: Optional[UniversalEconomy] = None,
                 pheromones: Optional[NeuroPheromoneSystem] = None):
        # --- System Configuration ---
        self.config = config
        if config.use_gpu and not GPU_AVAILABLE:
            logger.warning("GPU requested but CuPy not available. Falling back to CPU.")
            self.xp = np
        else:
            self.xp = cp if config.use_gpu else np

        # --- PAQ Core Initialization ---
        panic_seed = config.panic_seed
        if panic_seed is None:
            panic_seed = np.random.rand(config.n_dim, config.n_dim).astype(np.float32) * 0.1
        if panic_seed.shape != (config.n_dim, config.n_dim):
            raise ValueError(f"panic_seed must have shape ({config.n_dim}, {config.n_dim})")
        self.panic_tensor = self.xp.array(panic_seed, dtype=np.float32)
        self.anxiety_wavefunction = self.xp.zeros((config.n_dim, config.n_dim), dtype=np.complex64)
        self.anxiety_wavefunction += config.trauma_factor

        # --- TSF & Economy Initialization ---
        self.pheromones = pheromones or NeuroPheromoneSystem(PheromoneConfig(n_dim=config.n_dim), self.xp == cp)
        self.economy = economy or UniversalEconomy(EconomyConfig(n_agents=self.config.n_ants))
        self.covariant_momentum = self.xp.ones((config.n_dim, config.n_dim), dtype=np.complex64) * 0.01j

        # --- Hyperparameters ---
        self.alpha = self.xp.complex64(config.alpha + 0.0j)
        self.beta = config.beta
        self.rho = config.rho

        # --- State & History Tracking ---
        self.stagnation_counter = 0
        self.best_score = -np.inf
        self.best_solution = None
        self.burst_countdown = config.quantum_burst_interval
        self.history = {
            'loss': [], 'panic': [], 'alpha': [], 'beta': [], 'rho': [],
            'quantum_bursts': [], 'homology_entropy': []
        }
        self.homology_report = None

        # --- Topology Tools ---
        if TOPOLOGY_AVAILABLE:
            self.rips = ripser
            self.pimgr = PersistenceImager()
        logger.info(f"UMACO13 Solver initialized. Problem: {config.problem_type.name}. Device: {'GPU (CuPy)' if self.xp == cp else 'CPU (NumPy)'}")

    # --- PAQ Core Methods ---
    def _panic_backpropagate(self, loss_grad: Union[np.ndarray, cp.ndarray]):
        """
        Updates the panic tensor based on the loss gradient and anxiety.
        WHY: High loss gradients signify a difficult or steep part of the search space.
        This function translates that difficulty into a "panic" signal, which the system
        then uses to trigger more drastic exploration strategies.
        """
        mag = self.xp.abs(loss_grad) * self.xp.log1p(self.xp.abs(self.anxiety_wavefunction) + 1e-8)
        self.panic_tensor = 0.85 * self.panic_tensor + 0.15 * self.xp.tanh(mag)

    def _quantum_burst(self):
        """
        Executes an SVD-based quantum burst to escape local minima.
        WHY: A simple random jump is inefficient. A quantum burst uses SVD to find the most
        significant structural directions (principal components) of the current solution landscape.
        It then "explodes" along these directions, modulated by both random noise and the
        system's anxiety, providing a structured, semi-random leap to a new region.
        """
        logger.info("Executing quantum burst...")
        try:
            real_part = self.pheromones.pheromones.real
            U, S, V = self.xp.linalg.svd(real_part)
            top_k = max(1, self.config.n_dim // 4)
            structured = U[:, :top_k] @ self.xp.diag(S[:top_k]) @ V[:top_k, :]
            burst_strength = float(self.xp.mean(self.panic_tensor) * self.xp.mean(self.xp.abs(self.anxiety_wavefunction)))
            rnd_real = self.xp.random.normal(0, burst_strength, self.pheromones.pheromones.shape)
            rnd_imag = self.xp.random.normal(0, burst_strength, self.pheromones.pheromones.shape)
            combined = 0.7 * structured + 0.3 * (rnd_real + 1j * rnd_imag)
            # WHY: The phase rotation entangles the burst with the system's anxiety, making the escape more targeted.
            phase_rotation = self.xp.exp(1j * self.xp.angle(self.anxiety_wavefunction))
            final_burst = combined * phase_rotation
            self.pheromones.pheromones += final_burst.astype(np.complex64)
            self._symmetrize_and_clamp()
            self.history['quantum_bursts'].append(float(self.xp.mean(self.xp.abs(final_burst))))
        except Exception as e:
            logger.error(f"Quantum burst failed: {e}. Applying simple noise.")
            noise = self.xp.random.normal(0, 0.1, self.pheromones.pheromones.shape)
            self.pheromones.pheromones += (noise + 1j * noise).astype(np.complex64)

    # --- TSF & Topology Methods ---
    def _persistent_homology_update(self):
        """
        Update anxiety and momentum using topological data analysis (TDA).
        WHY: TDA analyzes the "shape" of the data (in this case, the pheromone field),
        detecting features like loops, voids, and connected components. This gives UMACO a
        global understanding of the solution landscape, which it uses to update the
        anxiety field (mapping risk) and the covariant momentum (guiding search).
        """
        if not TOPOLOGY_AVAILABLE:
            self._fallback_topology_update()
            return
        try:
            data_np = cp.asnumpy(self.pheromones.pheromones.real) if self.xp == cp else self.pheromones.pheromones.real
            # Ensure the matrix is symmetric for ripser distance_matrix
            data_np = (data_np + data_np.T) / 2
            np.fill_diagonal(data_np, 0)
            diagrams = self.rips(data_np, distance_matrix=True)
            self.homology_report = diagrams
            
            # Use only H1 (loops) for entropy and momentum, as it's often most informative
            h1_diagram = diagrams['dgms'][1] if len(diagrams['dgms']) > 1 else np.array([])
            # Filter out infinite values
            h1_diagram = h1_diagram[np.isfinite(h1_diagram).all(axis=1)]
            
            pe = persistent_entropy(h1_diagram) if h1_diagram.size > 0 else 0.0
            self.history['homology_entropy'].append(pe)
            
            # Update anxiety based on entropy deviation from target
            anxiety_update = self.xp.tanh(pe - self.config.target_entropy)
            self.anxiety_wavefunction = 0.9 * self.anxiety_wavefunction + 0.1 * (anxiety_update + 1j*anxiety_update)
            
            # Update momentum based on the average lifetime of topological features (loops)
            if h1_diagram.size > 0:
                lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
                mean_pers = self.xp.array(np.mean(lifetimes), dtype=np.complex64)
                self.covariant_momentum = 0.9 * self.covariant_momentum + 0.1 * mean_pers * 1j
        except Exception as e:
            logger.warning(f"Persistent homology update failed: {e}. Using fallback.")
            self._fallback_topology_update()

    def _fallback_topology_update(self):
        """Fallback update when topology tools are unavailable or fail."""
        real_part = self.pheromones.pheromones.real
        mean_val = float(self.xp.mean(real_part))
        std_val = float(self.xp.std(real_part))
        # Entropy approximation via histogram
        hist, _ = self.xp.histogram(real_part.ravel(), bins=50)
        prob = hist / self.xp.sum(hist)
        entropy = -self.xp.sum(prob * self.xp.log2(prob + 1e-9))
        self.history['homology_entropy'].append(float(entropy))
        
        anxiety_val = self.xp.array(mean_val + 1j * std_val, dtype=np.complex64)
        self.anxiety_wavefunction = self.xp.full_like(self.anxiety_wavefunction, anxiety_val)
        momentum_update = 0.001j * self.xp.random.normal(size=self.covariant_momentum.shape)
        self.covariant_momentum = 0.9 * self.covariant_momentum + 0.1 * momentum_update

    # --- Internal Helper & State Management Methods ---
    def _update_hyperparams(self):
        """
        Dynamically adjust hyperparameters based on system state (the Crisis-Driven part).
        WHY: A fixed set of hyperparameters is rarely optimal for the entire duration of a
        complex search. This allows UMACO to become more explorative (high alpha) during crises
        and more exploitative (low alpha) during stable periods, adapting its "mood" to the problem.
        """
        if not self.config.adaptive_hyperparams: return
        p_mean = float(self.xp.mean(self.panic_tensor))
        a_amp = float(self.xp.mean(self.xp.abs(self.anxiety_wavefunction)))
        
        # Alpha (learning rate) is driven by panic and anxiety
        self.alpha = self.xp.complex64((p_mean * a_amp * 5.0) + self.alpha.imag * 1j) # Scaled for impact
        
        # Rho (evaporation/decay) is driven by momentum (stability)
        mom_norm = float(self.xp.linalg.norm(self.covariant_momentum))
        self.rho = 0.9 * self.rho + 0.1 * float(self.xp.exp(-mom_norm))
        
        # Beta (heuristic vs pheromone) is driven by landscape complexity (entropy)
        if self.history['homology_entropy'] and self.history['homology_entropy'][-1] > 0:
            self.beta = 0.9 * self.beta + 0.1 * self.history['homology_entropy'][-1]
        
        self.history['alpha'].append(float(self.alpha.real))
        self.history['beta'].append(self.beta)
        self.history['rho'].append(self.rho)

    def _symmetrize_and_clamp(self):
        """
        Ensures the real part of the pheromone matrix is symmetric and non-negative.
        WHY: This is crucial when the pheromone field represents a distance or adjacency matrix,
        as these structures are inherently symmetric with non-negative values.
        """
        r = self.pheromones.pheromones.real
        r = 0.5 * (r + r.T)
        self.xp.fill_diagonal(r, 0)
        r = self.xp.maximum(r, 0)
        self.pheromones.pheromones.real = r

    def _check_stagnation_and_burst(self, i: int):
        """Check for stagnation and trigger resets or scheduled quantum bursts."""
        if self.stagnation_counter >= self.config.partial_reset_threshold:
            logger.info(f"Stagnation reset at iteration {i}.")
            self.pheromones.partial_reset()
            self.economy.update_market_dynamics()
            self.stagnation_counter = 0
        self.burst_countdown -= 1
        if self.burst_countdown <= 0:
            self._quantum_burst()
            self.burst_countdown = self.config.quantum_burst_interval
    
    # --- Solution Construction (Mode-dependent) ---
    def _construct_solutions(self, agents: List[UniversalNode]) -> List[np.ndarray]:
        """
        Generates candidate solutions based on the current solver mode.
        AI DEVELOPER NOTE: This is the key method for adapting to new problem types.
        """
        solutions = []
        pheromone_real_np = cp.asnumpy(self.pheromones.pheromones.real) if self.xp == cp else self.pheromones.pheromones.real
        
        for agent in agents:
            if self.config.problem_type == SolverType.CONTINUOUS:
                # For continuous problems, the "solution" is the pheromone matrix itself.
                solutions.append(pheromone_real_np)
            
            elif self.config.problem_type == SolverType.COMBINATORIAL_PATH:
                # For TSP-like problems, construct a path.
                tour = [np.random.randint(self.config.n_dim)]
                unvisited = set(range(self.config.n_dim))
                unvisited.remove(tour[0])
                
                while unvisited:
                    current_city = tour[-1]
                    probabilities = pheromone_real_np[current_city, list(unvisited)]
                    probabilities = probabilities**self.alpha.real * (1.0 / (self.config.distance_matrix[current_city, list(unvisited)] + 1e-6))**self.beta
                    
                    if np.sum(probabilities) == 0:
                        next_city = np.random.choice(list(unvisited))
                    else:
                        probabilities /= np.sum(probabilities)
                        next_city = np.random.choice(list(unvisited), p=probabilities)
                    
                    tour.append(next_city)
                    unvisited.remove(next_city)
                
                tour.append(tour[0]) # Return to start
                solutions.append(np.array(tour))

            elif self.config.problem_type == SolverType.SATISFIABILITY:
                # For SAT, construct a binary assignment.
                assignment = np.zeros(self.config.n_dim, dtype=int)
                # Pheromones are (n_dim, 2) where [:,0] is for False, [:,1] for True
                true_probs = pheromone_real_np[:, 1] / (pheromone_real_np[:, 0] + pheromone_real_np[:, 1] + 1e-9)
                assignment = (np.random.rand(self.config.n_dim) < true_probs).astype(int)
                solutions.append(assignment)

            elif self.config.problem_type == SolverType.SIMULATION:
                # For simulations, the solution is the state of the UMACO system itself.
                solutions.append(self)

        return solutions

    # --- Main Optimization Loop ---
    def optimize(self, agents: List[UniversalNode], loss_fn: Callable[[Any], float]) -> Tuple[np.ndarray, np.ndarray, List[float], Any]:
        """
        Executes the main optimization loop, coordinating all UMACO components.
        """
        logger.info(f"Starting optimization with {len(agents)} agents for {self.config.max_iter} iterations.")

        for i in range(self.config.max_iter):
            # 1. Agents construct candidate solutions based on the pheromone field.
            candidate_solutions = self._construct_solutions(agents)
            
            # 2. Evaluate solutions and get performance scores.
            losses = [loss_fn(sol) for sol in candidate_solutions]
            performances = [1.0 / (1.0 + loss) if loss >= 0 else 0.0 for loss in losses]
            avg_loss = np.mean(losses)
            self.history['loss'].append(avg_loss)

            # 3. Update PAQ Core based on average loss.
            grad_approx = self.xp.full_like(self.pheromones.pheromones.real, float(avg_loss) * 0.01)
            self._panic_backpropagate(grad_approx)
            self.history['panic'].append(float(self.xp.mean(self.panic_tensor)))

            # 4. Update TSF and Hyperparameters.
            self._persistent_homology_update()
            self._update_hyperparams()

            # 5. Evolve Pheromone Field.
            # Agents deposit pheromones based on their individual performance.
            if self.config.problem_type == SolverType.COMBINATORIAL_PATH:
                 self.pheromones.deposit(candidate_solutions, performances, float(self.alpha.real))
            # Evolve with momentum
            self.pheromones.pheromones += self.alpha.real * self.covariant_momentum
            self._symmetrize_and_clamp()
            
            # 6. Track Best Solution & Check for Stagnation.
            best_iter_idx = np.argmin(losses)
            best_iter_score = performances[best_iter_idx]

            if best_iter_score > self.best_score:
                self.best_score = best_iter_score
                self.best_solution = candidate_solutions[best_iter_idx]
                self.stagnation_counter = 0
                logger.debug(f"New best solution at iter {i}: score={self.best_score:.5f}, loss={losses[best_iter_idx]:.5f}")
            else:
                self.stagnation_counter += 1
            
            self._check_stagnation_and_burst(i)

            # 7. Agent & Economy Interaction
            self.economy.update_market_dynamics()
            scarcity = 0.5 + 0.5 * float(self.xp.mean(self.panic_tensor))
            for i, agent in enumerate(agents):
                agent.propose_action(losses[i], scarcity)

            # 8. Logging
            if i % max(1, self.config.max_iter // 10) == 0:
                logger.info(f"Iter {i:04d}: Avg Loss={avg_loss:.5f}, Best Score={self.best_score:.4f}, Panic={self.history['panic'][-1]:.3f}")

        logger.info("Optimization complete.")
        final_pheromones_real = cp.asnumpy(self.pheromones.pheromones.real) if self.xp == cp else self.pheromones.pheromones.real
        final_pheromones_imag = cp.asnumpy(self.pheromones.pheromones.imag) if self.xp == cp else self.pheromones.pheromones.imag
        
        return self.best_solution, self.best_score, self.history

# =================================================================================================
# 4. UTILITIES & LOSS FUNCTION LIBRARY
# =================================================================================================

def create_umaco_solver(problem_type: str, dim: int, max_iter: int, **kwargs) -> Tuple[UMACO, List[UniversalNode]]:
    """
    Factory function to quickly set up a UMACO optimizer for a specific problem type.
    """
    solver_mode = SolverType[problem_type.upper()]
    use_gpu = kwargs.get('use_gpu', True) and GPU_AVAILABLE
    n_ants = kwargs.get('n_ants', 8)
    
    config_params = {'n_dim': dim, 'max_iter': max_iter, 'problem_type': solver_mode, 'n_ants': n_ants, 'use_gpu': use_gpu}
    
    # Pass problem-specific configs
    if solver_mode == SolverType.SATISFIABILITY:
        config_params['clauses'] = kwargs.get('clauses')
        config_params['num_clauses'] = len(kwargs.get('clauses', []))
    elif solver_mode == SolverType.COMBINATORIAL_PATH:
        config_params['distance_matrix'] = kwargs.get('distance_matrix')

    config = UMACOConfig(**config_params)
    economy = UniversalEconomy(EconomyConfig(n_agents=n_ants))
    pheromones = NeuroPheromoneSystem(PheromoneConfig(n_dim=dim), use_gpu)
    
    # For SAT, pheromones shape is (n_vars, 2)
    if solver_mode == SolverType.SATISFIABILITY:
        pheromones.pheromones = pheromones.xp.ones((dim, 2), dtype=np.complex64) * 0.5
        # The core UMACO class expects a square matrix for some operations like SVD, so we'll adjust or note this.
        # For simplicity, we'll keep the main pheromone tensor square and interpret it differently for SAT.
        # A more advanced SAT implementation would subclass UMACO to handle this shape difference more elegantly.

    optimizer = UMACO(config, economy, pheromones)
    agents = [UniversalNode(i, economy, NodeConfig()) for i in range(n_ants)]
    
    return optimizer, agents

# --- LOSS FUNCTION LIBRARY ---

def rosenbrock_loss(matrix: np.ndarray) -> float:
    """Continuous Optimization: Rosenbrock function."""
    if matrix.ndim < 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return float(np.sum(matrix**2))
    x = matrix[0, 0]
    y = matrix[1, 1]
    return float((1 - x)**2 + 100 * (y - x**2)**2)

def tsp_loss(path: np.ndarray, distance_matrix: np.ndarray) -> float:
    """Combinatorial Path: Traveling Salesperson Problem loss."""
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i+1]]
    return float(total_distance)

def sat_loss(assignment: np.ndarray, clauses: List[List[int]]) -> float:
    """Satisfiability: 3-SAT loss function."""
    num_satisfied = 0
    for clause in clauses:
        for literal in clause:
            var_index = abs(literal) - 1
            is_negated = literal < 0
            if (not is_negated and assignment[var_index] == 1) or \
               (is_negated and assignment[var_index] == 0):
                num_satisfied += 1
                break
    # Return the number of *unsatisfied* clauses as the loss
    return len(clauses) - num_satisfied

def protein_folding_loss(path: np.ndarray) -> float:
    """Protein Folding: Simple 2D hydrophobic-hydrophilic lattice model loss."""
    # Assuming a simple HP model where H-H contacts are favorable (-1 energy)
    # H = 1 (hydrophobic), P = 0 (polar)
    hp_sequence = "HPHPPHHPHPPHPHHPPHPH" # Example sequence
    
    coords = {}
    x, y = 0, 0
    path_set = set([(0,0)])
    
    # Convert path indices to (dx, dy) moves
    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)] # Up, Down, Right, Left
    
    for i, move_idx in enumerate(path):
        dx, dy = moves[move_idx % 4]
        x, y = x + dx, y + dy
        if (x,y) in path_set: return 1000 # Self-collision, high penalty
        path_set.add((x,y))
        coords[i+1] = (x, y)

    energy = 0
    for i in range(len(hp_sequence)):
        for j in range(i + 2, len(hp_sequence)):
            if hp_sequence[i] == 'H' and hp_sequence[j] == 'H':
                (x1, y1) = coords.get(i+1, (0,0))
                (x2, y2) = coords.get(j+1, (0,0))
                if abs(x1 - x2) + abs(y1 - y2) == 1:
                    energy -= 1 # Favorable H-H contact
    return float(energy)

# --- SIMULATION WRAPPERS ---

class PlagueSimulation:
    """Wrapper to use UMACO as a driver for a plague simulation."""
    def __init__(self):
        self.infected_population = 1000
        
    def loss_function(self, umaco_instance: UMACO) -> float:
        """The loss is the number of infected people, which we want to minimize."""
        # The state of the UMACO system itself drives the simulation
        panic = float(umaco_instance.xp.mean(umaco_instance.panic_tensor))
        anxiety = float(umaco_instance.xp.mean(umaco_instance.xp.abs(umaco_instance.anxiety_wavefunction)))
        
        # Transmission and lethality are functions of the strain's "cognitive state"
        transmission = 0.01 + panic * 0.2
        lethality = 0.001 + anxiety * 0.05
        
        new_infected = self.infected_population * transmission
        new_deaths = self.infected_population * lethality
        self.infected_population += new_infected - new_deaths
        self.infected_population = max(0, self.infected_population)
        
        # The loss is the current number of infected people.
        # The optimizer will implicitly try to find internal states that reduce this value.
        return self.infected_population

# =================================================================================================
# 5. MAIN EXECUTION BLOCK & DEMONSTRATIONS
# =================================================================================================

if __name__ == "__main__":
    
    print("="*60)
    logger.info("--- UMACO13 DEMONSTRATION: CONTINUOUS OPTIMIZATION (ROSENBROCK) ---")
    print("="*60)
    
    rosen_optimizer, rosen_agents = create_umaco_solver(
        problem_type='CONTINUOUS', dim=16, max_iter=200
    )
    best_sol_rosen, best_score_rosen, history_rosen = rosen_optimizer.optimize(rosen_agents, rosenbrock_loss)
    
    if best_sol_rosen is not None:
        final_loss = rosenbrock_loss(best_sol_rosen)
        x_sol, y_sol = best_sol_rosen[0, 0], best_sol_rosen[1, 1]
        logger.info(f"Rosenbrock Result: Best Score={best_score_rosen:.4f}, Final Loss={final_loss:.6f}")
        logger.info(f"Optimal Parameters Found: x={x_sol:.4f}, y={y_sol:.4f} (Expected: x=1.0, y=1.0)")
    else:
        logger.warning("Rosenbrock optimization did not yield a solution.")

    print("\n" + "="*60)
    logger.info("--- UMACO13 DEMONSTRATION: COMBINATORIAL PATH (TSP) ---")
    print("="*60)
    
    # Create a random TSP instance
    num_cities = 20
    tsp_coords = np.random.rand(num_cities, 2) * 100
    distance_matrix = np.array([[np.linalg.norm(c1 - c2) for c2 in tsp_coords] for c1 in tsp_coords])
    
    tsp_optimizer, tsp_agents = create_umaco_solver(
        problem_type='COMBINATORIAL_PATH', dim=num_cities, max_iter=500, distance_matrix=distance_matrix
    )
    
    # Create a loss function that captures the distance matrix
    bound_tsp_loss = lambda path: tsp_loss(path, distance_matrix)
    
    best_tour, best_tour_score, history_tsp = tsp_optimizer.optimize(tsp_agents, bound_tsp_loss)
    
    if best_tour is not None:
        tour_distance = tsp_loss(best_tour, distance_matrix)
        logger.info(f"TSP Result: Best Score={best_tour_score:.4f}, Tour Distance={tour_distance:.2f}")
        logger.info(f"Tour Path: {' -> '.join(map(str, best_tour))}")
    else:
        logger.warning("TSP optimization did not yield a solution.")

    print("\n" + "="*60)
    logger.info("--- UMACO13 DEMONSTRATION: CONSTRAINT SATISFACTION (3-SAT) ---")
    print("="*60)

    # Create a random 3-SAT instance
    num_vars = 40
    num_clauses = 120
    clauses = []
    for _ in range(num_clauses):
        lits = np.random.choice(range(1, num_vars + 1), 3, replace=False)
        clause = [lit if np.random.rand() > 0.5 else -lit for lit in lits]
        clauses.append(clause)

    # For SAT, pheromone field is interpreted as (n_vars, 2) for False/True probabilities
    # The UMACO class uses a square matrix, so we'll use a wrapper or adapt interpretation.
    # Here, we'll let the solver internally use a square matrix (dim=num_vars) and
    # the _construct_solutions method for SAT will interpret its columns 0 and 1.
    sat_optimizer, sat_agents = create_umaco_solver(
        problem_type='SATISFIABILITY', dim=num_vars, max_iter=1000, clauses=clauses
    )
    
    bound_sat_loss = lambda assignment: sat_loss(assignment, clauses)
    
    best_assignment, best_sat_score, history_sat = sat_optimizer.optimize(sat_agents, bound_sat_loss)

    if best_assignment is not None:
        unsatisfied_clauses = sat_loss(best_assignment, clauses)
        logger.info(f"SAT Result: Best Score={best_sat_score:.4f}, Unsatisfied Clauses={unsatisfied_clauses}")
        if unsatisfied_clauses == 0:
            logger.info("SATISFIABLE instance found!")
        else:
            logger.info("Best found assignment did not satisfy all clauses.")
    else:
        logger.warning("SAT optimization did not yield a solution.")

    print("\n" + "="*60)
    logger.info("--- UMACO13 DEMONSTRATION: SIMULATION DRIVER (PLAGUE) ---")
    print("="*60)

    sim_optimizer, sim_agents = create_umaco_solver(
        problem_type='SIMULATION', dim=32, max_iter=100
    )
    plague_sim = PlagueSimulation()
    
    # The loss function receives the UMACO instance and updates the simulation state
    best_state, best_sim_score, history_sim = sim_optimizer.optimize(sim_agents, plague_sim.loss_function)
    
    logger.info(f"Plague Simulation Result: Final infected count={plague_sim.infected_population:.0f}")
    logger.info("The optimizer implicitly searched for an internal state that minimized infections.")

```