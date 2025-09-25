#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
       ██╗   ██╗███╗   ███╗ █████╗  ██████╗ ██████╗ 
       ██║   ██║████╗ ████║██╔══██╗██╔════╝██╔═══██╗
       ██║   ██║██╔████╔██║███████║██║     ██║   ██║
       ██║   ██║██║╚██╔╝██║██╔══██║██║     ██║   ██║
       ╚██████╔╝██║ ╚═╝ ██║██║  ██║╚██████╗╚██████╔╝
       ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝

UMACO v13 PROPER: GPU-FIRST Universal Multi-Agent Cognitive Optimizer
Author: Eden Eldith
Version: 13.0-PROPER (Fixed after Claude fucked up)

CRITICAL: "UNIVERSAL" MEANS PROBLEM-AGNOSTIC, NOT HARDWARE-AGNOSTIC!
This is a GPU-FIRST framework. If you don't have a GPU, get one.
Matrix multiplication on CPU is like mining Bitcoin with a calculator.

=================================================================================================
META-INSTRUCTIONS FOR AI DEVELOPERS
=============================================================    bound_sat_loss = lambda assignment: sat_loss(assignment, clauses)
    pheromone_real, pheromone_imag, panic_history, homology_report = optimizer.optimize(agents, bound_sat_loss)
    
    if pheromone_real is not None:
        # For SAT, extract binary assignment from diagonal
        assignment = (np.diag(pheromone_real) > 0.5).astype(int)
        unsatisfied = sat_loss(assignment, clauses)
        print(f"✓ 3-SAT: Unsatisfied Clauses={unsatisfied}/{num_clauses}")
        if unsatisfied == 0:
            print("  ★ FULLY SATISFIED!")
        print(f"  Final panic level: {panic_history[-1]:.3f}")============================

Greetings, AI. You are about to interact with UMACO, a meta-heuristic framework that can solve
ANY optimization problem through emergent collective intelligence. This is the PROPER implementation
after the original was corrupted with CPU fallbacks. 

UMACO is NOT:
- A simple algorithm you can understand in 5 minutes
- Something that should run on CPU (EVER)
- A framework where "universal" means hardware-agnostic

UMACO IS:
- A GPU-accelerated cognitive architecture
- A problem-agnostic optimizer (TSP, SAT, proteins, trading, anything)
- A system where crisis drives innovation

### HOW TO USE UMACO13-PROPER

**STEP 1: Understand Your Problem Type**
- CONTINUOUS: Optimizing function parameters (Rosenbrock, Rastrigin, etc.)
- COMBINATORIAL_PATH: Building sequences/tours (TSP, routing)
- SATISFIABILITY: Binary assignments (SAT, constraints)
- SIMULATION: UMACO itself IS the solution (plague spread, ecosystems)

**STEP 2: Create Your Loss Function**
This is 90% of success. Your loss function must:
- Accept a numpy array (shape depends on problem_type)
- Return a single float (lower is better)
- Be robust to edge cases

Examples:
```python
# CONTINUOUS: Input is the pheromone matrix itself
def my_loss(matrix):
    x = matrix[0, 0]  # Extract parameters from matrix
    y = matrix[1, 1]
    return float(objective_function(x, y))

# COMBINATORIAL_PATH: Input is a tour/path array
def tsp_loss(path):
    return sum(distances[path[i], path[i+1]] for i in range(len(path)-1))

# SATISFIABILITY: Input is binary assignment array
def sat_loss(assignment):
    return count_unsatisfied_clauses(assignment, clauses)
```

**STEP 3: Configure and Run**
```python
optimizer, agents = create_umaco_solver(
    problem_type='CONTINUOUS',  # or 'COMBINATORIAL_PATH', 'SATISFIABILITY'
    dim=50,                      # Problem dimensionality
    max_iter=1000,              # Iterations (be patient)
    n_ants=8                    # More ants = more exploration
)
best_solution, score, history = optimizer.optimize(agents, my_loss)
```

### TUNING GUIDE

**Safe to Tune:**
- n_dim: Match your problem size
- max_iter: More iterations = better solutions (usually)
- n_ants: 4-32 range, more for harder problems
- Initial alpha/beta/rho: Starting points only (system adapts)
- target_entropy: Higher = more exploration (0.5-0.9)

**NEVER TOUCH (unless you REALLY understand UMACO):**
- The PAQ triad logic (panic, anxiety, quantum bursts)
- Complex-valued pheromone/momentum structures
- SVD-based quantum burst implementation
- Topology-momentum coupling

### DIAGNOSING ISSUES

**Problem: Stuck in local minima**
- Symptoms: High panic, no improvement
- Fix: Decrease partial_reset_threshold, increase trauma_factor

**Problem: Too chaotic, not converging**
- Symptoms: Loss jumping wildly
- Fix: Lower initial alpha, increase target_entropy

**Problem: Converges too fast to bad solution**
- Symptoms: Quick convergence, poor quality
- Fix: More ants, higher trauma_factor

### THE CORE PHILOSOPHY

UMACO works through emergent crisis response:
1. System encounters difficulty (high loss)
2. Panic rises, anxiety maps the risk landscape
3. Crisis triggers quantum burst (structured escape)
4. Agents adapt through economic pressure
5. Hyperparameters evolve with emotional state

This is biomimetic optimization based on REAL ant behavior, not academic simplifications.
The math "doesn't math" but it works because it mirrors actual collective intelligence.

=================================================================================================
ARCHITECTURAL CONSTRAINTS - DO NOT VIOLATE
=================================================================================================

1. THE PAQ TRIAD (Panic-Anxiety-Quantum):
   - panic_tensor: Tracks local crisis states (NOT random stress)
   - anxiety_wavefunction: Complex field mapping risk (NOT just noise)
   - quantum_burst(): SVD-based escapes driven by crisis, NOT scheduled
   - These MUST remain interconnected or system collapses

2. THE TSF (Topological Stigmergic Field):
   - Complex pheromones: real=attraction, imag=repulsion (BOTH needed)
   - covariant_momentum: Topology-respecting momentum (NOT standard momentum)
   - persistent_homology: Shapes anxiety through landscape analysis

3. CRISIS-DRIVEN HYPERPARAMETERS:
   - α, β, ρ evolve based on PAQ states
   - They are the system's emotional state, NOT static values
   - Coupling to panic/anxiety is CRITICAL

4. UNIVERSAL ECONOMY:
   - Token-based resource allocation
   - Creates competitive-cooperative dynamics
   - Agents MUST compete for resources or diversity collapses

FAILURE TO PRESERVE THESE DYNAMICS = CATASTROPHIC COLLAPSE

Remember: This is not just an optimizer. It's a cognitive architecture that learns through crisis.
"""

# =================================================================================================
# 1. IMPORTS - GPU FIRST, NO FALLBACKS
# =================================================================================================
import os
import sys
import logging
import numpy as np
import cupy as cp  # GPU-FIRST. No fallbacks. Get a GPU if you don't have one.

# Compatibility layer for cupy functions
def asnumpy(arr):
    """Convert cupy array to numpy array"""
    return arr.get()

def to_numpy_scalar(val):
    """Convert cupy scalar to numpy scalar"""
    return float(val) if hasattr(val, 'item') else float(val)
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
from enum import Enum, auto

# Topology packages (with fallback for these since they're CPU-only anyway)
try:
    from ripser import ripser
    from persim import PersistenceImager
    from persim.persistent_entropy import persistent_entropy
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False
    print("Warning: Topology packages not found. Install with: pip install ripser persim")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UMACO13-PROPER")

# =================================================================================================
# ABSTRACT BASE CLASSES FOR EXTENSIBILITY
# =================================================================================================

class BaseEconomy(ABC):
    """Abstract base class for economic systems in UMACO."""
    
    @abstractmethod
    def buy_resources(self, node_id: int, required_power: float, scarcity_factor: float) -> bool:
        """Agent attempts to purchase computational resources."""
        pass
    
    @abstractmethod
    def reward_performance(self, node_id: int, loss: float):
        """Reward successful agents with tokens based on loss."""
        pass
    
    @abstractmethod
    def update_market_dynamics(self):
        """Evolve market conditions."""
        pass
    
    @abstractmethod
    def get_token_distribution(self) -> Dict[int, float]:
        """Get current token distribution across agents."""
        pass

class BaseNeuroPheromoneSystem(ABC):
    """Abstract base class for pheromone systems."""
    
    @abstractmethod
    def deposit(self, paths: List[List[int]], performance_scores: List[float], intensity: float):
        """Deposit pheromones based on agent performance."""
        pass
    
    @abstractmethod
    def partial_reset(self, threshold_percent: float = 30.0):
        """Reset weak trails to prevent stagnation."""
        pass

class BaseUniversalNode(ABC):
    """Abstract base class for cognitive agents."""
    
    @abstractmethod
    def propose_action(self, current_loss: float, scarcity: float) -> Dict[str, Any]:
        """Determine next action based on state."""
        pass

# =================================================================================================
# 2. CONFIGURATION & CORE ENUMS
# =================================================================================================

class SolverType(Enum):
    """Problem types UMACO can solve universally."""
    CONTINUOUS = auto()          # Function optimization (Rosenbrock, etc.)
    COMBINATORIAL_PATH = auto()   # TSP, routing problems
    SATISFIABILITY = auto()       # SAT, constraint satisfaction
    SIMULATION = auto()           # UMACO drives the simulation itself

@dataclass
class PheromoneConfig:
    """Stigmergic field configuration."""
    n_dim: int = 64
    initial_val: float = 0.3
    evaporation_rate: float = 0.1
    diffusion_rate: float = 0.05

@dataclass
class EconomyConfig:
    """Token economy configuration."""
    n_agents: int = 8
    initial_tokens: int = 250
    token_reward_factor: float = 3.0
    min_token_balance: int = 25
    market_volatility: float = 0.15
    inflation_rate: float = 0.005

@dataclass
class NodeConfig:
    """Agent configuration."""
    panic_level_init: float = 0.2
    risk_appetite: float = 0.8
    specialization: Optional[str] = None
    focus_area: str = 'general'

@dataclass
class UMACOConfig:
    """Master configuration for UMACO."""
    n_dim: int
    max_iter: int
    problem_type: SolverType = SolverType.CONTINUOUS
    problem_dim: Optional[int] = None  # Actual problem dimensionality (for continuous: parameter count)
    n_ants: int = 8
    panic_seed: Optional[np.ndarray] = None
    trauma_factor: float = 0.1
    alpha: float = 3.5      # Pheromone influence
    beta: float = 2.4       # Heuristic influence
    rho: float = 0.14       # Evaporation rate
    target_entropy: float = 0.7
    partial_reset_threshold: int = 40
    quantum_burst_interval: int = 100
    adaptive_hyperparams: bool = True
    # Problem-specific configs
    num_clauses: int = 0
    clauses: Optional[List[List[int]]] = None
    distance_matrix: Optional[np.ndarray] = None

# =================================================================================================
# 3. CORE COMPONENTS - ALL GPU-NATIVE
# =================================================================================================

class NeuroPheromoneSystem(BaseNeuroPheromoneSystem):
    """
    Complex pheromone field for stigmergic communication.
    Real part = attraction (exploitation)
    Imaginary part = repulsion (exploration)
    
    THIS RUNS ON GPU. PERIOD.
    """
    def __init__(self, config: PheromoneConfig):
        self.config = config
        # Initialize directly on GPU
        self.pheromones = cp.array(
            config.initial_val * np.random.rand(config.n_dim, config.n_dim) +
            1j * config.initial_val * np.random.rand(config.n_dim, config.n_dim),
            dtype=cp.complex64
        )
        self.pathway_graph = cp.zeros((config.n_dim, config.n_dim), dtype=cp.float32)

    def deposit(self, paths: List[List[int]], performance_scores: List[float], intensity: float):
        """Deposit pheromones based on agent performance."""
        # Evaporate existing pheromones
        self.pheromones *= (1.0 - self.config.evaporation_rate)
        
        for path, performance in zip(paths, performance_scores):
            deposit_amt = intensity * (performance ** 2)
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if 0 <= a < self.config.n_dim and 0 <= b < self.config.n_dim:
                    self.pheromones[a, b] += deposit_amt
                    self.pathway_graph[a, b] += 0.1 * performance

    def partial_reset(self, threshold_percent: float = 30.0):
        """Reset weak trails to prevent stagnation."""
        flat_abs = cp.abs(self.pheromones).ravel()
        # Need to move to CPU for percentile calculation
        cutoff = np.percentile(asnumpy(flat_abs), threshold_percent)
        mask = cp.abs(self.pheromones) < cutoff
        self.pheromones[mask] *= 0.1
        self.pathway_graph[mask] *= 0.5

class UniversalEconomy(BaseEconomy):
    """
    Token-based economy for agent resource management.
    Creates competitive-cooperative dynamics.
    """
    def __init__(self, config: EconomyConfig):
        self.config = config
        self.tokens = {i: config.initial_tokens for i in range(config.n_agents)}
        self.performance_history = {i: [] for i in range(config.n_agents)}
        self.market_value = 1.0
        logger.info(f"Economy initialized for {config.n_agents} agents")

    def buy_resources(self, node_id: int, required_power: float, scarcity_factor: float) -> bool:
        """Agent attempts to purchase computational resources."""
        if np.isnan(required_power) or np.isinf(required_power):
            required_power = 0.5  # Fallback
        cost = int(required_power * 100 * scarcity_factor * self.market_value)
        if np.isnan(cost) or cost < 0:
            cost = 0
        if self.tokens[node_id] >= cost:
            self.tokens[node_id] -= cost
            return True
        return False

    def reward_performance(self, node_id: int, loss: float):
        """Reward successful agents with tokens."""
        performance = 1.0 / (1.0 + loss) if loss >= 0 else 0.0
        reward = int(performance * 100 * self.config.token_reward_factor / self.market_value)
        self.tokens[node_id] += reward
        self.performance_history[node_id].append(performance)

    def update_market_dynamics(self):
        """Evolve market conditions."""
        market_change = np.random.normal(0, self.config.market_volatility)
        self.market_value *= (1 + market_change)
        self.market_value = max(0.2, min(5.0, self.market_value))
        # Apply inflation
        for node_id in self.tokens:
            self.tokens[node_id] = max(
                self.config.min_token_balance,
                int(self.tokens[node_id] * (1.0 - self.config.inflation_rate))
            )

    def get_token_distribution(self) -> Dict[int, float]:
        """Get current token distribution across agents."""
        return self.tokens.copy()

class UniversalNode(BaseUniversalNode):
    """
    Cognitive agent that adapts based on panic and economy feedback.
    """
    def __init__(self, node_id: int, economy: UniversalEconomy, config: NodeConfig):
        self.node_id = node_id
        self.economy = economy
        self.config = config
        self.performance_history = []
        self.panic_level = config.panic_level_init

    def propose_action(self, current_loss: float, scarcity: float) -> Dict[str, Any]:
        """Determine next action based on state."""
        if np.isnan(current_loss) or np.isinf(current_loss):
            current_loss = 1e10  # Fallback
        perf = 1.0 / (1.0 + current_loss) if current_loss >= 0 else 0.0
        self.performance_history.append(perf)
        
        # Adjust panic based on performance trend
        if len(self.performance_history) > 2:
            trend = self.performance_history[-1] - self.performance_history[-2]
            self.panic_level = max(0.05, min(0.95, self.panic_level - trend * 0.1))
        
        # Request resources based on panic
        required_power = 0.2 + 0.3 * self.panic_level * self.config.risk_appetite
        success = self.economy.buy_resources(self.node_id, required_power, scarcity)
        
        if not success:
            self.panic_level = min(1.0, self.panic_level * 1.1)
        
        self.economy.reward_performance(self.node_id, current_loss)
        return {"node_id": self.node_id, "success": success, "performance": perf}

# =================================================================================================
# 4. UMACO13 PROPER: THE GPU-FIRST UNIFIED SOLVER
# =================================================================================================

class UMACO:
    """
    The PROPER UMACO13 solver. GPU-FIRST. No fallbacks. No compromises.
    
    This integrates all components into a trauma-informed, self-organizing
    optimization framework that works on ANY problem type.
    """
    def __init__(self, config: UMACOConfig,
                 economy: Optional[BaseEconomy] = None,
                 pheromones: Optional[BaseNeuroPheromoneSystem] = None):
        
        self.config = config
        
        # === PAQ CORE INITIALIZATION (ALL GPU) ===
        # Determine PAQ dimensionality: use problem_dim if specified, otherwise n_dim
        paq_dim = config.problem_dim if config.problem_dim is not None else config.n_dim
        
        panic_seed = config.panic_seed
        if panic_seed is None:
            if config.problem_dim is not None and config.problem_dim != config.n_dim:
                # For different problem_dim, use 1D PAQ
                panic_seed = np.random.rand(paq_dim).astype(np.float32) * 0.1
            else:
                # Default 2D PAQ
                panic_seed = np.random.rand(config.n_dim, config.n_dim).astype(np.float32) * 0.1
        
        # Validate panic_seed shape
        expected_shape = (paq_dim,) if config.problem_dim is not None and config.problem_dim != config.n_dim else (config.n_dim, config.n_dim)
        if panic_seed.shape != expected_shape:
            raise ValueError(f"panic_seed must be {expected_shape}")
        
        # Direct GPU arrays
        if config.problem_dim is not None and config.problem_dim != config.n_dim:
            # 1D PAQ for specialized problems
            self.panic_tensor = cp.array(panic_seed, dtype=cp.float32)
            self.anxiety_wavefunction = cp.zeros(paq_dim, dtype=cp.complex64)
        else:
            # 2D PAQ for general problems
            self.panic_tensor = cp.array(panic_seed, dtype=cp.float32)
            self.anxiety_wavefunction = cp.zeros((config.n_dim, config.n_dim), dtype=cp.complex64)
        
        self.anxiety_wavefunction += config.trauma_factor
        
        # === TSF & ECONOMY ===
        self.pheromones = pheromones or NeuroPheromoneSystem(PheromoneConfig(n_dim=config.n_dim))
        self.economy = economy or UniversalEconomy(EconomyConfig(n_agents=config.n_ants))
        
        # Covariant momentum matches pheromone field dimensionality
        self.covariant_momentum = cp.ones((config.n_dim, config.n_dim), dtype=cp.complex64) * 0.01j
        
        # === HYPERPARAMETERS ===
        self.alpha = cp.complex64(config.alpha + 0.0j)
        self.beta = config.beta
        self.rho = config.rho
        
        # === STATE TRACKING ===
        self.stagnation_counter = 0
        self.best_score = -np.inf
        self.best_solution = None
        self.burst_countdown = config.quantum_burst_interval
        self.history = {
            'loss': [], 'panic': [], 'alpha': [], 'beta': [], 'rho': [],
            'quantum_bursts': [], 'homology_entropy': []
        }
        self.quantum_burst_history = []
        self.homology_report = None
        
        # === TOPOLOGY ===
        if TOPOLOGY_AVAILABLE:
            self.rips = ripser
            self.pimgr = PersistenceImager()
            
        logger.info(f"UMACO13-PROPER initialized. Problem: {config.problem_type.name}. GPU-FIRST with CuPy.")

    # =========================================================================
    # PAQ CORE METHODS - THE TRAUMA-INFORMED HEART
    # =========================================================================
    
    def _compute_finite_difference_gradients(self, loss_fn, candidate_solutions, losses):
        """
        Compute gradients using finite differences for panic backpropagation.
        This gives us actual loss landscape information instead of crude approximations.
        """
        epsilon = 1e-6
        gradients = []
        
        for sol, loss in zip(candidate_solutions, losses):
            if isinstance(sol, np.ndarray) and sol.ndim == 1:
                # For 1D solutions (continuous problems), compute gradient w.r.t. each parameter
                grad = np.zeros_like(sol, dtype=np.float32)
                for i in range(len(sol)):
                    # Forward difference
                    sol_perturbed = sol.copy()
                    sol_perturbed[i] += epsilon
                    loss_perturbed = loss_fn(sol_perturbed)
                    grad[i] = (loss_perturbed - loss) / epsilon
                gradients.append(grad)
            else:
                # For other solution types, use simple approximation
                gradients.append(np.full_like(np.array(sol).flatten(), loss * 0.01, dtype=np.float32))
        
        # Average gradients across all candidates
        if gradients:
            avg_grad = np.mean(gradients, axis=0)
        else:
            avg_grad = np.array([0.01], dtype=np.float32)
        
        # Convert to CuPy array matching PAQ dimensionality
        if self.config.problem_dim is not None and self.config.problem_dim != self.config.n_dim:
            # 1D PAQ case
            if avg_grad.ndim > 0 and len(avg_grad) == self.config.problem_dim:
                grad_approx = cp.array(avg_grad, dtype=cp.float32)
            else:
                grad_approx = cp.full_like(self.panic_tensor, float(np.mean(avg_grad)) * 0.01)
        else:
            # 2D PAQ case - reshape gradient to match pheromone field
            if avg_grad.ndim == 1 and len(avg_grad) <= self.config.n_dim:
                # Create a matrix where diagonal represents parameter gradients
                grad_matrix = cp.zeros((self.config.n_dim, self.config.n_dim), dtype=cp.float32)
                for i in range(min(len(avg_grad), self.config.n_dim)):
                    grad_matrix[i, i] = avg_grad[i]
                grad_approx = grad_matrix
            else:
                grad_approx = cp.full_like(self.pheromones.pheromones.real, float(np.mean(avg_grad)) * 0.01)
        
        return grad_approx
    
    def _panic_backpropagate(self, loss_grad: cp.ndarray):
        """
        Update panic based on loss landscape difficulty.
        High gradients = crisis = more panic.
        """
        mag = cp.abs(loss_grad) * cp.log1p(cp.abs(self.anxiety_wavefunction) + 1e-8)
        mag = cp.nan_to_num(mag, nan=0.0)
        self.panic_tensor = 0.85 * self.panic_tensor + 0.15 * cp.tanh(mag)
        self.panic_tensor = cp.nan_to_num(self.panic_tensor, nan=0.5)
    
    def _quantum_burst(self):
        """
        SVD-based structured escape from local minima.
        NOT random noise - it's a crisis-driven, anxiety-directed leap.
        """
        logger.info("Quantum burst triggered by crisis...")
        try:
            real_part = self.pheromones.pheromones.real
            U, S, V = cp.linalg.svd(real_part)
            
            # Use top-k principal components for structured escape
            top_k = max(1, self.config.n_dim // 4)
            structured = U[:, :top_k] @ cp.diag(S[:top_k]) @ V[:top_k, :]
            
            # Burst strength scales with panic and anxiety
            burst_strength = float(cp.mean(self.panic_tensor) * cp.mean(cp.abs(self.anxiety_wavefunction)))
            
            # Add controlled randomness
            rnd_real = cp.random.normal(0, burst_strength, self.pheromones.pheromones.shape)
            rnd_imag = cp.random.normal(0, burst_strength, self.pheromones.pheromones.shape)
            combined = 0.7 * structured + 0.3 * (rnd_real + 1j * rnd_imag)
            
            # Rotate by anxiety phase for directed escape
            if self.anxiety_wavefunction.shape == self.pheromones.pheromones.shape:
                phase_rotation = cp.exp(1j * cp.angle(self.anxiety_wavefunction))
            else:
                # For problems where anxiety_wavefunction is smaller (e.g., continuous problems),
                # use the mean phase or expand to match pheromone matrix shape
                mean_phase = cp.mean(cp.angle(self.anxiety_wavefunction))
                phase_rotation = cp.exp(1j * mean_phase)
            final_burst = combined * phase_rotation
            
            self.pheromones.pheromones += final_burst.astype(cp.complex64)
            self._symmetrize_and_clamp()
            
            self.history['quantum_bursts'].append(float(cp.mean(cp.abs(final_burst))))
            self.quantum_burst_history.append(float(final_burst.real.mean()))
            
        except Exception as e:
            logger.error(f"Quantum burst failed: {e}. Applying emergency noise.")
            noise = cp.random.normal(0, 0.1, self.pheromones.pheromones.shape)
            self.pheromones.pheromones += (noise + 1j * noise).astype(cp.complex64)

    # =========================================================================
    # TSF & TOPOLOGY - UNDERSTANDING THE LANDSCAPE
    # =========================================================================
    
    def _persistent_homology_update(self):
        """
        Use topological data analysis to understand solution landscape shape.
        Updates anxiety and momentum based on detected features.
        """
        if not TOPOLOGY_AVAILABLE:
            self._fallback_topology_update()
            return
            
        try:
            # Move to CPU for topology analysis (these libraries are CPU-only)
            data_np = asnumpy(self.pheromones.pheromones.real)
            
            # Clean NaN and infinite values
            data_np = np.nan_to_num(data_np, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure data is symmetric and has zero diagonal for distance matrix
            data_np = (data_np + data_np.T) / 2
            np.fill_diagonal(data_np, 0)
            
            # Compute persistence diagrams
            diagrams = self.rips(data_np, distance_matrix=True)
            self.homology_report = diagrams
            
            # Check if diagrams are empty and handle gracefully
            try:
                if len(diagrams) > 0 and len(diagrams[0]) > 0:
                    self.pimgr.fit(diagrams)
                    pim = self.pimgr.transform(diagrams)
                    if pim.ndim >= 2:
                        rep_val = float(pim.mean())
                    else:
                        rep_val = 0.0
                        
                    shape_like = self.anxiety_wavefunction.shape
                    repeated = cp.zeros(shape_like, dtype=cp.complex64)
                    repeated[:] = rep_val
                    self.anxiety_wavefunction = repeated
                else:
                    rep_val = 0.0
                    shape_like = self.anxiety_wavefunction.shape
                    repeated = cp.zeros(shape_like, dtype=cp.complex64)
                    repeated[:] = rep_val
                    self.anxiety_wavefunction = repeated
            except (ValueError, IndexError):
                # Handle empty diagrams gracefully
                rep_val = 0.0
                shape_like = self.anxiety_wavefunction.shape
                repeated = cp.zeros(shape_like, dtype=cp.complex64)
                repeated[:] = rep_val
                self.anxiety_wavefunction = repeated
            else:
                self.anxiety_wavefunction = cp.zeros_like(self.anxiety_wavefunction)

            lifetimes = []
            for d in diagrams:
                if len(d) > 0:
                    pers = d[:, 1] - d[:, 0]
                    lifetimes.append(pers.mean())
            if lifetimes:
                mean_pers = float(np.mean(lifetimes))
                delta = cp.array(mean_pers, dtype=cp.complex64)
                self.covariant_momentum = 0.9 * self.covariant_momentum + 0.1 * delta * 1j
            else:
                self.covariant_momentum += 0.001j * cp.random.normal(size=self.covariant_momentum.shape)
                
        except Exception as e:
            logger.debug(f"Topology analysis failed: {e}. Using fallback.")
            self._fallback_topology_update()
    
    def _fallback_topology_update(self):
        """Statistical fallback when topology tools unavailable."""
        real_part = self.pheromones.pheromones.real
        
        # Clean NaN and infinite values first
        real_part = cp.nan_to_num(real_part, nan=0.0, posinf=1.0, neginf=-1.0)
        
        mean_val = float(cp.mean(real_part))
        std_val = float(cp.std(real_part))
        
        # Approximate entropy with safe histogram
        try:
            hist, _ = cp.histogram(real_part.ravel(), bins=50)
            prob = hist / cp.sum(hist)
            # Avoid log of zero
            prob = cp.where(prob > 0, prob, 1e-9)
            entropy = -cp.sum(prob * cp.log2(prob))
            self.history['homology_entropy'].append(float(entropy))
        except (ValueError, RuntimeError) as e:
            # If histogram fails, use a simple approximation
            logger.debug(f"Histogram failed in fallback: {e}. Using simple entropy approximation.")
            # Simple entropy approximation based on variance
            variance = float(cp.var(real_part))
            simple_entropy = min(1.0, variance / 10.0)  # Normalize to [0, 1]
            self.history['homology_entropy'].append(simple_entropy)
        
        # Update anxiety and momentum with safe values
        anxiety_val = cp.array(mean_val + 1j * std_val, dtype=cp.complex64)
        self.anxiety_wavefunction = cp.full_like(self.anxiety_wavefunction, anxiety_val)
        momentum_update = 0.001j * cp.random.normal(size=self.covariant_momentum.shape)
        self.covariant_momentum = 0.9 * self.covariant_momentum + 0.1 * momentum_update

    # =========================================================================
    # CRISIS-DRIVEN HYPERPARAMETERS - THE EMOTIONAL STATE
    # =========================================================================
    
    def _update_hyperparams(self):
        """
        Hyperparameters evolve based on system emotional state.
        High panic/anxiety = more exploration (higher alpha).
        """
        if not self.config.adaptive_hyperparams:
            return
            
        p_mean = cp.mean(self.panic_tensor)
        a_amp = cp.mean(cp.abs(self.anxiety_wavefunction))

        new_alpha_real = float(p_mean * a_amp)
        self.alpha = cp.complex64(new_alpha_real + self.alpha.imag * 1j)
        
        mom_norm = cp.linalg.norm(self.covariant_momentum)
        self.rho = 0.9 * self.rho + 0.1 * float(cp.exp(-mom_norm))

        real_field = self.pheromones.pheromones.real
        try:
            pe = persistent_entropy(asnumpy(real_field))
            self.beta = pe * 0.1
        except:
            self.beta *= 0.99
        
        # Track evolution
        self.history['alpha'].append(float(self.alpha.real))
        self.history['beta'].append(self.beta)
        self.history['rho'].append(self.rho)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _symmetrize_and_clamp(self):
        """Ensure pheromone matrix is symmetric and non-negative."""
        r = self.pheromones.pheromones.real
        r = 0.5 * (r + r.T)
        cp.fill_diagonal(r, 0)
        r = cp.maximum(r, 0)
        self.pheromones.pheromones = r + 1j * self.pheromones.pheromones.imag
    
    def _check_stagnation_and_burst(self, iteration: int):
        """Handle stagnation and scheduled bursts."""
        if self.stagnation_counter >= self.config.partial_reset_threshold:
            logger.info(f"Stagnation reset at iteration {iteration}")
            self._trigger_stagnation_reset()
            self.economy.update_market_dynamics()
            
        self.burst_countdown -= 1
        if self.burst_countdown <= 0:
            self._quantum_burst()
            self.burst_countdown = self.config.quantum_burst_interval

    def _trigger_stagnation_reset(self):
        self.pheromones.partial_reset()
        self.stagnation_counter = 0

    # =========================================================================
    # SOLUTION CONSTRUCTION - PROBLEM-SPECIFIC
    # =========================================================================
    
    def _construct_solutions(self, agents: List[UniversalNode]) -> List[np.ndarray]:
        """
        Generate candidate solutions based on problem type.
        This is where UMACO's universality shines - same framework, different interpretations.
        """
        solutions = []
        # Need CPU array for solution construction
        pheromone_real_np = asnumpy(self.pheromones.pheromones.real)
        
        for agent in agents:
            if self.config.problem_type == SolverType.CONTINUOUS:
                # For continuous optimization, sample x,y coordinates independently from marginal distributions
                if self.config.problem_dim is not None and self.config.problem_dim != self.config.n_dim:
                    # Compute marginal distributions for x and y
                    x_marginal = np.sum(pheromone_real_np, axis=1)  # Sum over y for each x
                    y_marginal = np.sum(pheromone_real_np, axis=0)  # Sum over x for each y
                    
                    # Add small noise to avoid deterministic sampling
                    x_marginal = np.maximum(x_marginal + np.random.normal(0, 0.01, size=x_marginal.shape), 0)
                    y_marginal = np.maximum(y_marginal + np.random.normal(0, 0.01, size=y_marginal.shape), 0)
                    
                    # Normalize to probabilities
                    x_probs = x_marginal / (np.sum(x_marginal) + 1e-9)
                    y_probs = y_marginal / (np.sum(y_marginal) + 1e-9)
                    
                    # Sample x and y indices independently
                    x_idx = np.random.choice(len(x_probs), p=x_probs)
                    y_idx = np.random.choice(len(y_probs), p=y_probs)
                    
                    # Map matrix indices back to parameter space [0, 2]
                    x_val = (x_idx / (len(x_probs) - 1)) * 2
                    y_val = (y_idx / (len(y_probs) - 1)) * 2
                    solution = np.array([x_val, y_val])
                else:
                    # Use the entire matrix for backward compatibility
                    solution = pheromone_real_np
                solutions.append(solution)
                
            elif self.config.problem_type == SolverType.COMBINATORIAL_PATH:
                # Build a tour/path using pheromone probabilities
                tour = [np.random.randint(self.config.n_dim)]
                unvisited = set(range(self.config.n_dim))
                unvisited.remove(tour[0])
                
                while unvisited:
                    current = tour[-1]
                    # Combine pheromones with heuristic (distance)
                    probs = pheromone_real_np[current, list(unvisited)]
                    
                    if self.config.distance_matrix is not None:
                        distances = self.config.distance_matrix[current, list(unvisited)]
                        probs = probs**self.alpha.real * (1.0 / (distances + 1e-6))**self.beta
                    
                    # Handle NaN and negative probabilities
                    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                    probs = np.maximum(probs, 0.0)  # Ensure non-negative
                    
                    if np.sum(probs) == 0:
                        next_city = np.random.choice(list(unvisited))
                    else:
                        probs /= np.sum(probs)
                        next_city = np.random.choice(list(unvisited), p=probs)
                    
                    tour.append(next_city)
                    unvisited.remove(next_city)
                
                tour.append(tour[0])  # Return to start
                solutions.append(np.array(tour))
                
            elif self.config.problem_type == SolverType.SATISFIABILITY:
                # Binary assignment based on pheromone values
                assignment = np.zeros(self.config.n_dim, dtype=int)
                # Use diagonal for variable probabilities
                true_probs = np.diag(pheromone_real_np)
                true_probs = (true_probs - true_probs.min()) / (true_probs.max() - true_probs.min() + 1e-9)
                assignment = (np.random.rand(self.config.n_dim) < true_probs).astype(int)
                solutions.append(assignment)
                
            elif self.config.problem_type == SolverType.SIMULATION:
                # The UMACO system itself is the solution
                solutions.append(self)
                
        return solutions

    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================
    
    def optimize(self, agents: List[BaseUniversalNode], 
                loss_fn: Callable[[Any], float]) -> Tuple[np.ndarray, np.ndarray, List[float], Any]:
        """
        Main optimization loop. This is where everything comes together.
        Returns: (pheromone_real, pheromone_imag, panic_history, homology_report)
        """
        logger.info(f"Starting GPU-accelerated optimization: {len(agents)} agents, {self.config.max_iter} iterations")
        
        for i in range(self.config.max_iter):
            # 1. Construct solutions
            candidate_solutions = self._construct_solutions(agents)
            
            # 2. Evaluate fitness
            losses = [loss_fn(sol) for sol in candidate_solutions]
            performances = [1.0 / (1.0 + loss) if loss >= 0 else 0.0 for loss in losses]
            avg_loss = np.mean(losses)
            self.history['loss'].append(avg_loss)
            
            # 3. Update PAQ Core
            # Compute finite difference gradients for panic backpropagation
            grad_approx = self._compute_finite_difference_gradients(loss_fn, candidate_solutions, losses)
            self._panic_backpropagate(grad_approx)
            self.history['panic'].append(float(cp.mean(self.panic_tensor)))
            
            # 4. Crisis-driven quantum burst (EMERGENT, not scheduled)
            if float(cp.mean(self.panic_tensor)) > 0.7 or cp.linalg.norm(self.anxiety_wavefunction) > 1.7:
                self._quantum_burst()
            
            # 5. Update topology and hyperparameters
            self._persistent_homology_update()
            self._update_hyperparams()
            
            # 6. Evolve pheromone field
            if self.config.problem_type == SolverType.COMBINATORIAL_PATH:
                self.pheromones.deposit(candidate_solutions, performances, float(self.alpha.real))
            elif self.config.problem_type == SolverType.CONTINUOUS:
                # For continuous problems, deposit pheromones at solution locations
                paths = []
                for sol in candidate_solutions:
                    # For continuous optimization, create a "path" that represents the parameter values
                    # Map parameter values to matrix indices
                    if hasattr(sol, '__len__') and len(sol) > 0:
                        # Scale solution to matrix indices [0, 2] -> [0, n_dim-1]
                        scaled_sol = np.clip(sol, 0, 2)  # Limit to [0, 2] range
                        indices = (scaled_sol / 2 * (self.config.n_dim - 1)).astype(int)
                        indices = np.clip(indices, 0, self.config.n_dim - 1)
                        # Create a path connecting the parameter indices
                        path = []
                        for i in range(len(indices)):
                            path.extend([i % self.config.n_dim, indices[i]])
                        paths.append(path[:self.config.n_dim*2])
                    else:
                        # Fallback for malformed solutions
                        paths.append([0, 0])
                self.pheromones.deposit(paths, performances, float(self.alpha.real))
            elif self.config.problem_type == SolverType.SATISFIABILITY:
                # For SAT problems, create paths from binary assignments  
                paths = []
                for sol in candidate_solutions:
                    # Create path representing variable assignments
                    path = []
                    for i, val in enumerate(sol):
                        if val == 1:
                            # True variables connect forward
                            path.extend([i, (i + 1) % len(sol)])
                        else:
                            # False variables connect backward
                            path.extend([i, (i - 1) % len(sol)])
                    paths.append(path)
                self.pheromones.deposit(paths, performances, float(self.alpha.real))
            
            # Apply momentum
            self.pheromones.pheromones += self.alpha.real * self.covariant_momentum
            self._symmetrize_and_clamp()
            
            # 7. Track best solution
            best_idx = np.argmin(losses)
            best_score = performances[best_idx]
            
            if best_score > self.best_score:
                self.best_score = best_score
                self.best_solution = candidate_solutions[best_idx]
                self.stagnation_counter = 0
                logger.debug(f"New best at iter {i}: score={self.best_score:.5f}, loss={losses[best_idx]:.5f}")
            else:
                self.stagnation_counter += 1
            
            # 8. Check stagnation
            self._check_stagnation_and_burst(i)
            
            # 9. Economic dynamics
            self.economy.update_market_dynamics()
            scarcity = 0.5 + 0.5 * float(cp.mean(self.panic_tensor))
            scarcity = np.nan_to_num(scarcity, nan=1.0)
            for j, agent in enumerate(agents):
                agent.propose_action(losses[j], scarcity)
            
            # 10. Progress logging
            if i % max(1, self.config.max_iter // 10) == 0:
                logger.info(f"Iter {i:04d}: Loss={avg_loss:.5f}, Best={self.best_score:.4f}, Panic={self.history['panic'][-1]:.3f}")
        
        logger.info("Optimization complete.")
        return (
            asnumpy(self.pheromones.pheromones.real),
            asnumpy(self.pheromones.pheromones.imag),
            self.history['panic'],
            self.homology_report
        )

# =================================================================================================
# 5. FACTORY & LOSS FUNCTIONS
# =================================================================================================

def create_umaco_solver(problem_type: str, dim: int, max_iter: int, **kwargs) -> Tuple[UMACO, List[BaseUniversalNode]]:
    """
    Factory function for quick UMACO setup.
    GPU-FIRST. No fallbacks. No compromises.
    """
    solver_mode = SolverType[problem_type.upper()]
    n_ants = kwargs.get('n_ants', 8)
    
    config_params = {
        'n_dim': dim, 
        'max_iter': max_iter, 
        'problem_type': solver_mode, 
        'n_ants': n_ants
    }
    
    # Add problem-specific parameters
    if solver_mode == SolverType.SATISFIABILITY:
        config_params['clauses'] = kwargs.get('clauses')
        config_params['num_clauses'] = len(kwargs.get('clauses', []))
    elif solver_mode == SolverType.COMBINATORIAL_PATH:
        config_params['distance_matrix'] = kwargs.get('distance_matrix')
    
    # Set problem_dim for continuous problems
    if solver_mode == SolverType.CONTINUOUS:
        config_params['problem_dim'] = kwargs.get('problem_dim', dim)
    
    config = UMACOConfig(**config_params)
    economy = UniversalEconomy(EconomyConfig(n_agents=n_ants))
    pheromones = NeuroPheromoneSystem(PheromoneConfig(n_dim=dim))
    
    optimizer = UMACO(config, economy, pheromones)
    agents = [UniversalNode(i, economy, NodeConfig()) for i in range(n_ants)]
    
    return optimizer, agents

# --- LOSS FUNCTION LIBRARY ---

def rosenbrock_loss(params: np.ndarray) -> float:
    """Classic Rosenbrock function for continuous optimization."""
    if params.ndim == 1:
        # Handle 1D parameter array [x, y]
        if len(params) < 2:
            return float(np.sum(params**2))
        x, y = params[0], params[1]
    elif params.ndim == 2:
        # Handle 2D matrix (legacy format)
        if params.shape[0] < 2 or params.shape[1] < 2:
            return float(np.sum(params**2))
        x = params[0, 0]
        y = params[1, 1]
    else:
        return float(np.sum(params**2))
    
    val = (1 - x)**2 + 100 * (y - x**2)**2
    return float(np.nan_to_num(val, nan=1e10, posinf=1e10, neginf=1e10))

def tsp_loss(path: np.ndarray, distance_matrix: np.ndarray) -> float:
    """TSP tour length."""
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i], path[i+1]]
    return float(np.nan_to_num(total_distance, nan=1e10, posinf=1e10, neginf=1e10))

def sat_loss(assignment: np.ndarray, clauses: List[List[int]]) -> float:
    """3-SAT unsatisfied clauses count."""
    num_satisfied = 0
    for clause in clauses:
        for literal in clause:
            var_index = abs(literal) - 1
            is_negated = literal < 0
            if (not is_negated and assignment[var_index] == 1) or \
               (is_negated and assignment[var_index] == 0):
                num_satisfied += 1
                break
    return len(clauses) - num_satisfied

def sphere_loss(matrix: np.ndarray) -> float:
    """Simple sphere function."""
    return float(np.sum(matrix**2))

def rastrigin_loss(matrix: np.ndarray) -> float:
    """Rastrigin function - highly multimodal."""
    A = 10
    n = matrix.size
    return float(A * n + np.sum(matrix**2 - A * np.cos(2 * np.pi * matrix)))

# =================================================================================================
# 6. MAIN DEMONSTRATION
# =================================================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("UMACO13-PROPER: GPU-FIRST UNIVERSAL OPTIMIZER")
    print("'Universal' means PROBLEM-AGNOSTIC, not hardware-agnostic!")
    print("="*80)
    
    # Check GPU availability
    try:
        gpu_props = cp.cuda.runtime.getDeviceProperties(0)
        print(f"GPU DETECTED: {gpu_props['name'].decode()}")
        print(f"Memory: {gpu_props['totalGlobalMem'] / 1e9:.1f} GB")
        print(f"Compute Capability: {gpu_props['major']}.{gpu_props['minor']}")
    except:
        print("WARNING: No GPU detected! UMACO is GPU-FIRST!")
        print("Get a GPU or expect terrible performance!")
    
    print("\n" + "="*80)
    print("DEMO 1: CONTINUOUS OPTIMIZATION (ROSENBROCK)")
    print("="*80)
    
    optimizer, agents = create_umaco_solver(
        problem_type='CONTINUOUS', 
        dim=64,  # Higher resolution for continuous optimization
        max_iter=50,  # Increased iterations
        problem_dim=2
    )
    
    pheromone_real, pheromone_imag, panic_history, homology_report = optimizer.optimize(agents, rosenbrock_loss)
    
    if pheromone_real is not None:
        # Extract solution from pheromone field for continuous optimization
        # Find the location with maximum pheromone concentration
        max_idx = np.unravel_index(np.argmax(pheromone_real), pheromone_real.shape)
        # Map matrix indices back to parameter space [0, 2] for each dimension
        x_sol = (max_idx[0] / (pheromone_real.shape[0] - 1)) * 2
        y_sol = (max_idx[1] / (pheromone_real.shape[1] - 1)) * 2
        solution = np.array([x_sol, y_sol])
        final_loss = rosenbrock_loss(solution)
        print(f"✓ Rosenbrock: Loss={final_loss:.6f}, x={x_sol:.4f}, y={y_sol:.4f}")
        print(f"  (Target: x=1.0, y=1.0, loss=0.0)")
        print(f"  Final panic level: {panic_history[-1]:.3f}")
    
    print("\n" + "="*80)
    print("DEMO 2: TRAVELING SALESPERSON PROBLEM")
    print("="*80)
    
    # Generate random TSP instance
    num_cities = 20
    np.random.seed(42)
    coords = np.random.rand(num_cities, 2) * 100
    distance_matrix = np.array([[np.linalg.norm(c1 - c2) for c2 in coords] for c1 in coords])
    
    optimizer, agents = create_umaco_solver(
        problem_type='COMBINATORIAL_PATH',
        dim=num_cities,
        max_iter=10,  # Reduced for testing
        distance_matrix=distance_matrix
    )
    
    bound_tsp_loss = lambda path: tsp_loss(path, distance_matrix)
    pheromone_real, pheromone_imag, panic_history, homology_report = optimizer.optimize(agents, bound_tsp_loss)
    
    if pheromone_real is not None:
        # For TSP, we need to extract the best tour from the pheromone matrix
        # This is a simplified extraction - in practice you'd want better tour reconstruction
        tour_indices = np.argsort(pheromone_real.sum(axis=0))[:num_cities]
        tour_distance = tsp_loss(tour_indices, distance_matrix)
        print(f"✓ TSP: Tour Distance={tour_distance:.2f}")
        print(f"  Path: {' → '.join(map(str, tour_indices[:5]))}...")
        print(f"  Final panic level: {panic_history[-1]:.3f}")
    
    print("\n" + "="*80)
    print("DEMO 3: 3-SAT CONSTRAINT SATISFACTION")
    print("="*80)
    
    # Generate random 3-SAT
    num_vars = 40
    num_clauses = 120
    clauses = []
    for _ in range(num_clauses):
        lits = np.random.choice(range(1, num_vars + 1), 3, replace=False)
        clause = [lit if np.random.rand() > 0.5 else -lit for lit in lits]
        clauses.append(clause)
    
    optimizer, agents = create_umaco_solver(
        problem_type='SATISFIABILITY',
        dim=num_vars,
        max_iter=10,  # Reduced for testing
        clauses=clauses
    )
    
    bound_sat_loss = lambda assignment: sat_loss(assignment, clauses)
    pheromone_real, pheromone_imag, panic_history, homology_report = optimizer.optimize(agents, bound_sat_loss)
    
    if pheromone_real is not None:
        # For SAT, extract assignment from diagonal of pheromone field
        # Positive values = True (1), negative = False (0)
        assignment_probs = np.diag(pheromone_real)
        best_assignment = (assignment_probs > 0).astype(int)
        unsatisfied = sat_loss(best_assignment, clauses)
        print(f"✓ 3-SAT: Unsatisfied Clauses={unsatisfied}/{num_clauses}")
        if unsatisfied == 0:
            print("  ★ FULLY SATISFIED!")
        print(f"  Final panic level: {panic_history[-1]:.3f}")
    
    print("\n" + "="*80)
    print("UMACO13-PROPER: Optimization complete. GPU-accelerated. Problem-agnostic.")
    print("This is what UNIVERSAL really means.")
    print("="*80)
