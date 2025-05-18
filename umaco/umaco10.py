#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UMACO: Universal Multi-Agent Cognitive Optimization
==================================================

A complex optimization framework integrating Panic-Anxiety-Quantum dynamics, 
topological stigmergy, and multi-agent economics.

The UMACO framework implements a novel optimization approach based on the following interconnected systems:

1. PAQ Core (Panic-Anxiety-Quantum Triad System):
   - Tracks crisis states and existential risk gradients
   - Implements quantum perturbations to escape local minima
   - Backpropagates panic via loss landscape curvature

2. Topological Stigmergic Field:
   - Uses complex pheromones for attraction/repulsion signaling
   - Employs topological data analysis for landscape understanding
   - Maintains momentum based on topological structure

3. Universal Economy:
   - Token-based resource management between agents
   - Dynamic market forces with volatility
   - Performance-based reward mechanisms

4. Crisis-Driven Hyperparameters:
   - Parameters respond dynamically to system state
   - Exploration/exploitation balance adapts to landscape complexity

UMACO is designed for complex, non-convex optimization problems with numerous local minima,
where traditional methods might struggle. It can be adapted to various domains by
customizing the agents and loss functions.

Author: Original UMACO framework by [Author]
Version: 10.0
"""

import os
import logging
import numpy as np
import cupy as cp
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Tuple, Optional, Union
import warnings

try:
    from ripser import Rips
    from persim import PersistenceImager
    from persim.persistent_entropy import persistent_entropy
    TOPOLOGY_AVAILABLE = True
except ImportError:
    warnings.warn("Topological analysis packages (ripser, persim) not found. "
                 "Fallback mechanisms will be used. For full functionality, install with: "
                 "pip install ripser persim", ImportWarning)
    TOPOLOGY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UMACO")

###############################################################################
#                         UNIVERSAL ECONOMY & AGENTS                          #
###############################################################################

@dataclass
class EconomyConfig:
    """
    Configuration for the Universal Economy system.
    
    The economy manages token distribution, market dynamics, and resource allocation
    between cognitive agents in the UMACO system.
    
    Attributes:
        n_agents (int): Number of cognitive agents participating in the economy.
        initial_tokens (int): Starting token balance for each agent.
        token_reward_factor (float): Multiplier for performance-based rewards.
        min_token_balance (int): Minimum token floor to prevent agent starvation.
        market_volatility (float): Standard deviation of random market fluctuations.
        inflation_rate (float): Rate at which token value decreases over time.
    """
    n_agents: int = 8
    initial_tokens: int = 250
    token_reward_factor: float = 3.0
    min_token_balance: int = 24
    market_volatility: float = 0.15
    inflation_rate: float = 0.005


class UniversalEconomy:
    """
    Token-based economy for computational and generalized resources in UMACO.
    
    Manages a marketplace where agents purchase computational resources and receive
    token rewards based on their performance. Includes dynamic market forces and 
    partial resets to ensure lively trading and prevent monopolies.
    """
    
    def __init__(self, config: EconomyConfig):
        """
        Initialize the Universal Economy with the specified configuration.
        
        Args:
            config (EconomyConfig): Configuration parameters for the economy.
        """
        self.config = config
        self.tokens = {i: config.initial_tokens for i in range(config.n_agents)}
        self.performance_history = {i: [] for i in range(config.n_agents)}
        self.market_value = 1.0
        self.market_volatility = config.market_volatility
        self.inflation_rate = config.inflation_rate
        logger.info(f"Economy initialized with {config.n_agents} agents, {config.initial_tokens} initial tokens each")

    def buy_resources(self, node_id: int, required_power: float, scarcity_factor: float) -> bool:
        """
        Process a resource purchase request from an agent.
        
        Agents spend tokens to acquire computational resources, with cost influenced
        by the scarcity factor and required power.
        
        Args:
            node_id (int): Identifier of the purchasing agent.
            required_power (float): Amount of computational power requested.
            scarcity_factor (float): Current resource scarcity multiplier.
            
        Returns:
            bool: True if purchase was successful, False if insufficient tokens.
        """
        cost = int(required_power * 100 * scarcity_factor)
        if self.tokens[node_id] >= cost:
            self.tokens[node_id] -= cost
            if self.tokens[node_id] < self.config.min_token_balance:
                self.tokens[node_id] = self.config.min_token_balance
            return True
        return False

    def reward_performance(self, node_id: int, performance: float):
        """
        Reward an agent with tokens based on their performance.
        
        Better-performing agents receive more tokens, creating a feedback loop
        that allocates more resources to successful strategies.
        
        Args:
            node_id (int): Identifier of the agent to reward.
            performance (float): Performance metric, higher is better.
        """
        reward = int(performance * 100 * self.config.token_reward_factor)
        self.tokens[node_id] += reward
        if self.tokens[node_id] < self.config.min_token_balance:
            self.tokens[node_id] = self.config.min_token_balance
        self.performance_history[node_id].append(performance)

    def update_market_dynamics(self):
        """
        Update market forces including volatility and inflation.
        
        Simulates natural market fluctuations to prevent the economy from
        reaching a static equilibrium, encouraging continuous adaptation.
        """
        market_change = np.random.normal(0, self.market_volatility)
        self.market_value *= (1 + market_change)
        self.market_value = max(0.2, min(5.0, self.market_value))
        
        # Apply inflation to prevent token hoarding
        for node_id in self.tokens:
            self.tokens[node_id] = max(
                self.config.min_token_balance,
                int(self.tokens[node_id] * (1.0 - self.inflation_rate))
            )

    def get_token_distribution(self) -> Dict[int, float]:
        """
        Calculate the fractional distribution of tokens across agents.
        
        Returns:
            Dict[int, float]: Mapping from agent ID to their fraction of total tokens.
        """
        total = sum(self.tokens.values()) or 1.0
        return {k: v/total for k, v in self.tokens.items()}
    
    def redistribute_tokens(self, percentage: float = 0.2):
        """
        Redistribute a percentage of tokens to ensure market liquidity.
        
        Args:
            percentage (float): Fraction of total tokens to redistribute.
        """
        total_tokens = sum(self.tokens.values())
        redistribution_pool = int(total_tokens * percentage)
        
        # Remove tokens proportionally
        for node_id in self.tokens:
            removal = int(self.tokens[node_id] * percentage)
            self.tokens[node_id] -= removal
            
        # Distribute evenly
        per_node = redistribution_pool // len(self.tokens)
        for node_id in self.tokens:
            self.tokens[node_id] += per_node


@dataclass
class NodeConfig:
    """
    Configuration for cognitive nodes (agents) in the UMACO system.
    
    Attributes:
        panic_level_init (float): Initial panic level for the node.
        risk_appetite (float): How willing the node is to take risks (0-1).
        specialization (str, optional): Area of specialization for the node.
    """
    panic_level_init: float = 0.2
    risk_appetite: float = 0.8
    specialization: Optional[str] = None


class UniversalNode:
    """
    Cognitive node (agent) in the UMACO optimization system.
    
    Each node adapts its resource usage and strategy based on internal
    panic level and reward feedback from the environment.
    """
    
    def __init__(self, node_id: int, economy: UniversalEconomy, config: NodeConfig):
        """
        Initialize a cognitive node.
        
        Args:
            node_id (int): Unique identifier for the node.
            economy (UniversalEconomy): Reference to the shared economy.
            config (NodeConfig): Configuration parameters for this node.
        """
        self.node_id = node_id
        self.economy = economy
        self.config = config
        self.performance_history = []
        self.panic_level = config.panic_level_init
        self.last_action_success = True
        self.specialization = config.specialization
        logger.debug(f"Node {node_id} initialized with panic={config.panic_level_init}, "
                    f"risk={config.risk_appetite}")

    def propose_action(self, current_loss: float, scarcity: float) -> Dict[str, Any]:
        """
        Determine next action based on current state and performance history.
        
        The node evaluates its current performance, adjusts its panic level,
        and requests computational resources based on its strategy.
        
        Args:
            current_loss (float): Current optimization loss value.
            scarcity (float): Current resource scarcity factor.
            
        Returns:
            Dict[str, Any]: Information about the proposed action.
        """
        # Calculate performance (inverse of loss)
        perf = 1.0 / (1.0 + current_loss)
        self.performance_history.append(perf)
        
        # Adjust panic level based on performance trend
        if len(self.performance_history) > 2:
            trend = self.performance_history[-1] - self.performance_history[-3]
            self.panic_level = max(0.05, min(0.95, self.panic_level - trend))

        # Determine resource requirement based on panic and risk appetite
        required_power = 0.2 + 0.3 * self.panic_level * self.config.risk_appetite
        
        # Attempt to purchase resources
        success = self.economy.buy_resources(self.node_id, required_power, scarcity)
        if not success:
            self.panic_level *= 1.1  # Increase panic if resource acquisition fails
            self.last_action_success = False
        else:
            self.last_action_success = True
        
        # Receive reward based on performance
        self.economy.reward_performance(self.node_id, perf)
        
        return {
            "node_id": self.node_id,
            "panic_level": self.panic_level,
            "required_power": required_power,
            "success": success,
            "performance": perf,
            "tokens": self.economy.tokens[self.node_id]
        }
    
    def get_average_performance(self, window: int = 5) -> float:
        """
        Calculate average performance over recent history.
        
        Args:
            window (int): Number of recent performance values to average.
            
        Returns:
            float: Average performance over the specified window.
        """
        if not self.performance_history:
            return 0.0
        
        recent = self.performance_history[-min(window, len(self.performance_history)):]
        return sum(recent) / len(recent)


###############################################################################
#                     NEUROPHEROMONE + STIGMERGY SYSTEM                       #
###############################################################################

@dataclass
class PheromoneConfig:
    """
    Configuration for the NeuroPheromone system.
    
    Attributes:
        n_dim (int): Dimensionality of the pheromone field.
        initial_val (float): Initial value for pheromone concentrations.
        evaporation_rate (float): Rate at which pheromones evaporate.
        diffusion_rate (float, optional): Rate of pheromone diffusion to neighbors.
    """
    n_dim: int = 64
    initial_val: float = 0.3
    evaporation_rate: float = 0.1
    diffusion_rate: float = 0.05


class NeuroPheromoneSystem:
    """
    Manages a 2D complex pheromone field for UMACO optimization.
    
    Implements deposit, evaporation, diffusion, and partial reset mechanisms.
    The real part of pheromones represents attraction, and the imaginary part
    represents repulsion.
    """
    
    def __init__(self, config: PheromoneConfig):
        """
        Initialize the NeuroPheromone system.
        
        Args:
            config (PheromoneConfig): Configuration parameters.
        """
        self.config = config
        self.pheromones = cp.array(
            config.initial_val * cp.ones((config.n_dim, config.n_dim)),
            dtype=cp.complex64
        )
        self.pathway_graph = cp.array(
            0.01 * cp.ones((config.n_dim, config.n_dim)),
            dtype=cp.float32
        )
        self.evaporation_rate = config.evaporation_rate
        self.diffusion_rate = config.diffusion_rate
        logger.info(f"Pheromone system initialized with dimensions {config.n_dim}x{config.n_dim}")

    def deposit_pheromones(self, paths: List[List[int]], performance: float, intensity: float):
        """
        Deposit pheromones along specified paths based on performance.
        
        Better-performing paths receive stronger pheromone deposits, creating
        a memory of successful navigation through the optimization landscape.
        
        Args:
            paths (List[List[int]]): List of paths where each path is a list of node indices.
            performance (float): Performance metric determining deposit strength.
            intensity (float): Additional scaling factor for deposit amount.
        """
        # Apply evaporation
        self.pheromones *= (1.0 - self.evaporation_rate)
        
        # Calculate deposit amount based on performance
        deposit_amt = intensity * (performance ** 1.3)
        
        # Deposit along each path
        for path in paths:
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                if a < self.config.n_dim and b < self.config.n_dim:
                    self.pheromones[a, b] += deposit_amt
                    self.pathway_graph[a, b] += 0.1 * performance

    def apply_diffusion(self):
        """
        Apply diffusion to spread pheromones to neighboring cells.
        
        This creates smoother gradients in the pheromone field, helping
        agents navigate more effectively.
        """
        # Simple discrete diffusion
        original = self.pheromones.copy()
        n = self.config.n_dim
        
        for i in range(n):
            for j in range(n):
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        neighbors.append((ni, nj))
                
                if neighbors:
                    diffusion = sum(original[ni, nj] for ni, nj in neighbors) / len(neighbors)
                    self.pheromones[i, j] = (1.0 - self.diffusion_rate) * self.pheromones[i, j] + \
                                          self.diffusion_rate * diffusion

    def partial_reset(self, threshold_percent: float = 30.0):
        """
        Reset weaker pheromone trails to encourage exploration of new paths.
        
        Args:
            threshold_percent (float): Percentile threshold below which to reset pheromones.
        """
        flattened = cp.abs(self.pheromones).ravel()
        cutoff = np.percentile(cp.asnumpy(flattened), threshold_percent)
        mask = cp.where(cp.abs(self.pheromones) < cutoff)
        for x, y in zip(mask[0], mask[1]):
            self.pheromones[x, y] = 0.01
            self.pathway_graph[x, y] *= 0.5
    
    def get_strongest_paths(self, top_k: int = 3) -> List[Tuple[int, int, float]]:
        """
        Identify the strongest pheromone paths in the system.
        
        Args:
            top_k (int): Number of top paths to return.
            
        Returns:
            List[Tuple[int, int, float]]: List of (source, destination, strength) tuples.
        """
        flat_idx = cp.abs(self.pheromones).ravel().argsort()[-top_k:]
        n_dim = self.config.n_dim
        
        results = []
        for idx in flat_idx:
            i, j = idx // n_dim, idx % n_dim
            strength = float(cp.abs(self.pheromones[i, j]))
            results.append((int(i), int(j), strength))
        
        return sorted(results, key=lambda x: x[2], reverse=True)


###############################################################################
#                          UMACO10: UNIVERSAL SOLVER                          #
###############################################################################

@dataclass
class UMACO10Config:
    """
    Configuration for the UMACO10 universal optimizer.
    
    Attributes:
        n_dim (int): Dimensionality of the optimization space.
        panic_seed (np.ndarray): Initial panic tensor values.
        trauma_factor (float): Scaling factor for initial anxiety.
        alpha (float): Learning rate parameter.
        beta (float): Entropy regulation parameter.
        rho (float): Momentum decay parameter.
        max_iter (int): Maximum number of optimization iterations.
        target_entropy (float): Target entropy for self-regulation.
        partial_reset_threshold (int): Iterations without improvement before reset.
        quantum_burst_interval (int): Iterations between scheduled quantum bursts.
        adaptive_hyperparams (bool): Whether to dynamically adjust hyperparameters.
        use_gpu (bool): Whether to use GPU acceleration via CuPy.
    """
    n_dim: int
    panic_seed: np.ndarray
    trauma_factor: float
    alpha: float
    beta: float
    rho: float
    max_iter: int
    target_entropy: float = 0.68
    partial_reset_threshold: int = 40
    quantum_burst_interval: int = 100
    adaptive_hyperparams: bool = True
    use_gpu: bool = True


class UMACO10:
    """
    Universal Multi-Agent Cognitive Optimization, version 10.
    
    UMACO10 merges multiple interconnected systems:
    1) PAQ Core (Panic-Anxiety-Quantum Triad System)
    2) Topological Stigmergy Field
    3) Crisis-Driven Hyperparameters
    4) Universal Economy & Multi-Agent interplay
    
    This framework is designed for complex, non-convex optimization problems
    with numerous local optima where traditional methods might struggle.
    """

    def __init__(self, config: UMACO10Config, 
                economy: Optional[UniversalEconomy] = None, 
                pheromones: Optional[NeuroPheromoneSystem] = None):
        """
        Initialize the UMACO10 universal solver.
        
        Args:
            config (UMACO10Config): Configuration parameters.
            economy (UniversalEconomy, optional): Economy system or None to create a new one.
            pheromones (NeuroPheromoneSystem, optional): Pheromone system or None to create a new one.
            
        Raises:
            ValueError: If panic_seed dimensions don't match n_dim.
            RuntimeError: If use_gpu=True but CuPy is not available.
        """
        # PAQ Core
        self.config = config
        if config.panic_seed.shape != (config.n_dim, config.n_dim):
            raise ValueError(f"panic_seed must have shape ({config.n_dim}, {config.n_dim}), "
                            f"got {config.panic_seed.shape}")
        
        # Check GPU availability if requested
        if config.use_gpu and not hasattr(cp, 'array'):
            raise RuntimeError("GPU acceleration requested but CuPy is not available. "
                              "Install CuPy or set use_gpu=False.")
        
        # Initialize array library based on GPU setting
        self.xp = cp if config.use_gpu else np
        
        # Initialize PAQ Core components
        self.panic_tensor = self.xp.array(config.panic_seed, dtype=self.xp.float32)
        self.anxiety_wavefunction = self.xp.zeros((config.n_dim, config.n_dim), 
                                                dtype=self.xp.complex64)
        self.anxiety_wavefunction *= config.trauma_factor

        # Initialize or use provided systems
        self.pheromones = pheromones or NeuroPheromoneSystem(
            PheromoneConfig(n_dim=config.n_dim)
        )
        self.economy = economy or UniversalEconomy(
            EconomyConfig(n_agents=min(8, config.n_dim))
        )
        
        # Initialize topological components if available
        if TOPOLOGY_AVAILABLE:
            self.covariant_momentum = self.xp.ones((config.n_dim, config.n_dim), 
                                                 dtype=self.xp.complex64) * 0.01j
            self.rips = Rips()
            self.pimgr = PersistenceImager()
        else:
            # Fallback if topology packages not available
            self.covariant_momentum = self.xp.ones((config.n_dim, config.n_dim), 
                                                 dtype=self.xp.complex64) * 0.01j
            logger.warning("Topological analysis unavailable. Using fallback mechanisms.")

        # Hyperparameters
        self.alpha = self.xp.complex64(config.alpha + 0.0j)
        self.beta = config.beta
        self.rho = config.rho

        # History tracking
        self.max_iter = config.max_iter
        self.target_entropy = config.target_entropy
        self.quantum_burst_history = []
        self.panic_history = []
        self.loss_history = []
        self.alpha_history = []
        self.beta_history = []
        self.rho_history = []

        # Optimization state tracking
        self.stagnation_counter = 0
        self.best_score = -np.inf
        self.best_solution = None
        self.burst_countdown = config.quantum_burst_interval
        self.homology_report = None
        
        logger.info(f"UMACO10 initialized with n_dim={config.n_dim}, max_iter={config.max_iter}")

    ###########################################################################
    #                            PAQ CORE METHODS                             #
    ###########################################################################

    def panic_backpropagate(self, loss_grad: Union[np.ndarray, cp.ndarray]):
        """
        Update panic tensor based on loss gradient and anxiety.
        
        This function implements the core feedback mechanism where optimization
        difficulties (represented by loss gradients) increase local panic levels,
        which in turn influence exploration strategies.
        
        Args:
            loss_grad (Union[np.ndarray, cp.ndarray]): Gradient of the loss function.
        """
        # Ensure gradient is on the correct device
        if isinstance(loss_grad, np.ndarray) and self.config.use_gpu:
            loss_grad = cp.array(loss_grad)
        elif isinstance(loss_grad, cp.ndarray) and not self.config.use_gpu:
            loss_grad = np.array(loss_grad)
            
        # Calculate panic increase based on gradient magnitude and anxiety
        mag = self.xp.abs(loss_grad) * self.xp.log1p(self.xp.abs(self.anxiety_wavefunction) + 1e-8)
        
        # Update panic with smooth blending
        self.panic_tensor = 0.85 * self.panic_tensor + 0.15 * self.xp.tanh(mag)

    def quantum_burst(self):
        """
        Execute a quantum burst to escape local minima.
        
        The burst combines structured components (via SVD) with random noise,
        modulated by the anxiety wavefunction. This mechanism enables controlled
        yet stochastic exploration of the search space.
        """
        logger.info("Executing quantum burst")
        
        # Perform SVD on the real part of pheromones
        try:
            U, S, V = self.xp.linalg.svd(self.pheromones.pheromones.real)
            
            # Use top-k components for structured perturbation
            top_k = 3 if S.shape[0] >= 3 else S.shape[0]
            structured = U[:, :top_k] @ self.xp.diag(S[:top_k]) @ V[:top_k, :]
            
            # Calculate burst strength based on panic and anxiety
            burst_strength = float(self.xp.linalg.norm(self.panic_tensor) * 
                                 self.xp.abs(self.anxiety_wavefunction).mean())
            
            # Generate random components
            rnd_real = self.xp.random.normal(0, burst_strength, self.pheromones.pheromones.shape)
            rnd_imag = self.xp.random.normal(0, burst_strength, self.pheromones.pheromones.shape)
            
            # Combine structured and random components
            combined_burst = 0.7 * structured + 0.3 * (rnd_real + 1j*rnd_imag)
            
            # Rotate by anxiety wavefunction phase
            phase = self.xp.exp(1j * self.xp.angle(self.anxiety_wavefunction))
            final_burst = combined_burst * phase
            
            # Apply to pheromones
            self.pheromones.pheromones += final_burst.astype(self.xp.complex64)
            self._symmetrize_clamp()
            
            # Record burst history
            self.quantum_burst_history.append(float(final_burst.real.mean()))
            
        except Exception as e:
            logger.error(f"Quantum burst failed: {e}")
            # Fallback basic random perturbation
            noise = self.xp.random.normal(0, 0.1, self.pheromones.pheromones.shape)
            self.pheromones.pheromones += (noise + 1j*noise).astype(self.xp.complex64)
            self._symmetrize_clamp()

    ###########################################################################
    #                            TOPOLOGICAL FIELD                            #
    ###########################################################################

    def persistent_homology_update(self):
        """
        Update anxiety and momentum based on topological analysis.
        
        This function uses persistent homology to extract topological features
        from the pheromone field, which are then used to update the anxiety
        wavefunction and covariant momentum.
        """
        try:
            if not TOPOLOGY_AVAILABLE:
                self._fallback_topology_update()
                return
                
            # Convert to numpy for topology analysis
            real_data = self.pheromones.pheromones.real
            if self.config.use_gpu:
                real_data = cp.asnumpy(real_data)
                
            # Compute persistence diagrams
            diagrams = self.rips.fit_transform(real_data)
            self.homology_report = diagrams
            
            # Transform diagrams to get persistence images
            self.pimgr.fit(diagrams)
            pim = self.pimgr.transform(diagrams)
            
            # Update anxiety wavefunction
            if pim.ndim >= 2:
                rep_val = float(pim.mean())
                shape_2d = (self.config.n_dim, self.config.n_dim)
                repeated = self.xp.zeros(shape_2d, dtype=self.xp.complex64)
                repeated[:] = rep_val
                self.anxiety_wavefunction = repeated
            else:
                self.anxiety_wavefunction = self.xp.zeros_like(self.anxiety_wavefunction)

            # Calculate persistence lifetimes
            lifetimes = []
            for d in diagrams:
                if len(d) > 0:
                    pers = d[:, 1] - d[:, 0]
                    lifetimes.append(pers.mean())
                    
            # Update covariant momentum
            if lifetimes:
                mean_pers = float(np.mean(lifetimes))
                delta = self.xp.array(mean_pers, dtype=self.xp.complex64)
                self.covariant_momentum = 0.9 * self.covariant_momentum + 0.1 * delta * 1j
            else:
                # Fallback momentum update
                self.covariant_momentum += 0.001j * self.xp.random.normal(
                    size=self.covariant_momentum.shape)
                
        except Exception as e:
            logger.error(f"Persistent homology update failed: {e}")
            self._fallback_topology_update()

    def _fallback_topology_update(self):
        """
        Provide a fallback update mechanism when topological analysis fails.
        
        This simplified update uses statistics of the pheromone field rather
        than topological features.
        """
        logger.debug("Using fallback topology update")
        
        # Extract real part statistics
        real_part = self.pheromones.pheromones.real
        mean_val = float(self.xp.mean(real_part))
        std_val = float(self.xp.std(real_part))
        
        # Update anxiety based on statistics
        anxiety_val = self.xp.array(mean_val + 0.1j * std_val, dtype=self.xp.complex64)
        self.anxiety_wavefunction = self.xp.ones_like(self.anxiety_wavefunction) * anxiety_val
        
        # Simple momentum update
        momentum_update = 0.001j * self.xp.random.normal(size=self.covariant_momentum.shape)
        self.covariant_momentum = 0.9 * self.covariant_momentum + 0.1 * momentum_update

    ###########################################################################
    #                       CRISIS-DRIVEN HYPERPARAMS                         #
    ###########################################################################

    def _update_hyperparams(self):
        """
        Update hyperparameters based on the current system state.
        
        This function implements the crisis-driven hyperparameter adjustment,
        where parameters like alpha, beta, and rho respond dynamically to
        changes in panic levels, anxiety, and topological features.
        """
        if not self.config.adaptive_hyperparams:
            return
            
        try:
            # Calculate panic and anxiety metrics
            p_mean = float(self.xp.mean(self.panic_tensor))
            a_amp = float(self.xp.mean(self.xp.abs(self.anxiety_wavefunction)))

            # Update alpha based on panic and anxiety
            new_alpha_real = p_mean * a_amp
            self.alpha = self.xp.complex64(new_alpha_real + self.alpha.imag * 1j)
            
            # Update rho based on covariant momentum
            mom_norm = float(self.xp.linalg.norm(self.covariant_momentum))
            self.rho = 0.9 * self.rho + 0.1 * float(self.xp.exp(-mom_norm))

            # Update beta based on persistent entropy
            real_field = self.pheromones.pheromones.real
            if self.config.use_gpu:
                real_field = cp.asnumpy(real_field)
                
            if TOPOLOGY_AVAILABLE:
                try:
                    pe = persistent_entropy(real_field)
                    self.beta = pe * 0.1
                except Exception as e:
                    logger.debug(f"Persistent entropy calculation failed: {e}")
                    self.beta *= 0.99
            else:
                # Fallback entropy calculation
                flat_field = real_field.flatten()
                hist, _ = np.histogram(flat_field, bins=20, density=True)
                hist = hist[hist > 0]  # Remove zeros
                entropy = -np.sum(hist * np.log(hist)) if len(hist) > 0 else 0
                self.beta = entropy * 0.1
            
            # Record history
            self.alpha_history.append(float(self.alpha.real))
            self.beta_history.append(float(self.beta))
            self.rho_history.append(float(self.rho))
            
        except Exception as e:
            logger.error(f"Hyperparameter update failed: {e}")
            # Keep previous values

    def _symmetrize_clamp(self):
        """
        Ensure the real part of the pheromone field is symmetric and non-negative.
        
        This enforces constraints on the pheromone field to maintain its
        physical interpretation as distances or similarities.
        """
        r = self.pheromones.pheromones.real
        # Make symmetric (average with transpose)
        r = 0.5 * (r + r.T)
        # Remove diagonal
        r -= self.xp.diag(self.xp.diag(r))
        # Ensure non-negative
        r = self.xp.maximum(r, 0)
        # Update while preserving imaginary part
        self.pheromones.pheromones = r + 1j * self.pheromones.pheromones.imag

    def _trigger_stagnation_reset(self):
        """
        Reset parts of the system when optimization stagnates.
        
        This helps escape plateaus in the optimization landscape by
        performing a partial reset of the pheromone field.
        """
        logger.info(f"Stagnation detected after {self.stagnation_counter} iterations. Performing partial reset.")
        self.pheromones.partial_reset()
        self.stagnation_counter = 0
        
        # Also redistribute some tokens to prevent monopolies
        self.economy.redistribute_tokens(0.1)

    ###########################################################################
    #                              MAIN OPTIMIZE                              #
    ###########################################################################

    def optimize(self,
                 agents: List[UniversalNode],
                 loss_fn: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, np.ndarray, List[float], Any]:
        """
        Execute the main optimization loop.
        
        This function coordinates the interaction between all UMACO components:
        universal economy, cognitive agents, PAQ Core, and topological field.
        
        Args:
            agents (List[UniversalNode]): List of cognitive agents.
            loss_fn (Callable[[np.ndarray], float]): Loss function to minimize.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[float], Any]: Tuple containing:
                - Real part of final pheromone field
                - Imaginary part of final pheromone field
                - History of panic levels
                - Homology report (topological features)
        """
        logger.info(f"Starting optimization with {len(agents)} agents for {self.max_iter} iterations")
        
        for i in range(self.max_iter):
            # Extract real part of pheromones for loss calculation
            real_part = self.pheromones.pheromones.real
            if self.config.use_gpu:
                real_part = cp.asnumpy(real_part)
                
            # Calculate loss and update histories
            loss_val = loss_fn(real_part)
            self.loss_history.append(loss_val)
            
            # Approximate gradient
            grad_approx = self.xp.ones_like(self.pheromones.pheromones.real) * float(loss_val) * 0.01
            
            # Update panic based on gradient
            self.panic_backpropagate(grad_approx)
            self.panic_history.append(float(self.xp.mean(self.panic_tensor)))

            # Trigger quantum burst based on panic or anxiety levels
            if (float(self.xp.mean(self.panic_tensor)) > 0.7 or 
                    self.xp.linalg.norm(self.anxiety_wavefunction) > 1.7):
                self.quantum_burst()

            # Update topological features
            self.persistent_homology_update()
            
            # Update hyperparameters based on system state
            self._update_hyperparams()

            # Apply covariant momentum update to pheromones
            self.pheromones.pheromones += self.alpha.real * self.covariant_momentum
            self._symmetrize_clamp()

            # Check and adjust entropy if needed
            try:
                if TOPOLOGY_AVAILABLE:
                    real_field = self.pheromones.pheromones.real
                    if self.config.use_gpu:
                        real_field = cp.asnumpy(real_field)
                    ent = persistent_entropy(real_field)
                    
                    if abs(ent - self.target_entropy) > 0.1:
                        noise = self.xp.random.normal(0, 0.01, self.pheromones.pheromones.shape)
                        self.pheromones.pheromones += 0.01j * noise
            except Exception as e:
                logger.debug(f"Entropy adjustment failed: {e}")

            # Track best solution
            current_score = 1.0 / (1.0 + loss_val)
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_solution = real_part.copy()
                self.stagnation_counter = 0
                logger.debug(f"New best solution at iteration {i}: score={current_score:.5f}, loss={loss_val:.5f}")
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.config.partial_reset_threshold:
                    self._trigger_stagnation_reset()

            # Handle scheduled quantum bursts
            self.burst_countdown -= 1
            if self.burst_countdown <= 0:
                self.quantum_burst()
                self.burst_countdown = self.config.quantum_burst_interval

            # Update market conditions
            self.economy.update_market_dynamics()
            
            # Calculate resource scarcity
            scarcity = 0.7 * float(self.xp.mean(self.pheromones.pheromones.real))
            if self.config.use_gpu:
                scarcity = scarcity.get()
            scarcity += 0.3 * (1 - self.economy.market_value)
            
            # Let agents propose actions
            for agent in agents:
                agent.propose_action(loss_val, float(scarcity))
                
            # Log progress every 10% of iterations
            if i % max(1, self.max_iter // 10) == 0 or i == self.max_iter - 1:
                logger.info(f"Iteration {i}/{self.max_iter}: loss={loss_val:.5f}, "
                          f"panic={float(self.xp.mean(self.panic_tensor)):.3f}, "
                          f"best_score={self.best_score:.5f}")

        # Prepare return values, converting to numpy if needed
        pheromone_real = self.pheromones.pheromones.real
        pheromone_imag = self.pheromones.pheromones.imag
        
        if self.config.use_gpu:
            pheromone_real = cp.asnumpy(pheromone_real)
            pheromone_imag = cp.asnumpy(pheromone_imag)
            
        return (
            pheromone_real,
            pheromone_imag,
            self.panic_history,
            self.homology_report
        )
    
    def get_best_solution(self) -> np.ndarray:
        """
        Get the best solution found during optimization.
        
        Returns:
            np.ndarray: The best solution or None if optimization hasn't been run.
        """
        return self.best_solution
    
    def get_optimization_history(self) -> Dict[str, List[float]]:
        """
        Get the history of key metrics during optimization.
        
        Returns:
            Dict[str, List[float]]: Dictionary with histories of various metrics.
        """
        return {
            'loss': self.loss_history,
            'panic': self.panic_history,
            'alpha': self.alpha_history,
            'beta': self.beta_history,
            'rho': self.rho_history,
            'quantum_bursts': self.quantum_burst_history
        }


###############################################################################
#                                  UTILITIES                                  #
###############################################################################

def create_default_umaco(dim: int = 32, max_iter: int = 1000, use_gpu: bool = True) -> UMACO10:
    """
    Create a UMACO instance with default configuration.
    
    A convenience function for quickly setting up a UMACO optimizer
    with reasonable default parameters.
    
    Args:
        dim (int): Dimensionality of the optimization space.
        max_iter (int): Maximum number of optimization iterations.
        use_gpu (bool): Whether to use GPU acceleration.
        
    Returns:
        UMACO10: Configured UMACO optimizer.
    """
    # Create initial panic seed with small random values
    panic_seed = np.random.rand(dim, dim).astype(np.float32) * 0.2
    
    # Create standard configuration
    config = UMACO10Config(
        n_dim=dim,
        panic_seed=panic_seed,
        trauma_factor=0.1,
        alpha=0.25,
        beta=0.1,
        rho=0.25,
        max_iter=max_iter,
        use_gpu=use_gpu
    )
    
    # Create economy
    economy = UniversalEconomy(EconomyConfig(n_agents=min(8, dim)))
    
    # Create pheromone system
    pheromones = NeuroPheromoneSystem(PheromoneConfig(n_dim=dim))
    
    # Create and return UMACO instance
    return UMACO10(config, economy, pheromones)


def create_agents(economy: UniversalEconomy, n_agents: int) -> List[UniversalNode]:
    """
    Create a set of cognitive agents with varying configurations.
    
    A helper function to quickly set up a diverse set of agents for the UMACO system.
    
    Args:
        economy (UniversalEconomy): The shared economy for the agents.
        n_agents (int): Number of agents to create.
        
    Returns:
        List[UniversalNode]: List of configured agents.
    """
    agents = []
    
    # Possible specializations for diversity
    specializations = [
        "exploration", "exploitation", "quantum", "topological", 
        "memory", "crisis", "coordination", None
    ]
    
    for i in range(n_agents):
        # Create varied configurations
        config = NodeConfig(
            panic_level_init=0.1 + 0.2 * np.random.random(),
            risk_appetite=0.3 + 0.6 * np.random.random(),
            specialization=specializations[i % len(specializations)]
        )
        
        # Create and add agent
        agent = UniversalNode(i, economy, config)
        agents.append(agent)
        
    return agents


def simple_loss_function(x: np.ndarray) -> float:
    """
    A simple test loss function for UMACO optimization.
    
    Args:
        x (np.ndarray): Input array to evaluate.
        
    Returns:
        float: Loss value (lower is better).
    """
    # Rastrigin function (challenging non-convex function with many local minima)
    n = x.size
    return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


###############################################################################
#                               USAGE EXAMPLE                                 #
###############################################################################

def example_usage():
    """Demonstrates basic usage of the UMACO framework."""
    # Create optimizer with default settings
    umaco = create_default_umaco(dim=32, max_iter=500)
    
    # Create agents
    agents = create_agents(umaco.economy, n_agents=8)
    
    # Define a loss function (here using the simple test function)
    def my_loss(x):
        return simple_loss_function(x)
    
    # Run optimization
    result_real, result_imag, panic_history, homology = umaco.optimize(agents, my_loss)
    
    # Get best solution
    best_solution = umaco.get_best_solution()
    best_score = umaco.best_score
    
    print(f"Optimization complete. Best score: {best_score:.5f}")
    return best_solution


if __name__ == "__main__":
    example_usage()
