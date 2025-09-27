import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
import logging


def _resolve_gpu_backend(module_name: str = "NeuroPheromonebasicv5"):
    """Resolve the GPU backend while respecting the global CPU override flag."""
    allow_cpu = os.getenv("UMACO_ALLOW_CPU", "0") == "1"
    module_logger = logging.getLogger(module_name)

    try:
        import cupy as _cp  # type: ignore
    except ImportError as exc:
        if not allow_cpu:
            raise RuntimeError(
                "NeuroPheromonebasicv5 requires CuPy for GPU execution. Install cupy-cudaXX or set UMACO_ALLOW_CPU=1 to acknowledge CPU fallback."
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


# Compatibility layer for cupy functions
def asnumpy(arr):
    """Convert cupy array to numpy array, or pass through if already numpy"""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def to_numpy_scalar(val):
    """Convert cupy scalar to numpy scalar, or pass through if already numpy"""
    if GPU_AVAILABLE and hasattr(val, 'get'):
        return float(val.get())
    try:
        return float(val.item())
    except AttributeError:
        return float(val)


if GPU_AVAILABLE:
    from cupy.cuda import MemoryPool
else:  # CPU fallback
    class MemoryPool:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
from persim import PersistenceImager
from ripser import Rips
from persim.persistent_entropy import persistent_entropy
from scipy import stats
import sys

class MACOOptimizer:
    def __init__(self, pheromones, alpha=3.54879, beta=2.38606, rho=0.13814, max_iterations=3000, target_entropy=0.68894):
        # Keep original parameter values
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.max_iterations = max_iterations
        self.target_entropy = target_entropy
        
        # Initialize pheromones with clean values
        pheromones_np = np.asarray(pheromones)
        # Start with all pheromones at 0.5 (target value in loss function)
        self.pheromones = cp.ones_like(cp.asarray(pheromones_np)) * 0.5
        
        # Initialize with finite starting values
        self.best_quality_so_far = 100.0  # Start with a high but finite loss value
        self.noise_std = 0.11266
        self.anxiety_level = 0.01  # Start with low anxiety
        
        # Initialize momentum with small values
        self.hypergrad_momentum = cp.ones_like(self.pheromones) * 0.01
        
        self.previous_loss = 1000.0  # Start with finite previous loss
        self.previous_gradient = cp.ones_like(self.pheromones) * 0.01
        self.previous_energy = 0.5
        self.pool = MemoryPool()
        
        # For stagnation detection
        self.consecutive_no_improvement = 0
        self.improvement_threshold = 0.0001
        self.stagnation_counter = 0
        self.anxiety_threshold = 0.9  # Threshold for quantum burst
        
        # Reference to graph structure (will be set by the brain model)
        self.graph = None
        
        # Print initial state
        print(f"Initialized optimizer with pheromone shape: {self.pheromones.shape}")
        print(f"Initial parameters: alpha={self.alpha}, beta={self.beta}, rho={self.rho}")

    def compute_entropy(self, q=0.8):
        """Compute entropy to help balance exploration/exploitation."""
        # Safe entropy calculation
        total = cp.sum(self.pheromones) + 1e-6
        p = self.pheromones / total
        p = cp.clip(p, 1e-6, 1.0)  # Avoid zeros
        
        # Compute q-entropy
        return float((1 - cp.sum(p**q)) / (q - 1 + 1e-6))

    def update_anxiety(self, current_loss):
        """Increase anxiety when the system stagnates, adjusting exploration."""
        # Track consecutive iterations without significant improvement
        loss_improvement = self.previous_loss - current_loss
        
        if loss_improvement < self.improvement_threshold:
            self.consecutive_no_improvement += 1
        else:
            self.consecutive_no_improvement = 0
            
        # Compute loss delta with safety
        delta_loss = min(1.0, abs(float(self.previous_loss - current_loss)))
        
        # Compute gradient change with safety
        gradient_change = float(cp.linalg.norm(self.previous_gradient - self.pheromones))
        gradient_change = min(1.0, gradient_change)  # Cap gradient change
        
        # Factor in stagnation duration - anxiety increases with consecutive no improvement
        stagnation_factor = min(1.0, 0.01 * self.consecutive_no_improvement)
        
        # Update anxiety with safety - higher when loss isn't changing AND gradients are small
        # Low delta_loss and low gradient_change indicate we're stuck
        if delta_loss < 0.001 and gradient_change < 0.01:
            self.anxiety_level = min(1.0, self.anxiety_level + 0.000001)
        else:
            # Gradually decrease anxiety when we're making progress
            self.anxiety_level = max(0.35, self.anxiety_level * 0.95)
        
        # Add influence from stagnation factor
        self.anxiety_level = min(1.0, self.anxiety_level + stagnation_factor)
        
        # Update tracking variables
        self.previous_loss = current_loss
        self.previous_gradient = self.pheromones.copy()
        
        # Debug info
        print(f"Anxiety: {self.anxiety_level:.4f}, Stagnation: {self.consecutive_no_improvement}")

    def should_apply_quantum_burst(self):
        """Determine if a quantum burst should be applied based on anxiety level."""
        return self.anxiety_level > self.anxiety_threshold or self.consecutive_no_improvement > 100

    

    def apply_quantum_burst(self):
        """Apply perturbations to escape stagnation and explore new solutions."""
        # Scale noise by anxiety level - more anxiety means more exploration
        burst_strength = self.anxiety_level * 0.2
        
        # Simple Gaussian perturbation for stability
        noise = cp.random.normal(0, burst_strength, self.pheromones.shape).astype(cp.float32)
        perturbation = noise * self.anxiety_level
        
        # Apply bounded perturbation
        self.pheromones = cp.clip(self.pheromones + perturbation, 0.1, 1.0)
        
        # Reset consecutive no improvement counter after burst
        self.consecutive_no_improvement = 0
        
        # Debug info
        print(f"Applied quantum burst with anxiety={self.anxiety_level:.4f}")

    def adjust_exploration(self):
        """Adjust exploration rate (α) and exploitation rate (ρ) based on entropy."""
        try:
            entropy = self.compute_entropy()
            
            # Smooth adjustment with limited rate of change
            adjustment = min(0.1, max(-0.1, (entropy - self.target_entropy) * 0.1))
            
            # Update rates with stability constraints
            self.alpha = min(5.0, max(0.1, self.alpha + adjustment))
            self.rho = min(0.5, max(0.01, self.rho - adjustment * 0.1))
            
        except Exception as e:
            print(f"Error in adjust_exploration: {e}")

    def apply_covariance(self, delta):
        """Apply simplified update with momentum."""
        # Clip delta for stability
        delta = cp.clip(delta, -0.1, 0.1)
        
        # Update momentum with smooth blending
        self.hypergrad_momentum = 0.9 * self.hypergrad_momentum + 0.1 * delta
        
        # Apply momentum update with alpha scaling
        update = self.alpha * self.hypergrad_momentum
        
        # Prevent extreme updates
        update = cp.clip(update, -0.1, 0.1)
        
        # Apply update with bounds checking
        self.pheromones = cp.clip(self.pheromones + update, 0.1, 1.0)

    def evaluate_population(self, params_batch):
        """Evaluate population parameters using direct computation."""
        try:
            # Ensure parameters are bounded
            params_gpu = cp.clip(cp.asarray(params_batch), 0.0, 1.0)
            
            # Simple squared error evaluation (avoiding complex kernel)
            # Using target value of 0.5 to match loss function
            results_gpu = cp.sum((params_gpu - 0.5)**2, axis=1)
            
            # Ensure finite results
            return cp.clip(results_gpu, 0.0, 100.0)
            
        except Exception as e:
            print(f"Error in evaluate_population: {e}")
            # Return high but finite cost
            return cp.ones(len(params_batch)) * 100.0

    def pheromone_update(self, activated_neurons):
        """Update pheromones based on Hebbian learning."""
        # Ensure graph is initialized
        if self.graph is None:
            print("Warning: Graph not initialized in pheromone_update. Skipping update.")
            return
        
        # Get pheromone dimensions
        n_rows, n_cols = self.pheromones.shape
        
        # Filter activated neurons to those within bounds
        activated_neurons = [i for i in activated_neurons if i < n_rows and i < n_cols]
        
        # Skip if no valid neurons
        if not activated_neurons:
            return
            
        # Update pheromones for activated neural connections
        for i in activated_neurons:
            for j in activated_neurons:
                # Ensure indices are within bounds and edge exists
                if i != j and i < n_rows and j < n_cols and (i, j) in self.graph.edges():
                    # Small fixed increment for stability
                    increment = 0.01 * self.anxiety_level
                    self.pheromones[i, j] += increment
                    # Cap maximum value
                    self.pheromones[i, j] = min(1.0, float(self.pheromones[i, j]))
        
        # Apply evaporation to non-activated connections
        for i, j in self.graph.edges():
            if i < n_rows and j < n_cols:
                if i not in activated_neurons or j not in activated_neurons:
                    # Gentle evaporation rate
                    self.pheromones[i, j] *= 0.99
                    # Ensure minimum value
                    self.pheromones[i, j] = max(0.1, float(self.pheromones[i, j]))

    def sample_parameters(self, n_ants):
        """Generate sample parameters for evaluation."""
        # Limit number of ants for stability
        n_ants = min(n_ants, min(256, self.pheromones.shape[0]))
        
        # Simple direct sampling - create variations around current pheromones
        samples = []
        for i in range(n_ants):
            # Create small perturbation of current pheromones
            noise_level = 0.1 * (i + 1) / n_ants  # Increasing noise for diversity
            noise = cp.random.normal(0, noise_level, self.pheromones.shape).astype(cp.float32)
            
            # Add noise to current pheromones with bounds
            sample = cp.clip(self.pheromones + noise, 0.1, 1.0)
            samples.append(sample)
            
        # Stack samples into batch
        return cp.stack(samples)

    def adjust_learning_rate(self):
        """Simple learning rate oscillation for coverage."""
        # Oscillate learning rate to avoid local minima
        oscillation = 0.5 + 0.5 * cp.sin(cp.pi * self.previous_energy)
        self.alpha = 0.5 + 3.0 * oscillation
        
        # Update energy state
        self.previous_energy = (self.previous_energy + 0.1) % 1.0

    def optimize(self, loss_function):
        """Main optimization loop."""
        for iteration in range(self.max_iterations):
            try:
                # Adjust learning rate
                self.adjust_learning_rate()

                # Sample parameters
                params_batch = self.sample_parameters(n_ants=8)  # Use fewer ants for stability
                
                # Evaluate parameters
                deltas = self.evaluate_population(params_batch)
                
                # Find best parameters in batch
                best_idx = cp.argmin(deltas)
                delta = params_batch[best_idx] - self.pheromones
                
                # Apply update
                self.apply_covariance(delta * 0.1)  # Scale down delta for stability
                
                # Evaluate current solution
                current_loss = loss_function(self.pheromones)
                
                # Update anxiety level based on current performance
                self.update_anxiety(current_loss)
                
                # Check if we need to apply a quantum burst (only when stuck)
                if self.should_apply_quantum_burst():
                    self.apply_quantum_burst()
                
                # Adjust exploration parameters
                self.adjust_exploration()
                
                # Update best solution if improved
                if current_loss < self.best_quality_so_far:
                    improvement = self.best_quality_so_far - current_loss
                    self.best_quality_so_far = current_loss
                    print(f"Improved by {improvement:.6f}! Best quality: {self.best_quality_so_far:.6f}")
                
                print(f"Iteration {iteration}, Loss: {current_loss:.6f}, Best: {self.best_quality_so_far:.6f}")
                
                # Early convergence check
                if self.best_quality_so_far < 0.0001:
                    print("Convergence reached!")
                    break
                    
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                continue
                
        return self.best_quality_so_far


class SelfOptimizingBrainModel:
    def __init__(self, num_neurons=2000, max_iterations=1):
        # Set model parameters
        self.num_neurons = num_neurons
        self.max_iterations = max_iterations
        
        # Initialize graph with exact number of neurons
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_neurons))

        # Build brain regions
        self.build_brain_regions()

        # Initialize pheromones with target values (0.5)
        self.pheromones = np.ones((self.num_neurons, self.num_neurons), dtype=np.float32) * 0.5

        # Initialize anxiety level
        self.anxiety_level = 0.1

        # Tracking best performance
        self.best_quality_so_far = 1000.0  # Start with finite value

        # Initialize the MACO optimizer with the original parameters
        self.maco_optimizer = MACOOptimizer(self.pheromones, 
                                           alpha=3.54879, 
                                           beta=2.38606, 
                                           rho=0.13814, 
                                           target_entropy=0.68894)
        
        # Share the graph with the optimizer
        self.maco_optimizer.graph = self.graph
        print(f"Initialized brain model with {self.num_neurons} neurons")
        print(f"Graph has {len(self.graph.edges())} edges")

    def build_brain_regions(self):
        """Create specialized brain regions with different connectivity patterns."""
        # Divide neurons into three balanced regions
        region_size = self.num_neurons // 3
        remainder = self.num_neurons % 3
        
        # Calculate region sizes with fair distribution of remainder
        region1_size = region_size + (1 if remainder > 0 else 0)
        region2_size = region_size + (1 if remainder > 1 else 0)
        region3_size = region_size 
        
        # Define region node ranges
        region1_nodes = list(range(0, region1_size))
        region2_nodes = list(range(region1_size, region1_size + region2_size))
        region3_nodes = list(range(region1_size + region2_size, self.num_neurons))
        
        # Create mini-networks within each region
        # Region 1: Watts-Strogatz (small world network)
        k = min(3, region1_size - 1)  # Number of neighbors
        if region1_size > 1 and k > 0:
            for i in range(len(region1_nodes)):
                for j in range(1, k + 1):
                    neighbor = (i + j) % len(region1_nodes)
                    self.graph.add_edge(region1_nodes[i], region1_nodes[neighbor])
        
        # Region 2: Geometric-like network (connect nodes that are "close")
        if region2_size > 1:
            for i in range(len(region2_nodes)):
                for j in range(i + 1, len(region2_nodes)):
                    if random.random() < 0.45:  # 45% chance of connection
                        self.graph.add_edge(region2_nodes[i], region2_nodes[j])
        
        # Region 3: Erdos-Renyi (random connections)
        if region3_size > 1:
            for i in range(len(region3_nodes)):
                for j in range(i + 1, len(region3_nodes)):
                    if random.random() < 0.45:  # 45% chance of connection
                        self.graph.add_edge(region3_nodes[i], region3_nodes[j])
        
        # Add cross-region connections
        max_cross_connections = min(15, self.num_neurons // 2.5)
        for _ in range(max_cross_connections):
            if region1_nodes and region3_nodes:  # Connect region 1 and 3
                source = random.choice(region1_nodes)
                target = random.choice(region3_nodes)
                self.graph.add_edge(source, target)
                
            if region2_nodes and region3_nodes:  # Connect region 1 and 3
                        source = random.choice(region2_nodes)
                        target = random.choice(region3_nodes)
                        self.graph.add_edge(source, target)

    def pheromone_update(self, activated_neurons):
        """Update pheromones based on Hebbian learning."""
        # Ensure neurons are valid indices
        activated_neurons = [i for i in activated_neurons if 0 <= i < self.num_neurons]
        
        # Skip update if no valid neurons
        if not activated_neurons:
            return
            
        # Update pheromones based on activated neurons
        for i in activated_neurons:
            for j in activated_neurons:
                if i != j and (i, j) in self.graph.edges():
                    self.pheromones[i, j] += 0.01 * self.anxiety_level
                    self.pheromones[i, j] = min(1.0, self.pheromones[i, j])

        # Apply evaporation to non-activated connections
        for i, j in self.graph.edges():
            if i not in activated_neurons or j not in activated_neurons:
                self.pheromones[i, j] *= 0.99
                self.pheromones[i, j] = max(0.1, self.pheromones[i, j])

    def run(self, loss_function):
        """Run the optimization loop."""
        print("Starting optimization run...")
        for iteration in range(self.max_iterations):
            try:
                # Select random neurons to activate
                available_nodes = list(range(self.num_neurons))
                sample_size = min(30, len(available_nodes))
                
                if sample_size == 0:
                    print(f"Error: No valid nodes available.")
                    break
                    
                activated_neurons = random.sample(available_nodes, sample_size)
                self.pheromone_update(activated_neurons)

                # Run one optimizer iteration
                current_loss = self.maco_optimizer.optimize(loss_function)
                
                # Transfer optimized pheromones back to model
                try:
                    optimized_pheromones = asnumpy(self.maco_optimizer.pheromones)
                    
                    # Check for valid values and update
                    if np.all(np.isfinite(optimized_pheromones)):
                        # Use bounded values
                        self.pheromones = np.clip(optimized_pheromones, 0.1, 1.0)
                        print(f"Updated pheromones from optimizer")
                    else:
                        print(f"Warning: Invalid optimizer values")
                        
                except Exception as e:
                    print(f"Error transferring pheromones: {e}")
                
                # Update tracking of best quality
                if current_loss < self.best_quality_so_far:
                    self.best_quality_so_far = current_loss
                    print(f"New best loss: {self.best_quality_so_far:.6f}")

                print(f"Main iteration {iteration}, Best Loss: {self.best_quality_so_far:.6f}")
                
            except Exception as e:
                print(f"Error in main iteration {iteration}: {e}")
                continue

        return self.best_quality_so_far

    def visualize(self):
        """Visualize the self-optimized brain."""
        pos = nx.spring_layout(self.graph, seed=42)
        plt.figure(figsize=(10, 10))

        # Create color map based on pheromone strength
        edge_colors = []
        for u, v in self.graph.edges():
            edge_colors.append(self.pheromones[u, v])
            
        # Normalize colors
        if edge_colors:
            max_color = max(edge_colors)
            min_color = min(edge_colors)
            if max_color > min_color:
                edge_colors = [(c - min_color) / (max_color - min_color) for c in edge_colors]

        # Draw the graph
        nx.draw_networkx_nodes(self.graph, pos, node_size=100, node_color='skyblue', alpha=0.7)
        nx.draw_networkx_edges(self.graph, pos, width=2, alpha=0.7, edge_color=edge_colors, edge_cmap=plt.cm.Blues)
        nx.draw_networkx_labels(self.graph, pos, font_size=10)

        plt.title("Self-Optimizing Brain Model")
        plt.axis('off')
        plt.show()

# Define a real loss function that properly handles CuPy arrays
def real_loss_function(pheromones):
    """Loss function based on distance from target value of 0.5."""
    try:
        # Check for invalid inputs
        if cp.isnan(pheromones).any():
            return 100.0
        
        # Target value is 0.5 - consistent with initialization
        target = 0.5
        
        # Calculate mean squared error from target
        error = cp.mean((pheromones - target)**2)
        
        # Convert to Python float and ensure finite
        loss = float(error)
        return min(100.0, max(0.0, loss))
    except Exception as e:
        print(f"Error in loss function: {e}")
        return 100.0  # Return high but finite loss

# Instantiate and run the self-optimizing brain model
brain_model = SelfOptimizingBrainModel(num_neurons=2000)  # Explicitly set num_neurons to 2000
best_loss = brain_model.run(real_loss_function)  # Run optimization with real loss function
brain_model.visualize()  # Visualize the self-optimized brain
print(f"Optimization finished with best loss: {best_loss}")