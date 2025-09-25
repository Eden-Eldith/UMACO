#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic UMACO9 Optimization Example
=================================

This example demonstrates how to use UMACO9 to optimize a simple 2D function.
It shows the basic setup and configuration of the optimization process.
"""

import numpy as np
import matplotlib.pyplot as plt
from Umaco9 import (
    UMACO9, UMACO9Config,
    UniversalEconomy, EconomyConfig,
    UniversalNode, NodeConfig,
    NeuroPheromoneSystem, PheromoneConfig
)

# Define a 2D test function (Rosenbrock function)
def rosenbrock(x):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    if x.ndim == 2:
        # Extract x and y from the pheromone matrix (use diagonal elements)
        n = min(x.shape[0], x.shape[1])
        a = np.diag(x)[:n]
        if len(a) < 2:
            return np.sum(a**2)
        x, y = a[0], a[1]
    else:
        x, y = x[0], x[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def main():
    # Configure economy
    economy_config = EconomyConfig(
        n_agents=8,
        initial_tokens=250,
        token_reward_factor=3.0
    )
    economy = UniversalEconomy(economy_config)
    
    # Configure pheromone system
    n_dim = 64
    pheromone_config = PheromoneConfig(
        n_dim=n_dim,
        initial_val=0.3,
        evaporation_rate=0.1
    )
    pheromones = NeuroPheromoneSystem(pheromone_config)
    
    # Configure UMACO9
    initial_values = np.zeros((n_dim, n_dim))
    initial_values[0, 0] = -0.5  # Initialize x
    initial_values[1, 1] = -0.5  # Initialize y
    
    config = UMACO9Config(
        n_dim=n_dim,
        panic_seed=initial_values + 0.1,
        trauma_factor=0.5,
        alpha=0.2,
        beta=0.1,
        rho=0.3,
        max_iter=10,  # Reduced for testing
        quantum_burst_interval=50
    )
    
    # Create UMACO9 solver
    solver = UMACO9(config, economy, pheromones)
    
    # Create agents
    agents = [UniversalNode(i, economy, NodeConfig()) for i in range(8)]
    
    # Run optimization
    print("Starting optimization...")
    pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
        agents, rosenbrock
    )
    
    # Extract results (x, y values from the diagonal)
    final_x = pheromone_real[0, 0]
    final_y = pheromone_real[1, 1]
    final_value = rosenbrock(np.array([final_x, final_y]))
    
    print(f"Optimization completed.")
    print(f"Final solution: x = {final_x:.6f}, y = {final_y:.6f}")
    print(f"Final function value: {final_value:.6f}")
    print(f"Global minimum (expected): x = 1.0, y = 1.0, f(x,y) = 0.0")
    
    # Plot optimization progress
    plt.figure(figsize=(12, 8))
    
    # Plot panic history
    plt.subplot(2, 2, 1)
    plt.plot(panic_history)
    plt.title('Panic Level During Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Average Panic')
    plt.grid(True)
    
    # Plot pheromone matrix heatmap
    plt.subplot(2, 2, 2)
    plt.imshow(pheromone_real, cmap='viridis')
    plt.colorbar()
    plt.title('Final Pheromone Matrix (Real Part)')
    
    # Plot token distribution
    plt.subplot(2, 2, 3)
    token_distribution = list(economy.tokens.values())
    plt.bar(range(len(token_distribution)), token_distribution)
    plt.title('Final Token Distribution')
    plt.xlabel('Agent ID')
    plt.ylabel('Tokens')
    
    # Create contour plot of the Rosenbrock function
    plt.subplot(2, 2, 4)
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = rosenbrock(np.array([X[i, j], Y[i, j]]))
    
    # Use log scale for better visualization
    plt.contourf(X, Y, np.log(Z + 1), levels=50, cmap='viridis')
    plt.colorbar()
    plt.plot(final_x, final_y, 'ro', markersize=10, label='UMACO Solution')
    plt.plot(1.0, 1.0, 'g*', markersize=10, label='Global Minimum')
    plt.title('Rosenbrock Function (log scale)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('umaco_optimization_result.png')
    plt.show()

if __name__ == "__main__":
    main()
