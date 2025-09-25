#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic UMACO13 Optimization Example
==================================

This example demonstrates how to use UMACO13 to optimize a simple 2D function.
It shows the basic setup and configuration of the optimization process.
"""

import numpy as np
import matplotlib.pyplot as plt
from Umaco13 import create_umaco_solver, rosenbrock_loss

def main():
    print("UMACO13 Basic Optimization Demo")
    print("=" * 40)
    
    # Create UMACO solver using factory function
    optimizer, agents = create_umaco_solver(
        problem_type='CONTINUOUS',
        dim=64,  # Higher resolution for continuous optimization
        max_iter=50,
        problem_dim=2
    )
    
    # Run optimization
    print("Starting optimization...")
    pheromone_real, pheromone_imag, panic_history, homology_report = optimizer.optimize(
        agents, rosenbrock_loss
    )
    
    # Extract solution from pheromone field for continuous optimization
    if pheromone_real is not None:
        # Use the same marginal distribution approach as the algorithm
        # Compute marginal distributions for x and y
        x_marginal = np.sum(pheromone_real, axis=1)  # Sum over y for each x
        y_marginal = np.sum(pheromone_real, axis=0)  # Sum over x for each y
        
        # Normalize to probabilities
        x_probs = x_marginal / (np.sum(x_marginal) + 1e-9)
        y_probs = y_marginal / (np.sum(y_marginal) + 1e-9)
        
        # Compute expected values (weighted average of indices)
        x_indices = np.arange(len(x_probs))
        y_indices = np.arange(len(y_probs))
        x_expected = np.sum(x_indices * x_probs)
        y_expected = np.sum(y_indices * y_probs)
        
        # Map expected indices back to parameter space [0, 2]
        x_sol = (x_expected / (len(x_probs) - 1)) * 2
        y_sol = (y_expected / (len(y_probs) - 1)) * 2
        solution = np.array([x_sol, y_sol])
        final_loss = rosenbrock_loss(solution)
        
        print(f"Optimization completed.")
        print(f"Final solution: x = {x_sol:.6f}, y = {y_sol:.6f}")
        print(f"Final function value: {final_loss:.6f}")
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
        
        # Plot token distribution (if economy has tokens)
        plt.subplot(2, 2, 3)
        if hasattr(optimizer.economy, 'tokens') and optimizer.economy.tokens:
            token_distribution = list(optimizer.economy.tokens.values())
            plt.bar(range(len(token_distribution)), token_distribution)
            plt.title('Final Token Distribution')
            plt.xlabel('Agent ID')
            plt.ylabel('Tokens')
        else:
            plt.text(0.5, 0.5, 'Token data not available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Token Distribution (N/A)')
        
        # Create contour plot of the Rosenbrock function
        plt.subplot(2, 2, 4)
        x = np.linspace(-2, 2, 100)
        y = np.linspace(-1, 3, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = rosenbrock_loss(np.array([X[i, j], Y[i, j]]))
        
        # Use log scale for better visualization
        plt.contourf(X, Y, np.log(Z + 1), levels=50, cmap='viridis')
        plt.colorbar()
        plt.plot(x_sol, y_sol, 'ro', markersize=10, label='UMACO Solution')
        plt.plot(1.0, 1.0, 'g*', markersize=10, label='Global Minimum')
        plt.title('Rosenbrock Function (log scale)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('umaco_optimization_result.png')
        print("Plot saved to 'umaco_optimization_result.png'")
        # plt.show()  # Commented out for headless environments
    else:
        print("Optimization failed - no results returned")

if __name__ == "__main__":
    main()
