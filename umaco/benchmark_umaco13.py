#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMACO13 Benchmarking Suite
==========================

Comprehensive benchmarking against other optimization algorithms:
- SciPy optimizers (L-BFGS-B, SLSQP, etc.)
- CMA-ES (if available)
- Other ACO implementations (if available)
- Standard test functions: Rosenbrock, Rastrigin, Ackley, Sphere

Usage:
    python benchmark_umaco13.py --problems rosenbrock rastrigin --optimizers umaco scipy --runs 5
"""

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Callable, Any
import json
import os

# Import UMACO
from Umaco13 import create_umaco_solver, rosenbrock_loss

# Try to import other optimizers
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Install with: pip install scipy")

try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("Warning: CMA-ES not available. Install with: pip install cma")


# =================================================================================================
# TEST FUNCTIONS
# =================================================================================================

def sphere_loss(x: np.ndarray) -> float:
    """Sphere function: f(x) = sum(x_i^2), minimum at x=0, f=0"""
    return np.sum(x ** 2)

def rosenbrock_loss_benchmark(x: np.ndarray) -> float:
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1), f=0"""
    if len(x) < 2:
        return float('inf')
    return sum(100.0 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x)-1))

def rastrigin_loss(x: np.ndarray) -> float:
    """Rastrigin function: f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i)), minimum at x=0, f=0"""
    A = 10
    n = len(x)
    return A * n + sum(x_i**2 - A * np.cos(2 * np.pi * x_i) for x_i in x)

def ackley_loss(x: np.ndarray) -> float:
    """Ackley function: complex multimodal function, minimum at x=0, f=0"""
    a, b, c = 20, 0.2, 2 * np.pi
    n = len(x)
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.exp(1)

def griewank_loss(x: np.ndarray) -> float:
    """Griewank function: f(x) = 1 + sum(x_i^2)/4000 - prod(cos(x_i/sqrt(i))), minimum at x=0, f=0"""
    sum_sq = np.sum(x**2) / 4000
    prod_cos = np.prod([np.cos(x_i / np.sqrt(i+1)) for i, x_i in enumerate(x)])
    return 1 + sum_sq - prod_cos


# =================================================================================================
# OPTIMIZER WRAPPERS
# =================================================================================================

class OptimizerResult:
    """Standardized result from optimization run."""
    def __init__(self, solution: np.ndarray, loss: float, time_taken: float,
                 iterations: int = None, converged: bool = None):
        self.solution = solution
        self.loss = loss
        self.time_taken = time_taken
        self.iterations = iterations
        self.converged = converged

    def to_dict(self) -> Dict:
        return {
            'solution': self.solution.tolist(),
            'loss': float(self.loss),
            'time_taken': float(self.time_taken),
            'iterations': self.iterations,
            'converged': self.converged
        }


class UMACO13Optimizer:
    """Wrapper for UMACO13 optimizer."""

    def __init__(self, dim: int = 64, max_iter: int = 200):
        # Use larger pheromone matrix and more iterations for better convergence
        self.dim = dim
        self.max_iter = max_iter

    def optimize(self, loss_func: Callable, bounds: Tuple = None, x0: np.ndarray = None) -> OptimizerResult:
        """Run UMACO13 optimization."""
        start_time = time.time()

        solver, agents = create_umaco_solver(
            problem_type='CONTINUOUS',
            dim=self.dim,
            max_iter=self.max_iter,
            problem_dim=len(x0) if x0 is not None else 2,
            # Tuned hyperparameters for better convergence
            alpha=2.0,      # Lower pheromone influence for exploration
            beta=1.0,       # Lower heuristic influence
            rho=0.05,       # Lower evaporation for stability
            trauma_factor=0.05,  # Lower trauma for less chaos
            n_ants=12       # More agents for better exploration
        )

        pheromone_real, pheromone_imag, panic_history, loss_history, homology_report = solver.optimize(
            agents, loss_func
        )

        end_time = time.time()

        # Extract best solution from pheromone matrix
        # For continuous problems, use marginal distributions
        marginals = np.abs(pheromone_real).sum(axis=0) / np.abs(pheromone_real).sum()
        solution = marginals[:len(x0)] if x0 is not None else marginals[:2]
        final_loss = loss_func(solution)

        return OptimizerResult(
            solution=solution,
            loss=final_loss,
            time_taken=end_time - start_time,
            iterations=self.max_iter,
            converged=panic_history[-1] < 0.3  # Lower panic threshold for convergence
        )


class ScipyOptimizer:
    """Wrapper for SciPy optimizers."""

    def __init__(self, method: str = 'L-BFGS-B', max_iter: int = 1000):
        if not SCIPY_AVAILABLE:
            raise ImportError("SciPy not available")
        self.method = method
        self.max_iter = max_iter

    def optimize(self, loss_func: Callable, bounds: Tuple = None, x0: np.ndarray = None) -> OptimizerResult:
        """Run SciPy optimization."""
        start_time = time.time()

        # Set up bounds if provided
        scipy_bounds = None
        if bounds:
            scipy_bounds = [(bounds[0], bounds[1])] * len(x0)

        result = minimize(
            loss_func,
            x0,
            method=self.method,
            bounds=scipy_bounds,
            options={'maxiter': self.max_iter, 'disp': False}
        )

        end_time = time.time()

        return OptimizerResult(
            solution=result.x,
            loss=result.fun,
            time_taken=end_time - start_time,
            iterations=result.nit if hasattr(result, 'nit') else None,
            converged=result.success
        )


class CMAESOptimizer:
    """Wrapper for CMA-ES optimizer."""

    def __init__(self, sigma: float = 0.5, max_iter: int = 100):
        if not CMA_AVAILABLE:
            raise ImportError("CMA-ES not available")
        self.sigma = sigma
        self.max_iter = max_iter

    def optimize(self, loss_func: Callable, bounds: Tuple = None, x0: np.ndarray = None) -> OptimizerResult:
        """Run CMA-ES optimization."""
        start_time = time.time()

        # CMA-ES works with bounds by transforming variables
        if bounds:
            # Simple bounds handling - could be improved
            x0_scaled = (x0 - bounds[0]) / (bounds[1] - bounds[0])
            def scaled_loss(x):
                x_unscaled = x * (bounds[1] - bounds[0]) + bounds[0]
                return loss_func(x_unscaled)
            es = cma.CMAEvolutionStrategy(x0_scaled, self.sigma)
        else:
            es = cma.CMAEvolutionStrategy(x0, self.sigma)
            scaled_loss = loss_func

        es.optimize(scaled_loss, iterations=self.max_iter)
        end_time = time.time()

        if bounds:
            solution = es.result.xbest * (bounds[1] - bounds[0]) + bounds[0]
        else:
            solution = es.result.xbest

        return OptimizerResult(
            solution=solution,
            loss=es.result.fbest,
            time_taken=end_time - start_time,
            iterations=es.result.iterations,
            converged=es.result.stopevals is not None
        )


# =================================================================================================
# BENCHMARKING FUNCTIONS
# =================================================================================================

def run_single_benchmark(optimizer_name: str, optimizer_class: Any, loss_func: Callable,
                        problem_name: str, dim: int, bounds: Tuple = None,
                        x0: np.ndarray = None) -> Dict:
    """Run a single benchmark test."""
    try:
        optimizer = optimizer_class()
        result = optimizer.optimize(loss_func, bounds, x0)

        return {
            'optimizer': optimizer_name,
            'problem': problem_name,
            'dimension': dim,
            'final_loss': result.loss,
            'time_taken': result.time_taken,
            'iterations': result.iterations,
            'converged': result.converged,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'optimizer': optimizer_name,
            'problem': problem_name,
            'dimension': dim,
            'final_loss': None,
            'time_taken': None,
            'iterations': None,
            'converged': None,
            'success': False,
            'error': str(e)
        }


def run_benchmark_suite(problems: List[str], optimizers: List[str], dimensions: List[int],
                       runs: int = 3, output_dir: str = 'benchmark_results') -> pd.DataFrame:
    """Run complete benchmark suite."""

    # Define test problems
    test_problems = {
        'sphere': (sphere_loss, (-5.12, 5.12)),
        'rosenbrock': (rosenbrock_loss_benchmark, (-2.048, 2.048)),
        'rastrigin': (rastrigin_loss, (-5.12, 5.12)),
        'ackley': (ackley_loss, (-32.768, 32.768)),
        'griewank': (griewank_loss, (-600, 600))
    }

    # Define optimizers
    available_optimizers = {
        'umaco': UMACO13Optimizer,
        'scipy_lbfgs': lambda: ScipyOptimizer('L-BFGS-B'),
        'scipy_slsqp': lambda: ScipyOptimizer('SLSQP'),
        'cmaes': CMAESOptimizer
    }

    # Filter to requested optimizers
    selected_optimizers = {}
    for opt_name in optimizers:
        if opt_name in available_optimizers:
            try:
                # Test if optimizer can be instantiated
                available_optimizers[opt_name]()
                selected_optimizers[opt_name] = available_optimizers[opt_name]
            except ImportError as e:
                print(f"Skipping {opt_name}: {e}")
        else:
            print(f"Unknown optimizer: {opt_name}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    print(f"Running benchmark suite with {len(selected_optimizers)} optimizers, {len(problems)} problems, {len(dimensions)} dimensions, {runs} runs each")

    total_tests = len(selected_optimizers) * len(problems) * len(dimensions) * runs
    test_count = 0

    for problem_name in problems:
        if problem_name not in test_problems:
            print(f"Unknown problem: {problem_name}")
            continue

        loss_func, bounds = test_problems[problem_name]

        for dim in dimensions:
            # Generate initial point
            np.random.seed(42)  # For reproducibility
            x0 = np.random.uniform(bounds[0], bounds[1], dim)

            for optimizer_name, optimizer_class in selected_optimizers.items():
                print(f"Testing {optimizer_name} on {problem_name} (dim={dim})")

                for run in range(runs):
                    test_count += 1
                    print(f"  Run {run + 1}/{runs} ({test_count}/{total_tests})")

                    result = run_single_benchmark(
                        optimizer_name, optimizer_class, loss_func,
                        problem_name, dim, bounds, x0
                    )
                    result['run'] = run
                    all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.csv')
    df.to_csv(csv_file, index=False)

    # Generate summary statistics
    summary = df.groupby(['optimizer', 'problem', 'dimension']).agg({
        'final_loss': ['mean', 'std', 'min'],
        'time_taken': ['mean', 'std'],
        'success': 'mean'
    }).round(4)

    summary_file = os.path.join(output_dir, f'benchmark_summary_{timestamp}.csv')
    summary.to_csv(summary_file)

    # Generate plots
    generate_plots(df, output_dir, timestamp)

    return df


def generate_plots(df: pd.DataFrame, output_dir: str, timestamp: str):
    """Generate comparison plots."""
    try:
        # Filter successful runs only
        success_df = df[df['success'] == True].copy()

        if len(success_df) == 0:
            print("No successful runs to plot")
            return

        # Plot 1: Performance comparison by problem
        problems = success_df['problem'].unique()
        fig, axes = plt.subplots(len(problems), 1, figsize=(12, 4*len(problems)))
        if len(problems) == 1:
            axes = [axes]

        for i, problem in enumerate(problems):
            problem_data = success_df[success_df['problem'] == problem]
            ax = axes[i]

            # Group by optimizer and dimension, get mean loss
            grouped = problem_data.groupby(['optimizer', 'dimension'])['final_loss'].mean().unstack()

            grouped.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{problem.title()} Function - Final Loss by Optimizer and Dimension')
            ax.set_ylabel('Final Loss (lower is better)')
            ax.set_xlabel('Optimizer')
            ax.legend(title='Dimension', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'performance_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Timing comparison
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        timing_data = success_df.groupby(['optimizer', 'problem'])['time_taken'].mean().unstack()

        timing_data.plot(kind='bar', ax=ax)
        ax.set_title('Average Time Taken by Optimizer and Problem')
        ax.set_ylabel('Time (seconds)')
        ax.set_xlabel('Optimizer')
        ax.legend(title='Problem', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'timing_comparison_{timestamp}.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Plots saved to {output_dir}")

    except Exception as e:
        print(f"Error generating plots: {e}")


# =================================================================================================
# MAIN EXECUTION
# =================================================================================================

def main():
    parser = argparse.ArgumentParser(description='UMACO13 Benchmarking Suite')
    parser.add_argument('--problems', nargs='+',
                       default=['sphere', 'rosenbrock', 'rastrigin'],
                       help='Test problems to run')
    parser.add_argument('--optimizers', nargs='+',
                       default=['umaco'],
                       help='Optimizers to test')
    parser.add_argument('--dimensions', nargs='+', type=int,
                       default=[2, 5, 10],
                       help='Problem dimensions to test')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per test')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    print("UMACO13 Benchmarking Suite")
    print("=" * 50)
    print(f"Problems: {args.problems}")
    print(f"Optimizers: {args.optimizers}")
    print(f"Dimensions: {args.dimensions}")
    print(f"Runs per test: {args.runs}")
    print()

    # Run benchmarks
    results_df = run_benchmark_suite(
        args.problems,
        args.optimizers,
        args.dimensions,
        args.runs,
        args.output_dir
    )

    # Print summary
    print("\nBenchmark Summary:")
    print("=" * 50)

    success_df = results_df[results_df['success'] == True]
    if len(success_df) > 0:
        summary = success_df.groupby(['optimizer', 'problem']).agg({
            'final_loss': ['mean', 'std'],
            'time_taken': ['mean', 'std'],
            'success': 'mean'
        }).round(4)

        print(summary)
    else:
        print("No successful runs to summarize")

    print(f"\nDetailed results saved to {args.output_dir}")


if __name__ == '__main__':
    main()