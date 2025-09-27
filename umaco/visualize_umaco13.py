#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMACO13 Visualization Dashboard
==============================

Advanced visualization tools for UMACO13 optimization:
- Real-time optimization progress plots
- Pheromone matrix heatmaps
- Panic history and convergence tracking
- Topology visualization (if available)
- Performance comparison dashboards

Usage:
    python visualize_umaco13.py --mode live --problem rosenbrock
    python visualize_umaco13.py --mode analysis --results benchmark_results/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import time
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

# Import UMACO
from Umaco13 import create_umaco_solver, rosenbrock_loss

# Try to import topology visualization
try:
    from ripser import ripser
    from persim import PersistenceImager
    TOPOLOGY_AVAILABLE = True
except ImportError:
    TOPOLOGY_AVAILABLE = False


# =================================================================================================
# VISUALIZATION CLASSES
# =================================================================================================

class UMACOVisualizer:
    """Main visualization class for UMACO13."""

    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        self.figsize = figsize
        plt.style.use('seaborn-v0_8-darkgrid' if hasattr(plt.style, 'available') else 'default')
        sns.set_palette("husl")

    def plot_optimization_progress(self, panic_history: List[float],
                                loss_history: Optional[List[float]] = None,
                                title: str = "UMACO13 Optimization Progress") -> plt.Figure:
        """Plot optimization progress over time."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        iterations = list(range(len(panic_history)))

        # Panic level plot
        ax1.plot(iterations, panic_history, 'r-', linewidth=2, label='Panic Level')
        ax1.fill_between(iterations, panic_history, alpha=0.3, color='red')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Panic Level', color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.set_title('Crisis Response Over Time')
        ax1.grid(True, alpha=0.3)

        # Loss history plot (if available)
        if loss_history:
            ax2.plot(iterations, loss_history, 'b-', linewidth=2, label='Loss')
            ax2.fill_between(iterations, loss_history, alpha=0.3, color='blue')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Loss Value', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_yscale('log')
            ax2.set_title('Optimization Loss')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Loss history not available',
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Loss History (N/A)')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_pheromone_heatmap(self, pheromone_real: np.ndarray,
                             pheromone_imag: Optional[np.ndarray] = None,
                             title: str = "Pheromone Matrix") -> plt.Figure:
        """Plot pheromone matrix as heatmap."""
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Real part
        im1 = axes[0].imshow(pheromone_real, cmap='viridis', aspect='equal')
        axes[0].set_title('Real Part (Attraction)')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        axes[0].set_xlabel('Node j')
        axes[0].set_ylabel('Node i')

        # Imaginary part
        if pheromone_imag is not None:
            im2 = axes[1].imshow(pheromone_imag, cmap='plasma', aspect='equal')
            axes[1].set_title('Imaginary Part (Repulsion)')
            plt.colorbar(im2, ax=axes[1], shrink=0.8)
            axes[1].set_xlabel('Node j')
            axes[1].set_ylabel('Node i')
        else:
            axes[1].text(0.5, 0.5, 'No imaginary part',
                        transform=axes[1].transAxes, ha='center', va='center')

        # Magnitude
        magnitude = np.abs(pheromone_real + 1j * (pheromone_imag if pheromone_imag is not None else 0))
        im3 = axes[2].imshow(magnitude, cmap='inferno', aspect='equal')
        axes[2].set_title('Pheromone Magnitude')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        axes[2].set_xlabel('Node j')
        axes[2].set_ylabel('Node i')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_topology_analysis(self, pheromone_matrix: np.ndarray,
                             title: str = "Topological Analysis") -> plt.Figure:
        """Plot topological analysis of pheromone field."""
        if not TOPOLOGY_AVAILABLE:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, 'Topology analysis requires ripser and persim packages',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title(title)
            return fig

        try:
            # Clean the pheromone matrix (remove NaN and infinite values)
            clean_matrix = np.nan_to_num(pheromone_matrix, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure it's symmetric and has zero diagonal for distance matrix
            clean_matrix = (clean_matrix + clean_matrix.T) / 2
            np.fill_diagonal(clean_matrix, 0)
            
            # Compute persistent homology
            diagrams = ripser(clean_matrix, maxdim=2, distance_matrix=True)['dgms']

            # Create persistence images
            pimager = PersistenceImager(pixel_size=0.2, birth_range=(0, 1), pers_range=(0, 1))

            fig, axes = plt.subplots(2, 2, figsize=self.figsize)

            # Plot persistence diagrams
            for i, dgm in enumerate(diagrams[:2]):  # H0 and H1
                ax = axes[0, i]
                if len(dgm) > 0:
                    # Filter out infinite death times for visualization
                    finite_mask = np.isfinite(dgm[:, 1])
                    if np.any(finite_mask):
                        finite_dgm = dgm[finite_mask]
                        ax.scatter(finite_dgm[:, 0], finite_dgm[:, 1], alpha=0.6, s=20)
                        max_val = max(np.max(finite_dgm[:, 0]), np.max(finite_dgm[:, 1]))
                        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'No finite persistence pairs',
                               transform=ax.transAxes, ha='center', va='center')
                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.set_title(f'H{i} Persistence Diagram')
                ax.set_aspect('equal')

            # Plot persistence images
            for i, dgm in enumerate(diagrams[:2]):
                ax = axes[1, i]
                if len(dgm) > 0:
                    # Filter out infinite death times for persistence image
                    finite_mask = np.isfinite(dgm[:, 1])
                    if np.any(finite_mask):
                        finite_dgm = dgm[finite_mask]
                        img = pimager.transform([finite_dgm])[0]
                        im = ax.imshow(img, origin='lower', extent=pimager.extent, cmap='viridis')
                        plt.colorbar(im, ax=ax, shrink=0.8)
                    else:
                        ax.text(0.5, 0.5, 'No finite persistence pairs',
                               transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'H{i} Persistence Image')

            fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            return fig

        except Exception as e:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            ax.text(0.5, 0.5, f'Topology analysis failed: {str(e)}',
                   transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return fig

    def plot_convergence_analysis(self, multiple_runs_data: List[Dict],
                                title: str = "Convergence Analysis") -> plt.Figure:
        """Plot convergence analysis across multiple runs."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        # Extract data
        all_panic = []
        all_losses = []
        final_losses = []
        convergence_times = []

        for run_data in multiple_runs_data:
            if 'panic_history' in run_data:
                all_panic.append(run_data['panic_history'])
            if 'loss_history' in run_data:
                all_losses.append(run_data['loss_history'])
            if 'final_loss' in run_data:
                final_losses.append(run_data['final_loss'])
            if 'convergence_time' in run_data:
                convergence_times.append(run_data['convergence_time'])

        # Plot 1: Panic trajectories
        ax = axes[0, 0]
        for i, panic in enumerate(all_panic):
            ax.plot(panic, alpha=0.7, label=f'Run {i+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Panic Level')
        ax.set_title('Panic Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Loss trajectories
        ax = axes[0, 1]
        for i, loss in enumerate(all_losses):
            ax.plot(loss, alpha=0.7, label=f'Run {i+1}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_yscale('log')
        ax.set_title('Loss Trajectories')
        ax.grid(True, alpha=0.3)

        # Plot 3: Final loss distribution
        ax = axes[1, 0]
        if final_losses:
            ax.hist(final_losses, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(final_losses), color='red', linestyle='--',
                      label=f'Mean: {np.mean(final_losses):.4f}')
            ax.set_xlabel('Final Loss')
            ax.set_ylabel('Frequency')
            ax.set_title('Final Loss Distribution')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No final loss data',
                   transform=ax.transAxes, ha='center', va='center')

        # Plot 4: Convergence time distribution
        ax = axes[1, 1]
        if convergence_times:
            ax.hist(convergence_times, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(convergence_times), color='red', linestyle='--',
                      label=f'Mean: {np.mean(convergence_times):.2f}s')
            ax.set_xlabel('Convergence Time (s)')
            ax.set_ylabel('Frequency')
            ax.set_title('Convergence Time Distribution')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No convergence time data',
                   transform=ax.transAxes, ha='center', va='center')

        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_animation(self, optimization_history: List[Dict],
                        title: str = "UMACO13 Optimization Animation") -> animation.FuncAnimation:
        """Create animation of optimization process."""
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)

        def animate(frame):
            fig.clear()
            axes = fig.subplots(2, 2)

            if frame < len(optimization_history):
                data = optimization_history[frame]

                # Pheromone heatmap
                if 'pheromone_real' in data:
                    im = axes[0, 0].imshow(data['pheromone_real'], cmap='viridis', animated=True)
                    axes[0, 0].set_title(f'Pheromone Matrix (Iteration {frame})')
                    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

                # Panic plot
                if 'panic_history' in data:
                    panic_up_to_now = data['panic_history'][:frame+1]
                    axes[0, 1].plot(panic_up_to_now, 'r-', linewidth=2)
                    axes[0, 1].fill_between(range(len(panic_up_to_now)), panic_up_to_now, alpha=0.3, color='red')
                    axes[0, 1].set_title('Panic Level Over Time')
                    axes[0, 1].set_xlabel('Iteration')
                    axes[0, 1].set_ylabel('Panic Level')
                    axes[0, 1].grid(True, alpha=0.3)

                # Loss plot
                if 'loss_history' in data:
                    loss_up_to_now = data['loss_history'][:frame+1]
                    axes[1, 0].plot(loss_up_to_now, 'b-', linewidth=2)
                    axes[1, 0].set_yscale('log')
                    axes[1, 0].set_title('Loss Over Time')
                    axes[1, 0].set_xlabel('Iteration')
                    axes[1, 0].set_ylabel('Loss')
                    axes[1, 0].grid(True, alpha=0.3)

                # Current best solution (if available)
                if 'current_best' in data:
                    best = data['current_best']
                    axes[1, 1].scatter([best[0]], [best[1]], c='red', s=100, marker='*')
                    axes[1, 1].set_title('Current Best Solution')
                    axes[1, 1].set_xlabel('x1')
                    axes[1, 1].set_ylabel('x2')
                    axes[1, 1].grid(True, alpha=0.3)

            fig.suptitle(f'{title} - Frame {frame}', fontsize=14, fontweight='bold')
            plt.tight_layout()

        anim = animation.FuncAnimation(fig, animate, frames=len(optimization_history),
                                     interval=500, repeat=True)
        return anim


# =================================================================================================
# VISUALIZATION MODES
# =================================================================================================

def run_live_visualization(problem_name: str = 'rosenbrock', max_iter: int = 50):
    """Run live visualization during optimization."""
    print(f"Running live visualization for {problem_name} problem...")

    # Set up problem
    if problem_name == 'rosenbrock':
        loss_func = rosenbrock_loss
        problem_dim = 2
    else:
        print(f"Unknown problem: {problem_name}")
        return

    # Create visualizer
    viz = UMACOVisualizer()

    # Run optimization with progress tracking
    solver, agents = create_umaco_solver(
        problem_type='CONTINUOUS',
        dim=32,
        max_iter=max_iter,
        problem_dim=problem_dim
    )

    # Custom optimization loop to capture intermediate states
    optimization_history = []
    panic_history = []

    # Initialize tracking
    current_pheromone = solver.pheromones.pheromones.copy()
    current_panic = solver.panic_tensor.copy()

    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")

        # Store current state
        state = {
            'iteration': iteration,
            'pheromone_real': solver.pheromones.pheromones.real.get(),
            'panic_history': panic_history.copy()
        }
        optimization_history.append(state)

        # Run one iteration (this would need to be modified in the actual solver)
        # For now, just run the full optimization and show final results
        break

    # Run full optimization
    result = solver.optimize(agents, loss_func)
    pheromone_real = result.pheromone_real
    pheromone_imag = result.pheromone_imag
    panic_history = result.panic_history
    loss_history = result.loss_history
    homology_report = result.homology_report

    # Create visualizations
    fig1 = viz.plot_optimization_progress(panic_history,
                                        loss_history,  # Now we have loss history!
                                        title=f"UMACO13 {problem_name.title()} Optimization")
    fig1.savefig(f'umaco_{problem_name}_progress.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out for headless environments

    fig2 = viz.plot_pheromone_heatmap(pheromone_real,
                                    pheromone_imag if pheromone_imag is not None else None,
                                    title=f"Final Pheromone Matrix - {problem_name.title()}")
    fig2.savefig(f'umaco_{problem_name}_pheromones.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out for headless environments

    if TOPOLOGY_AVAILABLE:
        fig3 = viz.plot_topology_analysis(pheromone_real,
                                        title=f"Topological Analysis - {problem_name.title()}")
        fig3.savefig(f'umaco_{problem_name}_topology.png', dpi=300, bbox_inches='tight')
        # plt.show()  # Commented out for headless environments

    print("Live visualization complete. Check the generated PNG files.")


def run_analysis_visualization(results_dir: str = 'benchmark_results'):
    """Run analysis visualization from benchmark results."""
    print(f"Running analysis visualization from {results_dir}...")

    # Load benchmark results
    csv_files = [f for f in os.listdir(results_dir) if f.startswith('benchmark_results_') and f.endswith('.csv')]

    if not csv_files:
        print(f"No benchmark results found in {results_dir}")
        return

    # Load the most recent results
    latest_file = sorted(csv_files)[-1]
    df = pd.read_csv(os.path.join(results_dir, latest_file))

    print(f"Loaded results from {latest_file}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Create visualizer
    viz = UMACOVisualizer()

    # Filter successful runs
    success_df = df[df['success'] == True].copy()

    if len(success_df) == 0:
        print("No successful runs found")
        return

    # Create comparison plots
    problems = success_df['problem'].unique()

    for problem in problems:
        problem_data = success_df[success_df['problem'] == problem]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Performance by optimizer and dimension
        perf_data = problem_data.pivot_table(
            values='final_loss', index='optimizer', columns='dimension', aggfunc='mean'
        )
        perf_data.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title(f'{problem.title()} - Final Loss by Optimizer')
        axes[0, 0].set_ylabel('Final Loss (lower better)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Timing comparison
        timing_data = problem_data.groupby('optimizer')['time_taken'].agg(['mean', 'std'])
        timing_data['mean'].plot(kind='bar', yerr=timing_data['std'], ax=axes[0, 1], capsize=5)
        axes[0, 1].set_title(f'{problem.title()} - Timing Comparison')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Success rate
        success_rate = problem_data.groupby('optimizer')['success'].mean()
        success_rate.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title(f'{problem.title()} - Success Rate')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Loss distribution
        for optimizer in problem_data['optimizer'].unique():
            opt_data = problem_data[problem_data['optimizer'] == optimizer]
            axes[1, 1].hist(opt_data['final_loss'], alpha=0.5, label=optimizer, bins=20)
        axes[1, 1].set_title(f'{problem.title()} - Loss Distribution')
        axes[1, 1].set_xlabel('Final Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()

        fig.suptitle(f'UMACO13 Benchmark Analysis - {problem.title()}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'analysis_{problem}_{time.strftime("%Y%m%d_%H%M%S")}.png'),
                   dpi=300, bbox_inches='tight')
        plt.show()

    print("Analysis visualization complete.")


# =================================================================================================
# MAIN EXECUTION
# =================================================================================================

def main():
    parser = argparse.ArgumentParser(description='UMACO13 Visualization Dashboard')
    parser.add_argument('--mode', choices=['live', 'analysis'],
                       default='live', help='Visualization mode')
    parser.add_argument('--problem', default='rosenbrock',
                       help='Problem for live visualization')
    parser.add_argument('--results-dir', default='benchmark_results',
                       help='Directory containing benchmark results for analysis')
    parser.add_argument('--max-iter', type=int, default=50,
                       help='Maximum iterations for live visualization')

    args = parser.parse_args()

    print("UMACO13 Visualization Dashboard")
    print("=" * 50)

    if args.mode == 'live':
        run_live_visualization(args.problem, args.max_iter)
    elif args.mode == 'analysis':
        run_analysis_visualization(args.results_dir)


if __name__ == '__main__':
    main()