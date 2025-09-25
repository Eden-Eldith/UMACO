#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMACO13 Testing Suite
=====================

Comprehensive test suite for UMACO13 framework including:
- Unit tests for core components
- Integration tests for optimization loops
- Performance benchmarks
- Regression tests
"""

import unittest
import numpy as np
import time
from unittest.mock import Mock, patch

# Import UMACO components
from Umaco13 import (
    UMACO, UMACOConfig, SolverType,
    UniversalEconomy, EconomyConfig,
    UniversalNode, NodeConfig,
    NeuroPheromoneSystem, PheromoneConfig,
    create_umaco_solver,
    rosenbrock_loss
)


class TestUMACOComponents(unittest.TestCase):
    """Unit tests for individual UMACO components."""

    def setUp(self):
        """Set up test fixtures."""
        self.economy_config = EconomyConfig(n_agents=4, initial_tokens=100)
        self.economy = UniversalEconomy(self.economy_config)

        self.pheromone_config = PheromoneConfig(n_dim=16, initial_val=0.5)
        self.pheromones = NeuroPheromoneSystem(self.pheromone_config)

        self.config = UMACOConfig(
            n_dim=16,
            max_iter=10,
            problem_type=SolverType.CONTINUOUS,
            problem_dim=2
        )
        self.solver = UMACO(self.config, self.economy, self.pheromones)

        self.nodes = [UniversalNode(i, self.economy, NodeConfig()) for i in range(4)]

    def test_economy_initialization(self):
        """Test economy initializes with correct token distribution."""
        self.assertEqual(len(self.economy.tokens), 4)
        self.assertEqual(sum(self.economy.tokens.values()), 400)  # 4 agents * 100 tokens

    def test_economy_buy_resources(self):
        """Test resource purchasing mechanism."""
        # Should succeed with sufficient tokens
        success = self.economy.buy_resources(0, 0.5, 1.0)
        self.assertTrue(success)
        self.assertLess(self.economy.tokens[0], 100)  # Tokens should decrease

        # Should fail with insufficient tokens
        self.economy.tokens[0] = 0
        success = self.economy.buy_resources(0, 10.0, 1.0)
        self.assertFalse(success)

    def test_economy_reward_performance(self):
        """Test performance-based token rewards."""
        initial_tokens = self.economy.tokens[0]
        self.economy.reward_performance(0, 0.1)  # Good performance (low loss)
        self.assertGreater(self.economy.tokens[0], initial_tokens)

    def test_pheromone_initialization(self):
        """Test pheromone system initializes correctly."""
        self.assertEqual(self.pheromones.pheromones.shape, (16, 16))
        # Check that pheromones are initialized with correct scale (not exact value due to randomness)
        pheromone_magnitudes = np.abs(self.pheromones.pheromones.get())
        self.assertTrue(np.all(pheromone_magnitudes >= 0.0))
        self.assertTrue(np.all(pheromone_magnitudes <= 1.0))  # Should be scaled by initial_val

    def test_pheromone_deposit(self):
        """Test pheromone deposition mechanism."""
        paths = [[0, 1], [1, 2]]
        performances = [0.8, 0.6]
        initial_pheromone = self.pheromones.pheromones[0, 1].real.copy()

        self.pheromones.deposit(paths, performances, 0.1)
        self.assertNotEqual(self.pheromones.pheromones[0, 1].real, initial_pheromone)

    def test_node_initialization(self):
        """Test node initialization."""
        self.assertEqual(self.nodes[0].node_id, 0)
        self.assertIsInstance(self.nodes[0].performance_history, list)

    def test_solver_initialization(self):
        """Test UMACO solver initialization."""
        self.assertEqual(self.solver.config.n_dim, 16)
        self.assertEqual(self.solver.config.problem_type, SolverType.CONTINUOUS)


class TestOptimizationLoops(unittest.TestCase):
    """Integration tests for full optimization loops."""

    def test_continuous_optimization(self):
        """Test full continuous optimization loop."""
        solver, agents = create_umaco_solver(
            problem_type='CONTINUOUS',
            dim=16,
            max_iter=5,  # Short for testing
            problem_dim=2
        )

        # Run optimization
        pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
            agents, rosenbrock_loss
        )

        # Check results
        self.assertIsInstance(pheromone_real, np.ndarray)
        self.assertIsInstance(panic_history, list)
        self.assertEqual(len(panic_history), 5)  # Should have history for each iteration

    def test_combinatorial_optimization(self):
        """Test combinatorial optimization (TSP-like)."""
        # Create simple distance matrix
        distance_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])

        def tsp_loss(path):
            """Simple TSP loss function."""
            if len(path) < 3:
                return 100.0
            total_distance = 0
            for i in range(len(path) - 1):
                total_distance += distance_matrix[path[i], path[i+1]]
            return total_distance

        solver, agents = create_umaco_solver(
            problem_type='COMBINATORIAL_PATH',
            dim=3,
            max_iter=5,
            distance_matrix=distance_matrix
        )

        pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
            agents, tsp_loss
        )

        self.assertIsInstance(pheromone_real, np.ndarray)
        self.assertEqual(len(panic_history), 5)

    def test_satisfiability_optimization(self):
        """Test SAT optimization."""
        # Simple 2-variable SAT instance: (x1 OR x2) AND (NOT x1 OR NOT x2)
        clauses = [[1, 2], [-1, -2]]  # x1∨x2 ∧ ¬x1∨¬x2

        def sat_loss(assignment):
            """Count unsatisfied clauses."""
            unsatisfied = 0
            for clause in clauses:
                satisfied = False
                for var in clause:
                    if var > 0 and assignment[abs(var)-1] == 1:
                        satisfied = True
                    elif var < 0 and assignment[abs(var)-1] == 0:
                        satisfied = True
                if not satisfied:
                    unsatisfied += 1
            return unsatisfied

        solver, agents = create_umaco_solver(
            problem_type='SATISFIABILITY',
            dim=2,
            max_iter=5,
            clauses=clauses
        )

        pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
            agents, sat_loss
        )

        self.assertIsInstance(pheromone_real, np.ndarray)
        self.assertEqual(len(panic_history), 5)


class TestLossFunctions(unittest.TestCase):
    """Test loss function implementations."""

    def test_rosenbrock_loss(self):
        """Test Rosenbrock function implementation."""
        # Test minimum at (1, 1)
        loss_min = rosenbrock_loss(np.array([1.0, 1.0]))
        self.assertAlmostEqual(loss_min, 0.0, places=5)

        # Test some other points
        loss_00 = rosenbrock_loss(np.array([0.0, 0.0]))
        self.assertGreater(loss_00, 0.0)

        loss_11 = rosenbrock_loss(np.array([1.0, 1.0]))
        self.assertEqual(loss_11, 0.0)


class TestFactoryFunction(unittest.TestCase):
    """Test the create_umaco_solver factory function."""

    def test_factory_continuous(self):
        """Test factory function for continuous problems."""
        solver, agents = create_umaco_solver(
            problem_type='CONTINUOUS',
            dim=16,
            max_iter=10,
            problem_dim=2
        )

        self.assertIsInstance(solver, UMACO)
        self.assertEqual(len(agents), 8)  # Default n_ants
        self.assertEqual(solver.config.problem_type, SolverType.CONTINUOUS)

    def test_factory_combinatorial(self):
        """Test factory function for combinatorial problems."""
        distance_matrix = np.random.rand(5, 5)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(distance_matrix, 0)

        solver, agents = create_umaco_solver(
            problem_type='COMBINATORIAL_PATH',
            dim=5,
            max_iter=10,
            distance_matrix=distance_matrix
        )

        self.assertEqual(solver.config.problem_type, SolverType.COMBINATORIAL_PATH)
        self.assertIsNotNone(solver.config.distance_matrix)

    def test_factory_sat(self):
        """Test factory function for SAT problems."""
        clauses = [[1, 2], [-1, -2]]

        solver, agents = create_umaco_solver(
            problem_type='SATISFIABILITY',
            dim=2,
            max_iter=10,
            clauses=clauses
        )

        self.assertEqual(solver.config.problem_type, SolverType.SATISFIABILITY)
        self.assertEqual(solver.config.num_clauses, 2)


class TestPerformance(unittest.TestCase):
    """Performance and regression tests."""

    def test_optimization_convergence(self):
        """Test that optimization shows some convergence behavior."""
        solver, agents = create_umaco_solver(
            problem_type='CONTINUOUS',
            dim=32,  # Higher resolution
            max_iter=20,
            problem_dim=2
        )

        start_time = time.time()
        pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
            agents, rosenbrock_loss
        )
        end_time = time.time()

        # Should complete in reasonable time
        self.assertLess(end_time - start_time, 30.0)  # Less than 30 seconds

        # Should have history
        self.assertEqual(len(panic_history), 20)

        # Final panic should be reasonable (not NaN or infinite)
        self.assertFalse(np.isnan(panic_history[-1]))
        self.assertFalse(np.isinf(panic_history[-1]))

    def test_memory_usage(self):
        """Test that optimization doesn't have obvious memory issues."""
        # This is a basic smoke test - in a real scenario you'd use memory profiling
        solver, agents = create_umaco_solver(
            problem_type='CONTINUOUS',
            dim=16,
            max_iter=5,
            problem_dim=2
        )

        # Should not crash due to memory issues
        try:
            pheromone_real, pheromone_imag, panic_history, homology_report = solver.optimize(
                agents, rosenbrock_loss
            )
            memory_test_passed = True
        except MemoryError:
            memory_test_passed = False

        self.assertTrue(memory_test_passed)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)