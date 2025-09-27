"""Validation tests for the UMacoSolver facade."""

from __future__ import annotations

import os

import numpy as np
import pytest  # type: ignore[import-not-found]

# Ensure the solver allows CPU fallback during tests before importing the module under test.
os.environ.setdefault("UMACO_ALLOW_CPU", "1")

from universal_solver import ProblemSpecificationError, UMacoSolver


def test_ambiguous_problem_hints_raise():
    distance_matrix = np.ones((3, 3))
    clauses = [[1, -2]]

    with pytest.raises(ProblemSpecificationError) as exc_info:
        UMacoSolver(distance_matrix=distance_matrix, clauses=clauses)

    message = str(exc_info.value)
    assert "Ambiguous problem specification" in message
    assert "provide problem_type" in message.lower()


def test_distance_matrix_conflict_with_explicit_type():
    distance_matrix = np.ones((3, 3))

    with pytest.raises(ProblemSpecificationError) as exc_info:
        UMacoSolver(distance_matrix=distance_matrix, problem_type="continuous", problem_dim=3)

    assert "distance_matrix" in str(exc_info.value)


def test_missing_problem_dim_for_continuous_problem():
    with pytest.raises(ProblemSpecificationError) as exc_info:
        UMacoSolver(objective="sphere")

    assert "problem_dim" in str(exc_info.value)


def test_custom_loss_must_be_callable():
    with pytest.raises(ProblemSpecificationError) as exc_info:
        UMacoSolver(custom_loss=42, problem_dim=3)

    assert "custom_loss" in str(exc_info.value)
