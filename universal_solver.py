"""universal_solver.py
========================

GPU-first universal entry point for the UMACO ecosystem. This module exposes a
single high-level :class:`UMacoSolver` that orchestrates the complex
Panic–Anxiety–Quantum (PAQ) triad, topological stigmergy, and economic dynamics
implemented in :mod:`umaco.Umaco13` while providing an ergonomic, AI-friendly
API. The solver automatically interprets common optimization problem
specifications, constructs appropriate loss functions, and delegates execution
to the underlying GPU-accelerated UMACO engine.

The design goals for this interface are:

* **Zero-guessing problem onboarding.** Accept raw problem artefacts such as
  distance matrices or SAT clauses and infer a consistent solver
  configuration.
* **Crisis-driven defaults.** Preserve UMACO's architectural intent by
  preferring the existing hyperparameter dynamics and GPU-first execution
  semantics.
* **Expert override controls.** Allow knowledgeable users (human or AI) to
  surgically override any configuration element while keeping validation and
  error messaging explicit.
* **Self-documenting API.** Provide type hints, docstrings, logging, and
  illustrative usage examples so downstream AI agents can interact with the
  solver safely.

Example
-------

.. code-block:: python

    from universal_solver import UMacoSolver
    import numpy as np

    # Travelling Salesperson Problem (distance matrix provided ⇒ automatic
    # COMBINATORIAL_PATH configuration)
    distances = np.random.rand(15, 15)
    distances = (distances + distances.T) / 2  # Symmetric for realism
    solver = UMacoSolver(distance_matrix=distances, max_iter=200)
    result = solver.optimize()
    print(result['score'])

    # Custom continuous objective with explicit dimensionality
    def my_loss(matrix: np.ndarray) -> float:
        return float(np.sum((matrix - 0.5)**2))

    solver = UMacoSolver(custom_loss=my_loss, problem_dim=32, n_ants=12)
    solution_bundle = solver.optimize()

    # SAT with clause list (each clause is a list of literals with 1-based
    # indices; negatives encode logical negation)
    clauses = [[1, -2, 3], [-1, 2, 4]]
    solver = UMacoSolver(clauses=clauses)
    sat_result = solver.optimize()

The :func:`optimize` method always delegates to the GPU-native
:class:`~umaco.Umaco13.UMACO` implementation, returning a dictionary containing
best solution, score, history, and configuration context for downstream use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    import numpy as cp  # Use numpy as fallback
    HAS_CUPY = False
import numpy as np

from Umaco13 import (
    SolverType,
    UMACO,
    UniversalNode,
    create_umaco_solver,
    rastrigin_loss,
    rosenbrock_loss,
    sat_loss,
    sphere_loss,
    tsp_loss,
)

__all__ = ["ProblemSpecificationError", "UMacoSolver"]


logger = logging.getLogger("UMacoSolver")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class ProblemSpecificationError(ValueError):
    """Exception raised when an invalid or incomplete problem specification is provided."""


@dataclass(frozen=True)
class SolverResult:
    """Container returned by :meth:`UMacoSolver.optimize`.

    Attributes
    ----------
    best_solution:
        The best candidate discovered by the optimization loop. For simulation
        problems this is the solver itself; for other domains it is typically a
        :class:`numpy.ndarray`.
    score:
        Best performance score (higher is better).
    history:
        Diagnostic history captured by the underlying UMACO engine.
    optimizer:
        Reference to the configured :class:`~umaco.Umaco13.UMACO` instance for
        advanced post-processing.
    agents:
        The list of :class:`~umaco.Umaco13.UniversalNode` agents that
        participated in the run.
    """

    best_solution: Any
    score: float
    history: Mapping[str, List[float]]
    optimizer: UMACO
    agents: Sequence[UniversalNode]


class UMacoSolver:
    """High-level, AI-oriented facade for configuring and executing UMACO.

    Parameters
    ----------
    problem_type:
        Optional explicit problem category name matching
        :class:`~umaco.Umaco13.SolverType`. If omitted, the solver attempts to
        infer the type from the remaining keyword arguments.
    custom_loss:
        Optional callable mapping a candidate solution to a scalar loss. When
        supplied, it takes precedence over automatically generated loss
        templates.
    problem_dim:
        Dimensionality of the search space. Required for continuous problems
        without auto-deducible structure.
    distance_matrix:
        Square matrix encoding pairwise distances. Presence of this argument
        triggers COMBINATORIAL_PATH behaviour unless ``problem_type`` explicitly
        requests something else.
    clauses:
        Iterable of clauses where each clause is an iterable of integer literals
        using the conventional 1-based SAT encoding with negatives indicating
        logical negation.
    objective:
        Name of a built-in continuous objective (``"sphere"``, ``"rastrigin"``,
        or ``"rosenbrock"``). Ignored when ``custom_loss`` is provided.
    max_iter:
        Override for the number of optimization iterations. When omitted, a
        type-dependent default is selected.
    n_ants:
        Number of cognitive agents driving the stigmergic field.
    config_overrides:
        Mapping of configuration attribute names to override within the
        underlying :class:`~umaco.Umaco13.UMACO` configuration. This allows
        expert users to adapt PAQ or stigmergic behaviour without rewriting the
        solver bootstrap.
    loss_kwargs:
        Additional keyword arguments forwarded to the generated loss function
        when invoked.

    Notes
    -----
    * The constructor validates the presence of a CUDA-capable GPU; UMACO is a
      GPU-first framework and will raise a runtime error when no GPU is
      available.
    * Input validation is intentionally strict. The goal is to prevent silent
      misconfiguration and uphold the "impossible to use incorrectly" mantra.
    """

    _CONTINUOUS_OBJECTIVES: Mapping[str, Callable[[np.ndarray], float]] = {
        "sphere": sphere_loss,
        "rastrigin": rastrigin_loss,
        "rosenbrock": rosenbrock_loss,
    }

    def __init__(
        self,
        *,
        problem_type: Optional[str] = None,
        custom_loss: Optional[Callable[[Any], float]] = None,
        problem_dim: Optional[int] = None,
        distance_matrix: Optional[np.ndarray] = None,
        clauses: Optional[Iterable[Iterable[int]]] = None,
        objective: str = "sphere",
        max_iter: Optional[int] = None,
        n_ants: Optional[int] = None,
        config_overrides: Optional[Mapping[str, Any]] = None,
        loss_kwargs: Optional[Mapping[str, Any]] = None,
        **solver_kwargs: Any,
    ) -> None:
        self._validate_gpu_presence()

        self._raw_distance_matrix = self._normalize_distance_matrix(distance_matrix)
        self._raw_clauses = self._normalize_clauses(clauses)
        self._loss_kwargs: Dict[str, Any] = dict(loss_kwargs or {})

        self.problem_type = self._resolve_problem_type(problem_type, custom_loss)
        self.problem_dim = self._resolve_problem_dim(problem_dim)
        self.loss_fn = self._build_loss_function(custom_loss, objective)

        self.max_iter = max_iter or self._default_iterations()
        self.n_ants = n_ants or self._default_ants()
        self._config_overrides = dict(config_overrides or {})
        self._solver_kwargs = dict(solver_kwargs)

        self.optimizer, self.agents = self._bootstrap_solver()
        self._apply_overrides()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        *,
        iterations: Optional[int] = None,
        loss_fn: Optional[Callable[[Any], float]] = None,
    ) -> SolverResult:
        """Execute the GPU-accelerated optimization loop.

        Parameters
        ----------
        iterations:
            Optional override for ``optimizer.config.max_iter`` during this
            invocation.
        loss_fn:
            Optional temporary loss function to evaluate candidate solutions.
            When omitted, the loss prepared during initialization is reused.

        Returns
        -------
        SolverResult
            Structured result bundle containing the best solution, score,
            diagnostic history, and references to the optimizer and agents.
        """

        if iterations is not None:
            if iterations <= 0:
                raise ValueError("iterations must be a positive integer")
            self.optimizer.config.max_iter = int(iterations)
        else:
            self.optimizer.config.max_iter = int(self.max_iter)

        active_loss = loss_fn or self.loss_fn
        if active_loss is None:
            raise ProblemSpecificationError("A loss function must be defined before optimization")

        best_solution, score, history = self.optimizer.optimize(self.agents, active_loss)
        return SolverResult(
            best_solution=best_solution,
            score=float(score),
            history=history,
            optimizer=self.optimizer,
            agents=self.agents,
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def solver_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the current solver configuration."""

        return {
            "problem_type": self.problem_type.name,
            "problem_dim": self.problem_dim,
            "max_iter": self.optimizer.config.max_iter,
            "n_ants": len(self.agents),
            "loss_function": getattr(self.loss_fn, "__name__", repr(self.loss_fn)),
            "config_overrides": self._config_overrides.copy(),
        }

    @classmethod
    def available_objectives(cls) -> Tuple[str, ...]:
        """List the names of built-in continuous objectives."""

        return tuple(sorted(cls._CONTINUOUS_OBJECTIVES))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_gpu_presence(self) -> None:
        """Ensure a CUDA-capable GPU is visible before continuing."""

        try:
            device_count = cp.cuda.runtime.getDeviceCount()
        except cp.cuda.runtime.CUDARuntimeError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "UMACO requires a CUDA-capable GPU. No compatible device was detected."
            ) from exc

        if device_count < 1:  # pragma: no cover - environment-specific
            raise RuntimeError("UMACO requires at least one CUDA-capable GPU")

    def _resolve_problem_type(
        self,
        explicit: Optional[str],
        custom_loss: Optional[Callable[[Any], float]],
    ) -> SolverType:
        if explicit is not None:
            try:
                solver_type = SolverType[explicit.upper()]
            except KeyError as exc:  # pragma: no cover - defensive
                raise ProblemSpecificationError(
                    f"Unknown problem_type '{explicit}'. Valid options: {[e.name.lower() for e in SolverType]}"
                ) from exc
            self._validate_type_consistency(solver_type)
            return solver_type

        if self._raw_distance_matrix is not None:
            solver_type = SolverType.COMBINATORIAL_PATH
        elif self._raw_clauses is not None:
            solver_type = SolverType.SATISFIABILITY
        elif custom_loss is not None:
            solver_type = SolverType.CONTINUOUS
        else:
            solver_type = SolverType.CONTINUOUS

        self._validate_type_consistency(solver_type)
        return solver_type

    def _validate_type_consistency(self, solver_type: SolverType) -> None:
        if solver_type == SolverType.COMBINATORIAL_PATH and self._raw_distance_matrix is None:
            raise ProblemSpecificationError("COMBINATORIAL_PATH problems require a distance_matrix")
        if solver_type == SolverType.SATISFIABILITY and self._raw_clauses is None:
            raise ProblemSpecificationError("SATISFIABILITY problems require clauses")
        if solver_type == SolverType.SIMULATION and not any((self._raw_distance_matrix, self._raw_clauses)):
            # Simulation requires an explicit custom loss
            return

    def _resolve_problem_dim(self, supplied: Optional[int]) -> int:
        if self.problem_type == SolverType.COMBINATORIAL_PATH:
            assert self._raw_distance_matrix is not None  # for mypy
            return int(self._raw_distance_matrix.shape[0])

        if self.problem_type == SolverType.SATISFIABILITY:
            assert self._raw_clauses is not None  # for mypy
            return self._infer_variable_count(self._raw_clauses)

        if supplied is None:
            if self.problem_type == SolverType.CONTINUOUS:
                raise ProblemSpecificationError(
                    "problem_dim must be provided for continuous problems when it cannot be inferred"
                )
            raise ProblemSpecificationError("problem_dim is required for the selected problem type")

        if supplied <= 0:
            raise ProblemSpecificationError("problem_dim must be a positive integer")
        return int(supplied)

    def _build_loss_function(
        self,
        custom_loss: Optional[Callable[[Any], float]],
        objective: str,
    ) -> Callable[[Any], float]:
        if custom_loss is not None:
            return self._wrap_loss(custom_loss)

        if self.problem_type == SolverType.COMBINATORIAL_PATH:
            assert self._raw_distance_matrix is not None
            return self._build_tsp_loss(self._raw_distance_matrix)
        if self.problem_type == SolverType.SATISFIABILITY:
            assert self._raw_clauses is not None
            return self._build_sat_loss(self._raw_clauses)
        if self.problem_type == SolverType.CONTINUOUS:
            try:
                base_loss = self._CONTINUOUS_OBJECTIVES[objective.lower()]
            except KeyError as exc:
                raise ProblemSpecificationError(
                    f"Unknown objective '{objective}'. Available: {self.available_objectives()}"
                ) from exc
            return self._wrap_loss(base_loss)
        if self.problem_type == SolverType.SIMULATION:
            raise ProblemSpecificationError(
                "SIMULATION mode requires a custom_loss describing system evolution"
            )

        raise ProblemSpecificationError(f"No loss construction path for {self.problem_type}")

    def _wrap_loss(self, loss_fn: Callable[[Any], float]) -> Callable[[Any], float]:
        def wrapped(candidate: Any) -> float:
            value = loss_fn(candidate, **self._loss_kwargs) if self._loss_kwargs else loss_fn(candidate)
            if not np.isfinite(value):
                raise ValueError("Loss function returned a non-finite value")
            return float(value)

        wrapped.__name__ = getattr(loss_fn, "__name__", loss_fn.__class__.__name__)
        return wrapped

    def _build_tsp_loss(self, distance_matrix: np.ndarray) -> Callable[[np.ndarray], float]:
        def loss(path: np.ndarray) -> float:
            path_array = np.asarray(path, dtype=int)
            if path_array.ndim != 1:
                raise ValueError("TSP candidate must be a 1D array of city indices")
            if path_array.size < 2:
                raise ValueError("TSP candidate must contain at least two cities")
            if np.any(path_array < 0) or np.any(path_array >= distance_matrix.shape[0]):
                raise ValueError("TSP candidate contains invalid city indices")
            return tsp_loss(path_array, distance_matrix)

        loss.__name__ = "auto_tsp_loss"
        return loss

    def _build_sat_loss(self, clauses: Sequence[Sequence[int]]) -> Callable[[np.ndarray], float]:
        def loss(assignment: np.ndarray) -> float:
            arr = np.asarray(assignment, dtype=int)
            if arr.ndim != 1:
                raise ValueError("SAT candidate must be a 1D assignment vector")
            if arr.size != self.problem_dim:
                raise ValueError(
                    f"SAT candidate dimensionality {arr.size} does not match expected {self.problem_dim}"
                )
            if not np.all((arr == 0) | (arr == 1)):
                raise ValueError("SAT candidate must contain binary values {0, 1}")
            return sat_loss(arr, clauses)

        loss.__name__ = "auto_sat_loss"
        return loss

    def _default_iterations(self) -> int:
        if self.problem_type == SolverType.COMBINATORIAL_PATH:
            return 500
        if self.problem_type == SolverType.SATISFIABILITY:
            return 400
        if self.problem_type == SolverType.SIMULATION:
            return 300
        return 250

    def _default_ants(self) -> int:
        if self.problem_type == SolverType.COMBINATORIAL_PATH:
            return 16
        if self.problem_type == SolverType.SATISFIABILITY:
            return 12
        if self.problem_type == SolverType.SIMULATION:
            return 20
        return 8

    def _bootstrap_solver(self) -> Tuple[UMACO, List[UniversalNode]]:
        creation_kwargs: Dict[str, Any] = dict(self._solver_kwargs)
        if self.problem_type == SolverType.COMBINATORIAL_PATH:
            creation_kwargs["distance_matrix"] = self._raw_distance_matrix
        elif self.problem_type == SolverType.SATISFIABILITY:
            creation_kwargs["clauses"] = self._raw_clauses

        optimizer, agents = create_umaco_solver(
            problem_type=self.problem_type.name,
            dim=self.problem_dim,
            max_iter=self.max_iter,
            n_ants=self.n_ants,
            **creation_kwargs,
        )

        logger.info(
            "Initialized UMACO solver | type=%s | dim=%d | max_iter=%d | n_ants=%d",
            self.problem_type.name,
            self.problem_dim,
            self.max_iter,
            self.n_ants,
        )

        return optimizer, agents

    def _apply_overrides(self) -> None:
        if not self._config_overrides:
            return

        for key, value in self._config_overrides.items():
            if not hasattr(self.optimizer.config, key):
                raise ProblemSpecificationError(
                    f"Unknown configuration attribute '{key}' for override"
                )
            setattr(self.optimizer.config, key, value)
            logger.debug("Applied config override %s=%r", key, value)

    # ------------------------------------------------------------------
    # Normalization & inference utilities
    # ------------------------------------------------------------------

    def _normalize_distance_matrix(
        self, distance_matrix: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if distance_matrix is None:
            return None

        matrix = np.asarray(distance_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ProblemSpecificationError("distance_matrix must be a square 2D array")
        if matrix.shape[0] < 2:
            raise ProblemSpecificationError("distance_matrix must describe at least two nodes")
        if np.any(matrix < 0):
            raise ProblemSpecificationError("distance_matrix cannot contain negative distances")
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            logger.warning("Distance matrix is not symmetric; proceeding but optimization may struggle")
        return matrix

    def _normalize_clauses(
        self, clauses: Optional[Iterable[Iterable[int]]]
    ) -> Optional[List[List[int]]]:
        if clauses is None:
            return None

        normalized: List[List[int]] = []
        for clause_idx, clause in enumerate(clauses):
            clause_list = [int(lit) for lit in clause]
            if not clause_list:
                raise ProblemSpecificationError(f"Clause {clause_idx} is empty")
            if any(lit == 0 for lit in clause_list):
                raise ProblemSpecificationError("Literals use 1-based indexing; zero is not allowed")
            normalized.append(clause_list)

        if not normalized:
            raise ProblemSpecificationError("At least one clause must be provided for SAT problems")

        return normalized

    def _infer_variable_count(self, clauses: Sequence[Sequence[int]]) -> int:
        highest_literal = max(abs(literal) for clause in clauses for literal in clause)
        if highest_literal <= 0:
            raise ProblemSpecificationError("Unable to infer variable count from clauses")
        return highest_literal


if __name__ == "__main__":  # pragma: no cover - illustrative usage only
    logging.getLogger().setLevel(logging.INFO)
    print("UMacoSolver module provides the UMacoSolver class. Import it to configure optimizers.")
    print("Available continuous objectives:", UMacoSolver.available_objectives())
