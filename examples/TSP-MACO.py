import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from Umaco13 import create_umaco_solver, OptimizationResult


@dataclass
class TSPResult:
    """Container for the route discovered by UMACO13."""

    route: np.ndarray
    distance: float
    coordinates: np.ndarray
    optimization: OptimizationResult


def generate_coordinates(num_cities: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate deterministic city coordinates for reproducibility."""

    rng = np.random.default_rng(seed)
    return rng.random((num_cities, 2), dtype=np.float32)


def build_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute the Euclidean distance matrix for the given coordinates."""

    delta = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(delta, axis=2).astype(np.float32)


def tsp_route_length(route: Sequence[int], distances: np.ndarray) -> float:
    """Evaluate the total length of a TSP tour (closing the loop when needed)."""

    tour = np.asarray(route, dtype=np.int64).ravel()
    if tour.size < 2:
        return float("inf")

    total = float(np.sum(distances[tour[:-1], tour[1:]]))
    if tour[-1] != tour[0]:
        total += float(distances[tour[-1], tour[0]])
    return total


def solve_tsp_with_umaco(
    num_cities: int,
    max_iter: int,
    n_ants: Optional[int] = None,
    seed: Optional[int] = 42,
) -> TSPResult:
    """Solve a TSP instance using the universal UMACO13 solver."""

    coords = generate_coordinates(num_cities, seed)
    distances = build_distance_matrix(coords)

    solver, agents = create_umaco_solver(
        problem_type="COMBINATORIAL_PATH",
        dim=num_cities,
        max_iter=max_iter,
        n_ants=n_ants or min(16, num_cities),
        distance_matrix=distances,
    )

    def loss(candidate: Sequence[int]) -> float:
        return tsp_route_length(candidate, distances)

    optimization = solver.optimize(agents, loss)

    best_solution = optimization.best_solution
    if hasattr(best_solution, "get"):
        best_solution = best_solution.get()
    route = np.asarray(best_solution, dtype=np.int64)
    if route.size == 0:
        raise RuntimeError("UMACO13 did not return a valid TSP tour")
    if route[-1] != route[0]:
        route = np.concatenate([route, [route[0]]])

    distance = tsp_route_length(route, distances)

    return TSPResult(
        route=route,
        distance=distance,
        coordinates=coords,
        optimization=optimization,
    )


def _format_route(route: np.ndarray) -> str:
    return " -> ".join(str(int(city)) for city in route)


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Solve TSP using UMACO13's universal solver")
    parser.add_argument("--cities", type=int, default=12, help="Number of cities in the tour")
    parser.add_argument("--max-iter", type=int, default=200, help="UMACO iterations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument(
        "--ants",
        type=int,
        default=None,
        help="Number of stigmergic agents (defaults to min(16, cities))",
    )

    args = parser.parse_args()

    result = solve_tsp_with_umaco(
        num_cities=args.cities,
        max_iter=args.max_iter,
        n_ants=args.ants,
        seed=args.seed,
    )

    print("UMACO13 TSP Solution")
    print("=====================")
    print(f"Cities           : {args.cities}")
    print(f"Iterations       : {args.max_iter}")
    print(f"Agents           : {args.ants or min(16, args.cities)}")
    print(f"Final distance   : {result.distance:.4f}")
    print(f"Final route      : {_format_route(result.route)}")
    print(f"Final panic level: {result.optimization.panic_history[-1]:.4f}")


if __name__ == "__main__":
    run_cli()