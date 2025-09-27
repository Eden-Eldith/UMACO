import numpy as np
from dataclasses import dataclass


@dataclass
class TSPResult:
    route: np.ndarray
    distance: float
    history: list

class TSP_MACO:
    def __init__(self, num_cities, alpha=1.0, beta=0.5, rho=0.1, max_iterations=1000, coordinates=None, rng=None):
        self.num_cities = num_cities
        self.alpha = alpha  # Influence of pheromone on the movement of ants
        self.beta = beta  # Influence of heuristic (distance to the next city) on the movement of ants
        self.rho = rho  # Pheromone evaporation rate
        self.max_iterations = max_iterations
        self.random_state = np.random.default_rng() if rng is None else rng
        self.coordinates = coordinates if coordinates is not None else self.random_state.random((num_cities, 2))
        self.distances = self.calculate_distances()
        self.pheromone = np.full((num_cities, num_cities), 1.0, dtype=float)
        self.best_route = None
        self.best_distance = float('inf')
        self.history = []

    def calculate_distances(self):
        """Calculate the Euclidean distances between all pairs of cities."""
        delta = self.coordinates[:, None, :] - self.coordinates[None, :, :]
        return np.linalg.norm(delta, axis=2)

    def pheromone_update(self, route, distance):
        """Update the pheromone matrix based on the quality of the solution (route)."""
        evaporation_factor = 1 - self.rho
        self.pheromone *= evaporation_factor
        deposit_amount = 1.0 / max(distance, 1e-9)
        for idx in range(len(route) - 1):
            i, j = route[idx], route[idx + 1]
            self.pheromone[i, j] += deposit_amount
            self.pheromone[j, i] += deposit_amount

    def construct_route(self):
        """Construct a new route using the pheromone matrix."""
        route = [0]  # Start at city 0
        unvisited_cities = set(range(1, self.num_cities))
        while unvisited_cities:
            current_city = route[-1]
            next_city_options = list(unvisited_cities)
            if next_city_options:
                probabilities = self.probability(route, next_city_options)
                if probabilities.sum() == 0 or np.isnan(probabilities).any():
                    probabilities = np.ones_like(probabilities) / len(probabilities)
                next_city = self.random_state.choice(next_city_options, p=probabilities)
                route.append(next_city)
                unvisited_cities.remove(next_city)
        route.append(0)  # Return to the starting city
        return np.array(route)

    def probability(self, route, next_city_options):
        """Calculate the probabilities for choosing the next city based on pheromone and heuristic."""
        last_city = route[-1]
        pheromones = np.array([self.pheromone[last_city, city] for city in next_city_options])
        distances = np.array([self.distances[last_city, city] for city in next_city_options])
        heuristic = np.power(1.0 / np.maximum(distances, 1e-9), self.beta)
        desirability = np.power(np.maximum(pheromones, 1e-9), self.alpha) * heuristic
        total = desirability.sum()
        if total <= 0:
            return np.ones_like(desirability) / len(desirability)
        return desirability / total

    def optimize(self):
        """Run the MACO algorithm to find an optimal route for the TSP."""
        for _ in range(self.max_iterations):
            route = self.construct_route()
            distance = self.calculate_total_distance(route)
            self.pheromone_update(route, distance)
            self.history.append(distance)
            
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route

        return TSPResult(route=self.best_route, distance=self.best_distance, history=self.history)

    def calculate_total_distance(self, route):
        """Calculate the total distance of a given route."""
        total_distance = 0
        for i in range(self.num_cities):
            total_distance += self.distances[route[i], route[i + 1]]
        return total_distance

    def print_route(self):
        """Print the best route and its distance."""
        print("Best Route:", self.best_route)
        print("Best Distance:", self.best_distance)

# Example usage
if __name__ == "__main__":
    # Define the number of cities
    num_cities = 10
    tsp_maco = TSP_MACO(num_cities=num_cities, alpha=2.0, beta=2.0, rho=0.1, max_iterations=5000)
    
    # Run the MACO algorithm
    result = tsp_maco.optimize()
    
    # Print the best route and its distance
    tsp_maco.print_route()