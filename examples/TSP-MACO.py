To provide a full, working code implementation based on the provided explanation of the MACO framework, along with the predefined structure and function sketches, we would need to integrate the MACO algorithm with a specific problem domain. In your description, there are indications of how MACO could be applied to various domains like the Zombie Swarm Simulation (ZVSS) using Pygame. However, for a collectible and fully working code, choosing a single domain for implementation is necessary.

Let's focus on implementing a simplified version of MACO for solving the Traveling Salesman Problem (TSP), a well-known combinatorial optimization problem, as it's described in the "Examples" section of your document. The TSP involves finding the shortest possible route that visits each city (node) exactly once and returns to the origin city, which makes it an excellent candidate for illustrating the MACO methodology.

Here's a simplified implementation that will not cover all the DRY (Don't Repeat Yourself) principles or every detail from the MACOFramework to keep the example concise but functional:

```python
import numpy as np
import random
import time

class TSP_MACO:
    def __init__(self, num_cities, alpha=1.0, beta=0.5, rho=0.8, max_iterations=1000):
        self.num_cities = num_cities
        self.alpha = alpha  # Influence of pheromone on the movement of ants
        self.beta = beta  # Influence of heuristic (distance to the next city) on the movement of ants
        self.rho = rho  # Pheromone evaporation rate
        self.max_iterations = max_iterations
        self.pheromone = np.zeros((num_cities, num_cities))  # Pheromone matrix
        self.distances = self.calculate_distances()
        self.best_route = None
        self.best_distance = float('inf')

    def calculate_distances(self):
        """Calculate the Euclidean distances between all pairs of cities."""
        distances = np.zeros((self.num_cities, self.num_cities))
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if i != j:
                    distances[i, j] = np.sqrt((np.random.rand() * 10 - 5) ** 2 + (np.random.rand() * 10 - 5) ** 2)
        return distances

    def pheromone_update(self, route, distance):
        """Update the pheromone matrix based on the quality of the solution (route)."""
        for i in range(self.num_cities):
            for j in range(self.num_cities):
                if j == route[i]:
                    self.pheromone[i, j] += distance
        self.pheromone *= self.rho  # Evaporation

    def construct_route(self):
        """Construct a new route using the pheromone matrix."""
        route = [0]  # Start at city 0
        unvisited_cities = set(range(1, self.num_cities))
        while unvisited_cities:
            current_city = route[-1]
            next_city_options = list(unvisited_cities)
            if next_city_options:
                # Use a weighted random choice based on the pheromone levels and distance heuristic
                next_city = np.random.choice(next_city_options, p=self.probability(route, next_city_options))
                route.append(next_city)
                unvisited_cities.remove(next_city)
        route.append(0)  # Return to the starting city
        return np.array(route)

    def probability(self, route, next_city_options):
        """Calculate the probabilities for choosing the next city based on pheromone and heuristic."""
        probabilities = []
        for next_city in next_city_options:
            pheromone_level = self.pheromone[route[-1], next_city]
            heuristic = 1.0 / self.distances[route[-1], next_city]  # Smaller distance = higher heuristic
            probabilities.append(self.alpha * pheromone_level + self.beta * heuristic)
        return np.array(probabilities) / np.sum(probabilities)

    def optimize(self):
        """Run the MACO algorithm to find an optimal route for the TSP."""
        for _ in range(self.max_iterations):
            route = self.construct_route()
            distance = self.calculate_total_distance(route)
            self.pheromone_update(route, distance)
            
            if distance < self.best_distance:
                self.best_distance = distance
                self.best_route = route

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
    tsp_maco = TSP_MACO(num_cities=num_cities, alpha=2.0, beta=0.5, rho=0.5, max_iterations=5000)
    
    # Run the MACO algorithm
    tsp_maco.optimize()
    
    # Print the best route and its distance
    tsp_maco.print_route()
