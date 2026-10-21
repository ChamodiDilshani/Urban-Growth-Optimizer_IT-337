"""
Continuous Spatial Genetic Algorithm for site selection optimization.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

class SpatialGeneticAlgorithm:
    """
    Continuous Spatial Genetic Algorithm for site selection optimization.
    """

    def __init__(
        self,
        suitability_matrix: np.ndarray,
        elevation: np.ndarray,
        slope: np.ndarray,
        constraint_mask: np.ndarray,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.1,
        mutation_sigma: float = 5.0,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
        slope_threshold: float = 20.0,
        penalty_value: float = -999.0,
        random_seed: int = 42
    ):
        """
        Initialize the Continuous Spatial Genetic Algorithm.

        Args:
            suitability_matrix (np.ndarray): 2D suitability surface S(x,y).
            elevation (np.ndarray): Elevation grid for z-lookup.
            slope (np.ndarray): Slope grid in degrees (raw 0-45 degrees).
            constraint_mask (np.ndarray): Binary constraint mask.
            population_size (int): Size of the population.
            generations (int): Number of generations.
            mutation_rate (float): Probability of mutation.
            mutation_sigma (float): Standard deviation for Gaussian mutation.
            crossover_rate (float): Probability of crossover.
            tournament_size (int): Number of individuals in tournament selection.
            slope_threshold (float): Maximum allowable slope in degrees.
            penalty_value (float): Penalty for constraint violation.
            random_seed (int): Random seed.
        """
        self.suitability_matrix = suitability_matrix
        self.elevation = elevation
        self.slope = slope
        self.constraint_mask = constraint_mask
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.slope_threshold = slope_threshold
        self.penalty_value = penalty_value
        self.random_seed = random_seed
        
        self.grid_size_x = suitability_matrix.shape[0]
        self.grid_size_y = suitability_matrix.shape[1]
        
        np.random.seed(self.random_seed)
        self.fitness_history: List[float] = []

    def _initialize_population(self) -> np.ndarray:
        """
        Initialize population with random uniform values within grid bounds.

        Returns:
            np.ndarray: Initialized population of shape (population_size, 2).
        """
        pop_x = np.random.uniform(0.0, self.grid_size_x - 1, self.population_size)
        pop_y = np.random.uniform(0.0, self.grid_size_y - 1, self.population_size)
        return np.column_stack((pop_x, pop_y))

    def _evaluate_fitness(self, individual: np.ndarray) -> float:
        """
        Evaluate the fitness of a single individual.

        Args:
            individual (np.ndarray): Chromosome representing (x, y) continuous coordinates.

        Returns:
            float: Fitness value of the individual.
        """
        x, y = individual
        ix = int(np.round(x))
        iy = int(np.round(y))
        
        ix = np.clip(ix, 0, self.grid_size_x - 1)
        iy = np.clip(iy, 0, self.grid_size_y - 1)
        
        if self.constraint_mask[ix, iy] == 0:
            return self.penalty_value
            
        if self.slope[ix, iy] > self.slope_threshold:
            return self.penalty_value
            
        return float(self.suitability_matrix[ix, iy])

    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """
        Select an individual from the population using tournament selection.

        Args:
            population (np.ndarray): Current population.
            fitness (np.ndarray): Fitness values of the population.

        Returns:
            np.ndarray: Selected individual.
        """
        indices = np.random.choice(self.population_size, self.tournament_size, replace=False)
        best_idx = indices[np.argmax(fitness[indices])]
        return population[best_idx].copy()

    def _arithmetic_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform arithmetic crossover between two parents.

        Args:
            parent1 (np.ndarray): First parent.
            parent2 (np.ndarray): Second parent.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two children resulting from crossover.
        """
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.uniform(0.0, 1.0)
            child1 = alpha * parent1 + (1.0 - alpha) * parent2
            child2 = (1.0 - alpha) * parent1 + alpha * parent2
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _gaussian_mutation(self, individual: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian mutation to an individual.

        Args:
            individual (np.ndarray): Individual to mutate.

        Returns:
            np.ndarray: Mutated individual.
        """
        if np.random.rand() < self.mutation_rate:
            noise = np.random.normal(0, self.mutation_sigma, size=2)
            individual = individual + noise
            
        individual[0] = np.clip(individual[0], 0, self.grid_size_x - 1)
        individual[1] = np.clip(individual[1], 0, self.grid_size_y - 1)
        return individual

    def optimize(self) -> Dict[str, Any]:
        """
        Run the genetic algorithm optimization loop.

        Returns:
            Dict[str, Any]: Optimization results including best coordinates, 
                            fitness, history, and total generations.
        """
        population = self._initialize_population()
        self.fitness_history = []
        
        best_overall_individual = None
        best_overall_fitness = -np.inf
        
        for generation in range(self.generations):
            fitness = np.array([self._evaluate_fitness(ind) for ind in population])
            
            gen_best_idx = np.argmax(fitness)
            gen_best_fitness = fitness[gen_best_idx]
            
            if gen_best_fitness > best_overall_fitness:
                best_overall_fitness = gen_best_fitness
                best_overall_individual = population[gen_best_idx].copy()
                
            self.fitness_history.append(float(best_overall_fitness))
            
            new_population = [best_overall_individual.copy()]
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness)
                parent2 = self._tournament_selection(population, fitness)
                
                child1, child2 = self._arithmetic_crossover(parent1, parent2)
                
                child1 = self._gaussian_mutation(child1)
                child2 = self._gaussian_mutation(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
                    
            population = np.array(new_population)
            
        best_x, best_y = best_overall_individual
        ix = int(np.round(best_x))
        iy = int(np.round(best_y))
        
        ix = np.clip(ix, 0, self.grid_size_x - 1)
        iy = np.clip(iy, 0, self.grid_size_y - 1)
        best_z = float(self.elevation[ix, iy])
        
        return {
            'best_x': float(best_x),
            'best_y': float(best_y),
            'best_z': best_z,
            'best_fitness': float(best_overall_fitness),
            'fitness_history': self.fitness_history,
            'generations': self.generations
        }

    def get_fitness_history(self) -> List[float]:
        """
        Get the fitness history.

        Returns:
            List[float]: Best fitness per generation.
        """
        return self.fitness_history

if __name__ == '__main__':
    from data_generator import SyntheticRasterGenerator
    from ahp_engine import AHPEngine
    from suitability_surface import SuitabilitySurface

    # Generate synthetic data
    generator = SyntheticRasterGenerator(grid_size=200, random_seed=42)
    layers = generator.generate_all()

    # Raw slope in degrees for constraint checking
    raw_slope = generator.calculate_slope(layers["elevation"])

    # Run AHP
    criteria_names = [
        "Road Proximity", "Slope Safety",
        "Future Growth Velocity", "Existing Amenity Buffer",
    ]
    ahp = AHPEngine(criteria_names=criteria_names)
    weights, _ = ahp.run()

    criteria_rasters = {
        "Road Proximity": layers["road_proximity"],
        "Slope Safety": layers["slope"],
        "Future Growth Velocity": layers["population_growth"],
        "Existing Amenity Buffer": layers["population_growth"] * 0.8,
    }

    surface = SuitabilitySurface(
        criteria_rasters=criteria_rasters,
        weights=weights,
        constraint_mask=layers["constraint_mask"],
        elevation=layers["elevation"],
    )
    suitability = surface.get_suitability_matrix()

    # Run GA
    ga = SpatialGeneticAlgorithm(
        suitability_matrix=suitability,
        elevation=layers["elevation"],
        slope=raw_slope,
        constraint_mask=layers["constraint_mask"],
        population_size=50,
        generations=100,
    )

    results = ga.optimize()

    print("\n--- GA Results ---")
    print(f"Optimal X: {results['best_x']:.2f}")
    print(f"Optimal Y: {results['best_y']:.2f}")
    print(f"Optimal Z (Elevation): {results['best_z']:.2f} m")
    print(f"Best Suitability Score: {results['best_fitness']:.4f}")

    # Plot convergence
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, results["generations"] + 1),
        results["fitness_history"],
        marker="o", markersize=3, linestyle="-", color="r",
    )
    plt.title("GA Convergence History")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("outputs/ga_convergence.png", dpi=200, bbox_inches="tight")
    print("Convergence plot saved to outputs/ga_convergence.png")
