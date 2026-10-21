"""
Continuous Particle Swarm Optimization for Spatial Site Selection.

This module implements a Particle Swarm Optimization (PSO) algorithm adapted
for finding the optimal site on a 2D continuous suitability surface with constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpatialPSO:
    """
    Continuous Particle Swarm Optimization for spatial site selection.
    
    This optimizer searches a 2D suitability landscape to find the global maximum,
    respecting slope constraints and categorical constraint masks.
    
    Attributes:
        suitability_matrix (np.ndarray): 2D array of site suitability scores S(x,y).
        elevation (np.ndarray): 2D array representing elevation.
        slope (np.ndarray): 2D array representing terrain slope in degrees.
        constraint_mask (np.ndarray): 2D binary array where 1=valid, 0=invalid.
        swarm_size (int): Number of particles in the swarm.
        iterations (int): Maximum number of iterations.
        w_start (float): Initial inertia weight.
        w_end (float): Final inertia weight.
        c1 (float): Cognitive acceleration coefficient.
        c2 (float): Social acceleration coefficient.
        v_max_fraction (float): Maximum velocity as a fraction of the grid dimensions.
        slope_threshold (float): Maximum allowable slope in degrees.
        penalty_value (float): Fitness penalty for invalid locations.
        random_seed (int): Seed for reproducibility.
    """

    def __init__(self, 
                 suitability_matrix: np.ndarray, 
                 elevation: np.ndarray, 
                 slope: np.ndarray,
                 constraint_mask: np.ndarray, 
                 swarm_size: int = 50, 
                 iterations: int = 100,
                 w_start: float = 0.9, 
                 w_end: float = 0.4, 
                 c1: float = 1.5, 
                 c2: float = 1.5,
                 v_max_fraction: float = 0.1, 
                 slope_threshold: float = 20.0,
                 penalty_value: float = -999.0, 
                 random_seed: int = 42):
        """Initializes the spatial PSO optimizer with the given environment and parameters."""
        
        # Validation
        if not (suitability_matrix.shape == elevation.shape == slope.shape == constraint_mask.shape):
            raise ValueError("All input grids (suitability, elevation, slope, constraint) must have the same shape.")
            
        self.suitability_matrix = suitability_matrix
        self.elevation = elevation
        self.slope = slope
        self.constraint_mask = constraint_mask
        
        self.grid_shape = self.suitability_matrix.shape
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.slope_threshold = slope_threshold
        self.penalty_value = penalty_value
        
        self.rng = np.random.default_rng(seed=random_seed)
        
        # Calculate maximum velocity based on grid size
        self.v_max = np.array([
            self.grid_shape[0] * v_max_fraction,
            self.grid_shape[1] * v_max_fraction
        ])
        
        # Internal state
        self.positions: np.ndarray = np.zeros((swarm_size, 2))
        self.velocities: np.ndarray = np.zeros((swarm_size, 2))
        self.p_best_positions: np.ndarray = np.zeros((swarm_size, 2))
        self.p_best_fitness: np.ndarray = np.full(swarm_size, self.penalty_value)
        
        self.g_best_position: np.ndarray = np.zeros(2)
        self.g_best_fitness: float = float('-inf')
        
        self.convergence_history: List[float] = []

    def _initialize_swarm(self) -> None:
        """
        Initializes particle positions and velocities, and evaluates initial fitness.
        """
        logger.info("Initializing PSO swarm...")
        
        # Initialize positions randomly uniform in [0, grid_size-1]
        self.positions[:, 0] = self.rng.uniform(0, self.grid_shape[0] - 1, self.swarm_size)
        self.positions[:, 1] = self.rng.uniform(0, self.grid_shape[1] - 1, self.swarm_size)
        
        # Initialize velocities randomly uniform in [-v_max, v_max]
        self.velocities[:, 0] = self.rng.uniform(-self.v_max[0], self.v_max[0], self.swarm_size)
        self.velocities[:, 1] = self.rng.uniform(-self.v_max[1], self.v_max[1], self.swarm_size)
        
        # Evaluate initial fitness
        for i in range(self.swarm_size):
            fitness = self._evaluate_fitness(self.positions[i])
            self.p_best_positions[i] = self.positions[i].copy()
            self.p_best_fitness[i] = fitness
            
            if fitness > self.g_best_fitness:
                self.g_best_fitness = fitness
                self.g_best_position = self.positions[i].copy()
                
        logger.info(f"Swarm initialized. Initial best fitness: {self.g_best_fitness:.4f}")

    def _evaluate_fitness(self, position: np.ndarray) -> float:
        """
        Evaluates the fitness of a given continuous position on the discrete grid.
        
        Args:
            position (np.ndarray): The continuous [x, y] coordinates.
            
        Returns:
            float: Suitability score or penalty if constraints are violated.
        """
        # Convert continuous position to nearest integer grid indices
        ix = int(np.round(position[0]))
        iy = int(np.round(position[1]))
        
        # Clip to valid grid range
        ix = max(0, min(ix, self.grid_shape[0] - 1))
        iy = max(0, min(iy, self.grid_shape[1] - 1))
        
        # Check constraint mask
        if self.constraint_mask[ix, iy] == 0:
            return self.penalty_value
            
        # Check slope constraint
        if self.slope[ix, iy] > self.slope_threshold:
            return self.penalty_value
            
        return float(self.suitability_matrix[ix, iy])

    def _update_velocity(self, i: int, w: float) -> np.ndarray:
        """
        Updates and returns the velocity for the i-th particle.
        
        Args:
            i (int): Index of the particle.
            w (float): Current inertia weight.
            
        Returns:
            np.ndarray: The new velocity vector.
        """
        r1 = self.rng.uniform(0, 1, 2)
        r2 = self.rng.uniform(0, 1, 2)
        
        cognitive_comp = self.c1 * r1 * (self.p_best_positions[i] - self.positions[i])
        social_comp = self.c2 * r2 * (self.g_best_position - self.positions[i])
        
        v_new = w * self.velocities[i] + cognitive_comp + social_comp
        
        # Clamp velocity to [-v_max, v_max]
        v_new[0] = np.clip(v_new[0], -self.v_max[0], self.v_max[0])
        v_new[1] = np.clip(v_new[1], -self.v_max[1], self.v_max[1])
        
        return v_new

    def _update_position(self, i: int) -> np.ndarray:
        """
        Updates and returns the position for the i-th particle with boundary reflection.
        
        Args:
            i (int): Index of the particle.
            
        Returns:
            np.ndarray: The new position vector.
        """
        new_pos = self.positions[i] + self.velocities[i]
        
        # X boundary reflection
        if new_pos[0] < 0:
            new_pos[0] = -new_pos[0]
            self.velocities[i, 0] *= -1.0
        elif new_pos[0] > self.grid_shape[0] - 1:
            new_pos[0] = 2 * (self.grid_shape[0] - 1) - new_pos[0]
            self.velocities[i, 0] *= -1.0
            
        # Y boundary reflection
        if new_pos[1] < 0:
            new_pos[1] = -new_pos[1]
            self.velocities[i, 1] *= -1.0
        elif new_pos[1] > self.grid_shape[1] - 1:
            new_pos[1] = 2 * (self.grid_shape[1] - 1) - new_pos[1]
            self.velocities[i, 1] *= -1.0
            
        # Final safety clip
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_shape[0] - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_shape[1] - 1)
        
        return new_pos

    def optimize(self) -> Dict[str, Any]:
        """
        Executes the main PSO optimization loop.
        
        Returns:
            Dict[str, Any]: Optimization results including best position, fitness, 
                            elevation, and convergence history.
        """
        self._initialize_swarm()
        self.convergence_history = []
        
        logger.info(f"Starting optimization for {self.iterations} iterations...")
        
        for iter_idx in range(self.iterations):
            # Calculate linearly decreasing inertia weight
            w = self.w_start - ((self.w_start - self.w_end) * (iter_idx / self.iterations))
            
            for i in range(self.swarm_size):
                # Update velocity and position
                self.velocities[i] = self._update_velocity(i, w)
                self.positions[i] = self._update_position(i)
                
                # Evaluate new fitness
                fitness = self._evaluate_fitness(self.positions[i])
                
                # Update personal best
                if fitness > self.p_best_fitness[i]:
                    self.p_best_fitness[i] = fitness
                    self.p_best_positions[i] = self.positions[i].copy()
                    
                    # Update global best
                    if fitness > self.g_best_fitness:
                        self.g_best_fitness = fitness
                        self.g_best_position = self.positions[i].copy()
                        
            # Record global best for convergence tracking
            self.convergence_history.append(self.g_best_fitness)
            
            if (iter_idx + 1) % 10 == 0:
                logger.info(f"Iteration {iter_idx + 1}/{self.iterations} - Best Fitness: {self.g_best_fitness:.4f}")
                
        logger.info("Optimization complete.")
        
        # Get optimal integers for elevation lookup
        opt_ix = int(np.round(self.g_best_position[0]))
        opt_iy = int(np.round(self.g_best_position[1]))
        
        opt_ix = np.clip(opt_ix, 0, self.grid_shape[0] - 1)
        opt_iy = np.clip(opt_iy, 0, self.grid_shape[1] - 1)
        best_z = float(self.elevation[opt_ix, opt_iy])
        
        return {
            'best_x': float(self.g_best_position[0]),
            'best_y': float(self.g_best_position[1]),
            'best_z': best_z,
            'best_fitness': self.g_best_fitness,
            'convergence_history': self.convergence_history,
            'iterations': self.iterations
        }

    def get_convergence_history(self) -> List[float]:
        """
        Retrieves the convergence history.
        
        Returns:
            List[float]: The history of the global best fitness values per iteration.
        """
        return self.convergence_history


if __name__ == '__main__':
    from data_generator import SyntheticRasterGenerator
    from ahp_engine import AHPEngine
    from suitability_surface import SuitabilitySurface

    logger.info("Generating synthetic spatial data...")
    generator = SyntheticRasterGenerator(grid_size=200, random_seed=42)
    layers = generator.generate_all()

    # Raw slope in degrees for constraint checking
    raw_slope = generator.calculate_slope(layers["elevation"])

    logger.info("Running AHP...")
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

    logger.info("Running Spatial PSO...")
    pso = SpatialPSO(
        suitability_matrix=suitability,
        elevation=layers["elevation"],
        slope=raw_slope,
        constraint_mask=layers["constraint_mask"],
        swarm_size=50,
        iterations=100,
    )

    results = pso.optimize()

    print("\n--- PSO Results ---")
    print(f"Optimal X: {results['best_x']:.2f}")
    print(f"Optimal Y: {results['best_y']:.2f}")
    print(f"Optimal Z (Elevation): {results['best_z']:.2f} m")
    print(f"Best Suitability Score: {results['best_fitness']:.4f}")

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, results["iterations"] + 1), results["convergence_history"],
             marker="o", markersize=3, linestyle="-", color="b")
    plt.title("PSO Convergence History")
    plt.xlabel("Iteration")
    plt.ylabel("Global Best Fitness")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig("outputs/pso_convergence.png", dpi=200, bbox_inches="tight")
    logger.info("Convergence plot saved to outputs/pso_convergence.png")
