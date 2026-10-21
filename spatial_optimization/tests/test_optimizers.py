"""Unit tests for Spatial Genetic Algorithm and Particle Swarm Optimization."""

import unittest
import numpy as np
from genetic_algorithm import SpatialGeneticAlgorithm
from pso_optimizer import SpatialPSO


class TestSpatialOptimizers(unittest.TestCase):
    """Test convergence, bounds safety, and constraint adherence of GA and PSO."""

    def setUp(self):
        # 50x50 synthetic test landscape
        self.size = 50
        self.suitability = np.ones((self.size, self.size), dtype=float) * 0.5
        # Peak at center (25, 25)
        self.suitability[25, 25] = 0.95
        self.elevation = np.linspace(100, 500, self.size * self.size).reshape(self.size, self.size)
        self.slope = np.ones((self.size, self.size), dtype=float) * 5.0  # gentle
        self.mask = np.ones((self.size, self.size), dtype=float)

    def test_pso_runs_and_improves_fitness(self):
        pso = SpatialPSO(
            suitability_matrix=self.suitability,
            elevation=self.elevation,
            slope=self.slope,
            constraint_mask=self.mask,
            swarm_size=20,
            iterations=25,
            random_seed=42
        )
        res = pso.optimize()
        self.assertIn("best_x", res)
        self.assertIn("best_y", res)
        self.assertIn("best_fitness", res)
        # Best fitness should be >= 0.5 (initial plateau)
        self.assertGreaterEqual(res["best_fitness"], 0.5)
        # Position must be within bounds
        self.assertTrue(0 <= res["best_x"] < self.size)
        self.assertTrue(0 <= res["best_y"] < self.size)

    def test_ga_runs_and_bounds_safety(self):
        ga = SpatialGeneticAlgorithm(
            suitability_matrix=self.suitability,
            elevation=self.elevation,
            slope=self.slope,
            constraint_mask=self.mask,
            population_size=20,
            generations=25,
            random_seed=42
        )
        res = ga.optimize()
        self.assertIn("best_x", res)
        self.assertIn("best_y", res)
        self.assertIn("best_fitness", res)
        self.assertTrue(0 <= res["best_x"] < self.size)
        self.assertTrue(0 <= res["best_y"] < self.size)

    def test_steep_slope_constraint_penalty(self):
        # Set steep slope on the peak (25, 25)
        steep_slope = self.slope.copy()
        steep_slope[25, 25] = 35.0  # > 20 NBRO limit

        pso = SpatialPSO(
            suitability_matrix=self.suitability,
            elevation=self.elevation,
            slope=steep_slope,
            constraint_mask=self.mask,
            slope_threshold=20.0,
            swarm_size=20,
            iterations=20,
            random_seed=42
        )
        # Fitness evaluated directly at steep point must be penalized
        val = pso._evaluate_fitness([25.0, 25.0])
        self.assertEqual(val, pso.penalty_value)

    def test_constraint_mask_zero_penalty(self):
        # Mask out peak
        mask_with_hole = self.mask.copy()
        mask_with_hole[25, 25] = 0.0  # Restricted conservation area

        ga = SpatialGeneticAlgorithm(
            suitability_matrix=self.suitability,
            elevation=self.elevation,
            slope=self.slope,
            constraint_mask=mask_with_hole,
            population_size=15,
            generations=15,
            random_seed=42
        )
        fit = ga._evaluate_fitness(np.array([25.0, 25.0]))
        self.assertEqual(fit, ga.penalty_value)


if __name__ == "__main__":
    unittest.main()
