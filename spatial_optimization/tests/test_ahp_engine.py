"""Unit tests for Analytic Hierarchy Process (AHP) Engine."""

import unittest
import numpy as np
from ahp_engine import AHPEngine, RI_TABLE
from interactive_dashboard import INFRASTRUCTURE_PRESETS


class TestAHPEngine(unittest.TestCase):
    """Test cases for AHP mathematical consistency and eigenvalue computations."""

    def test_default_matrix_initialization(self):
        criteria = ["Criteria 1", "Criteria 2", "Criteria 3", "Criteria 4"]
        engine = AHPEngine(criteria_names=criteria)
        self.assertEqual(engine.comparison_matrix.shape, (4, 4))
        # Diagonal elements must be 1.0
        for i in range(4):
            self.assertAlmostEqual(engine.comparison_matrix[i, i], 1.0)

    def test_weights_sum_to_one(self):
        criteria = ["A", "B", "C", "D"]
        engine = AHPEngine(criteria_names=criteria)
        weights = engine.calculate_weights()
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0.0))

    def test_all_system_presets_achieve_cr_below_threshold(self):
        """Saaty's axiom: CR must be < 0.10 across all production infrastructure presets."""
        for name, preset in INFRASTRUCTURE_PRESETS.items():
            engine = AHPEngine(
                criteria_names=preset["criteria"],
                comparison_matrix=np.array(preset["matrix"])
            )
            weights, metrics = engine.run()
            cr = metrics["cr"]
            self.assertLess(
                cr, 0.10,
                f"Preset '{name}' has CR={cr:.4f} which violates Saaty's 0.10 consistency threshold!"
            )
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)

    def test_invalid_matrix_shape_raises_error(self):
        criteria = ["A", "B", "C"]
        invalid_matrix = np.array([
            [1.0, 2.0],
            [0.5, 1.0]
        ])
        with self.assertRaises(ValueError):
            AHPEngine(criteria_names=criteria, comparison_matrix=invalid_matrix)

    def test_reciprocal_matrix_property(self):
        for name, preset in INFRASTRUCTURE_PRESETS.items():
            mat = np.array(preset["matrix"], dtype=float)
            n = mat.shape[0]
            for i in range(n):
                for j in range(n):
                    # a_ij * a_ji should equal 1.0
                    product = mat[i, j] * mat[j, i]
                    self.assertAlmostEqual(
                        product, 1.0, places=4,
                        msg=f"Reciprocal violation in preset '{name}' at ({i}, {j})"
                    )


if __name__ == "__main__":
    unittest.main()
