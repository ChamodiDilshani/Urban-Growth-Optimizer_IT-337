"""Unit tests for Building Feature & Architectural Sizing Predictor."""

import unittest
import numpy as np
from building_feature_predictor import BuildingFeaturePredictor


class TestBuildingFeaturePredictor(unittest.TestCase):
    """Test architectural sizing, MoE pupil capacities, CIDA multipliers, and solar outputs."""

    def setUp(self):
        self.pop_grid = np.ones((200, 200), dtype=float) * 5.0

    def test_school_capacity_and_space_standards(self):
        specs = BuildingFeaturePredictor.predict_features(
            infra_type="national_school",
            best_row=100.0,
            best_col=100.0,
            elevation_val=350.0,
            slope_deg=4.0,  # flat
            population_grid=self.pop_grid
        )
        self.assertIn("capacity", specs)
        self.assertIn("architectural", specs)
        self.assertIn("construction_cost", specs)

        # 1AB school student capacity must be within reasonable national school range
        cap = specs["capacity"]["value"]
        self.assertGreaterEqual(cap, 2000)
        self.assertLessEqual(cap, 5000)

        # Space per student must adhere to ~5.5 m2
        built_area = specs["architectural"]["total_built_up_m2"]
        self.assertAlmostEqual(built_area / cap, 5.5, delta=0.5)

    def test_slope_adapts_foundation_and_floors(self):
        # Flat terrain
        flat_specs = BuildingFeaturePredictor.predict_features(
            infra_type="school",
            best_row=100.0,
            best_col=100.0,
            elevation_val=200.0,
            slope_deg=3.0,
            population_grid=self.pop_grid
        )
        # Steep terrain
        steep_specs = BuildingFeaturePredictor.predict_features(
            infra_type="school",
            best_row=100.0,
            best_col=100.0,
            elevation_val=200.0,
            slope_deg=22.0,
            population_grid=self.pop_grid
        )

        # Steep terrain must have higher cost multiplier
        self.assertGreater(
            steep_specs["construction_cost"]["cost_multiplier"],
            flat_specs["construction_cost"]["cost_multiplier"]
        )
        # Steep terrain foundation should mention retaining or piles
        self.assertIn("Retaining", steep_specs["architectural"]["foundation_type"])

    def test_solar_pv_generation_is_positive(self):
        specs = BuildingFeaturePredictor.predict_features(
            infra_type="hospital",
            best_row=100.0,
            best_col=100.0,
            elevation_val=250.0,
            slope_deg=5.0,
            population_grid=self.pop_grid
        )
        solar = specs["solar_energy"]
        self.assertGreater(solar["solar_capacity_kwp"], 0.0)
        self.assertGreater(solar["annual_solar_kwh"], 0)
        self.assertGreater(solar["annual_co2_offset_tons"], 0.0)


if __name__ == "__main__":
    unittest.main()
