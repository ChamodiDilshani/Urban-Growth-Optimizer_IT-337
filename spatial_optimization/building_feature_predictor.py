"""
Building Feature & Architectural Sizing Predictor for Infrastructure Siting.

Predicts recommended building parameters for optimal infrastructure sites
based on surrounding geospatial demographics (WorldPop), terrain slope,
and urban planning standards (UDA, Sri Lanka MoE, WHO).

Predicted Features:
    1. Population Catchment & Service Demand (3km / 5km radius)
    2. Recommended Facility Capacity (Students / Hospital Beds / Daily Footfall)
    3. Built-up Area & Footprint (sq. meters and sq. feet)
    4. Recommended Number of Floors (Stories based on terrain gradient)
    5. Solar PV Energy Generation Potential (kWp and kWh/year)
    6. Foundation Engineering Type & Construction Cost Index
"""

from typing import Dict, Any, Tuple
import numpy as np

# Spatial resolution of our 200x200 grid across Matale (approx 250m per cell)
METERS_PER_CELL = 250.0

# Sri Lankan Baseline Architectural & Engineering Constants
SRI_LANKA_SOLAR_GHI_KWH_M2_YEAR = 1825.0  # ~5.0 kWh/m2/day baseline in Matale region
SOLAR_PV_EFFICIENCY = 0.18  # 18% panel efficiency
SOLAR_PERFORMANCE_RATIO = 0.75  # System losses factor
CO2_OFFSET_KG_PER_KWH = 0.70  # Clean energy carbon offset factor

_SOLAR_GRID_CACHE = None
_BUILDING_DENSITY_CACHE = None


def _get_solar_grid():
    global _SOLAR_GRID_CACHE
    if _SOLAR_GRID_CACHE is None:
        import os
        path = os.path.join("data", "matale_solar_radiation.npy")
        if os.path.exists(path):
            _SOLAR_GRID_CACHE = np.load(path)
    return _SOLAR_GRID_CACHE


def _get_building_density():
    global _BUILDING_DENSITY_CACHE
    if _BUILDING_DENSITY_CACHE is None:
        import os
        path = os.path.join("data", "matale_building_density.npy")
        if os.path.exists(path):
            _BUILDING_DENSITY_CACHE = np.load(path)
    return _BUILDING_DENSITY_CACHE


class BuildingFeaturePredictor:
    """Predicts building specifications for selected infrastructure sites."""

    @staticmethod
    def calculate_catchment_population(
        population_grid: np.ndarray,
        center_row: int,
        center_col: int,
        radius_km: float = 3.0
    ) -> float:
        """
        Calculate total estimated population living within a given radius (km).
        """
        grid_h, grid_w = population_grid.shape
        radius_cells = radius_km * 1000.0 / METERS_PER_CELL
        
        y_indices, x_indices = np.ogrid[:grid_h, :grid_w]
        dist_sq = (y_indices - center_row)**2 + (x_indices - center_col)**2
        mask = dist_sq <= radius_cells**2
        
        # WorldPop 100m raster is calibrated in people per cell
        # Sum population within buffer
        catchment_pop = float(np.sum(population_grid[mask]))
        return max(catchment_pop, 250.0)

    @classmethod
    def predict_features(
        cls,
        infra_type: str,
        best_row: float,
        best_col: float,
        elevation_val: float,
        slope_deg: float,
        population_grid: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive building architectural & capacity predictions.
        """
        r_idx = int(np.clip(np.round(best_row), 0, population_grid.shape[0] - 1))
        c_idx = int(np.clip(np.round(best_col), 0, population_grid.shape[1] - 1))
        
        # 1. Demographic Catchment (2.5km, 3km, 5km, and 10km regional)
        pop_2_5km = cls.calculate_catchment_population(population_grid, r_idx, c_idx, radius_km=2.5)
        pop_3km = cls.calculate_catchment_population(population_grid, r_idx, c_idx, radius_km=3.0)
        pop_5km = cls.calculate_catchment_population(population_grid, r_idx, c_idx, radius_km=5.0)
        pop_10km = cls.calculate_catchment_population(population_grid, r_idx, c_idx, radius_km=10.0)

        # Scale raw proxy population to approximate real-world Matale density
        calibrated_pop_2_5km = int(max(pop_2_5km * 12.5, 3200))
        calibrated_pop_3km = int(max(pop_3km * 12.5, 4500))
        calibrated_pop_5km = int(max(pop_5km * 12.5, 9500))
        calibrated_pop_10km = int(max(pop_10km * 12.5, 28000))

        # 2. Infrastructure-Specific Sizing Predictions
        base_rate_per_m2_lkr = 135000.0

        if infra_type in ("school", "national_school"):
            capacity = int(np.clip(calibrated_pop_10km * 0.185 * 0.35, 2200, 4200))
            capacity_label = "Students (1AB National School Enrollment)"
            capacity_unit = "Students"
            built_up_area_m2 = capacity * 5.5
            unit_subdivisions = int(np.ceil(capacity / 40))
            unit_subdivision_label = f"{unit_subdivisions} Classrooms + 6 Science Labs + 3 IT Labs + 1,200-seat Auditorium"
            land_requirement = "8.0 - 15.0 Acres (UDA 1AB Standard)"
            sports_facilities = "Standard 400m 8-lane Running Track, Cricket/Football Ground, Indoor Gymnasium"
            traffic_guidance = "Internal 2-lane drop-off ring road + 8 dedicated school bus bays"
            base_rate_per_m2_lkr = 145000.0
            max_floors_cap = 4

        elif infra_type == "primary_school":
            capacity = int(np.clip(calibrated_pop_2_5km * 0.075 * 0.60, 350, 750))
            capacity_label = "Students (Primary Gr 1-5 Enrollment)"
            capacity_unit = "Students"
            built_up_area_m2 = capacity * 4.5
            unit_subdivisions = int(np.ceil(capacity / 35))
            unit_subdivision_label = f"{unit_subdivisions} Classrooms + Activity Center + Sick Room + Library"
            land_requirement = "1.5 - 2.5 Acres (Safe Child Buffer)"
            sports_facilities = "Junior Soft-turf Playground & Mini Athletics Area"
            traffic_guidance = "Frontage pedestrian zebra crossing buffer + School van pick-up lane"
            base_rate_per_m2_lkr = 125000.0
            max_floors_cap = 2  # Max 2 floors for young child safety!

        elif infra_type == "secondary_school":
            capacity = int(np.clip(calibrated_pop_5km * 0.14 * 0.40, 1000, 1800))
            capacity_label = "Students (Provincial 1C Gr 6-13 Enrollment)"
            capacity_unit = "Students"
            built_up_area_m2 = capacity * 5.0
            unit_subdivisions = int(np.ceil(capacity / 35))
            unit_subdivision_label = f"{unit_subdivisions} Classrooms + 3 Science Labs + 2 IT Labs + Assembly Hall"
            land_requirement = "3.5 - 5.5 Acres"
            sports_facilities = "200m Running Track, Volleyball & Netball Courts"
            traffic_guidance = "School van parking bay & public bus drop-off link"
            base_rate_per_m2_lkr = 135000.0
            max_floors_cap = 3

        elif infra_type == "girls_school":
            capacity = int(np.clip(calibrated_pop_5km * 0.185 * 0.50 * 0.70, 1500, 2800))
            capacity_label = "Students (Girls' High School Enrollment)"
            capacity_unit = "Students"
            built_up_area_m2 = capacity * 5.2
            unit_subdivisions = int(np.ceil(capacity / 35))
            unit_subdivision_label = f"{unit_subdivisions} Classrooms + Home Science Lab + Bio/Chem Labs + Indoor Court"
            land_requirement = "5.0 - 8.0 Acres (High Security Perimeter Wall)"
            sports_facilities = "Netball & Basketball Courts, 200m Track, Indoor Gymnasium"
            traffic_guidance = "Gated security drop-off loop & van holding bay"
            base_rate_per_m2_lkr = 140000.0
            max_floors_cap = 3

        elif infra_type == "boys_school":
            capacity = int(np.clip(calibrated_pop_5km * 0.185 * 0.50 * 0.70, 1500, 2800))
            capacity_label = "Students (Boys' College Enrollment)"
            capacity_unit = "Students"
            built_up_area_m2 = capacity * 5.2
            unit_subdivisions = int(np.ceil(capacity / 35))
            unit_subdivision_label = f"{unit_subdivisions} Classrooms + Tech Workshop + Science Labs + Cadet Ground"
            land_requirement = "6.0 - 10.0 Acres"
            sports_facilities = "Rugby/Cricket Pitch, Pavilion, 400m Track, Gymnasium"
            traffic_guidance = "Dedicated bus drop-off zone & bicycle parking bays"
            base_rate_per_m2_lkr = 140000.0
            max_floors_cap = 3

        elif infra_type == "hospital":
            capacity = int(np.clip(calibrated_pop_5km * 0.0035, 40, 300))
            capacity_label = "Inpatient Beds (Inpatient Bed Capacity)"
            capacity_unit = "Beds"
            built_up_area_m2 = capacity * 55.0
            wards_count = int(np.ceil(capacity / 25))
            unit_subdivision_label = f"{wards_count} Clinical Wards + Emergency and OPD Unit"
            land_requirement = "3.0 - 6.0 Acres (Emergency Access Buffer)"
            sports_facilities = "Physiotherapy & Rehabilitation Garden"
            traffic_guidance = "Dedicated 24/7 Ambulance Emergency Corridor & Visitor Parking"
            base_rate_per_m2_lkr = 165000.0
            max_floors_cap = 4

        elif infra_type == "transport_hub":
            capacity = int(np.clip(calibrated_pop_5km * 0.15, 1200, 8000))
            capacity_label = "Daily Commuter Capacity"
            capacity_unit = "Passengers/Day"
            built_up_area_m2 = float(np.clip(capacity * 0.8, 1200, 4500))
            unit_subdivision_label = f"{int(np.clip(capacity // 400, 6, 20))} Bus Bays + Ticket Terminal"
            land_requirement = "4.0 - 7.0 Acres (Multi-modal Turnaround Radius)"
            sports_facilities = "Driver Rest Area & Commercial Kiosks"
            traffic_guidance = "Direct access to A-Grade arterial highway without bottlenecking"
            base_rate_per_m2_lkr = 130000.0
            max_floors_cap = 3

        else:  # Market / Commercial Center
            capacity = int(np.clip(calibrated_pop_3km * 0.20, 800, 5000))
            capacity_label = "Daily Footfall Capacity"
            capacity_unit = "Shoppers/Day"
            built_up_area_m2 = float(np.clip(capacity * 1.2, 1000, 4000))
            stalls_count = int(np.clip(built_up_area_m2 / 25, 30, 150))
            unit_subdivision_label = f"{stalls_count} Commercial Retail Stalls + Storage"
            land_requirement = "2.5 - 4.5 Acres (Customer & Freight Parking)"
            sports_facilities = "Public Plaza & Pedestrian Promenade"
            traffic_guidance = "Separate heavy goods loading bay from customer parking"
            base_rate_per_m2_lkr = 130000.0
            max_floors_cap = 3

        # 3. Floor Count & Ground Footprint (Optimized for terrain slope & facility safety cap)
        if slope_deg > 14.0:
            recommended_floors = min(max_floors_cap, 4)
            foundation_type = "Stepped Reinforced Retaining Foundation (Steep Terrain)"
            slope_cost_multiplier = 1.35
        elif slope_deg > 8.0:
            recommended_floors = min(max_floors_cap, 3)
            foundation_type = "Semi-Stepped Pad Footing with Ground Beams (Moderate Slope)"
            slope_cost_multiplier = 1.18
        else:
            recommended_floors = min(max_floors_cap, 2)
            foundation_type = "Standard Isolated Strip / Pad Footing (Flat Terrain)"
            slope_cost_multiplier = 1.02

        ground_footprint_m2 = built_up_area_m2 / recommended_floors
        built_up_area_sqft = built_up_area_m2 * 10.7639
        ground_footprint_sqft = ground_footprint_m2 * 10.7639

        # 4. Solar Rooftop Energy Generation Potential (Calibrated with NASA POWER Solar Grid)
        solar_grid = _get_solar_grid()
        if solar_grid is not None and 0 <= r_idx < solar_grid.shape[0] and 0 <= c_idx < solar_grid.shape[1]:
            local_ghi = float(solar_grid[r_idx, c_idx])
        else:
            local_ghi = SRI_LANKA_SOLAR_GHI_KWH_M2_YEAR

        b_density_grid = _get_building_density()
        if b_density_grid is not None and 0 <= r_idx < b_density_grid.shape[0] and 0 <= c_idx < b_density_grid.shape[1]:
            neighborhood_buildings = int(b_density_grid[r_idx, c_idx])
        else:
            neighborhood_buildings = 0

        # Usable roof area ~ 70% of ground footprint
        usable_roof_m2 = ground_footprint_m2 * 0.70
        # Peak Solar Power (kWp) = Area * 1000 W/m2 * efficiency
        solar_capacity_kwp = round(usable_roof_m2 * SOLAR_PV_EFFICIENCY, 1)
        # Annual Solar Energy (kWh/year) = kWp * Annual Sun Hours * PR
        annual_solar_kwh = int(solar_capacity_kwp * (local_ghi / 1000.0) * SOLAR_PERFORMANCE_RATIO * 1000)
        annual_co2_offset_tons = round(annual_solar_kwh * CO2_OFFSET_KG_PER_KWH / 1000.0, 1)

        # 5. Estimated Construction Budget & Cost Index
        total_estimated_cost_lkr = built_up_area_m2 * base_rate_per_m2_lkr * slope_cost_multiplier
        total_estimated_cost_million_lkr = round(total_estimated_cost_lkr / 1_000_000.0, 1)

        return {
            "facility_type": infra_type,
            "slope_deg": round(slope_deg, 1),
            "elevation_m": round(elevation_val, 1),
            "neighborhood_buildings": neighborhood_buildings,
            "empirical_solar_ghi_kwh_m2_yr": round(local_ghi, 1),
            "catchment": {
                "population_2_5km": calibrated_pop_2_5km,
                "population_3km": calibrated_pop_3km,
                "population_5km": calibrated_pop_5km,
                "population_10km": calibrated_pop_10km,
            },
            "campus_specs": {
                "land_requirement": land_requirement,
                "sports_facilities": sports_facilities,
                "traffic_guidance": traffic_guidance,
            },
            "capacity": {
                "value": capacity,
                "label": capacity_label,
                "unit": capacity_unit,
                "subdivisions": unit_subdivision_label,
            },
            "architectural": {
                "total_built_up_m2": round(built_up_area_m2, 0),
                "total_built_up_sqft": int(built_up_area_sqft),
                "ground_footprint_m2": round(ground_footprint_m2, 0),
                "ground_footprint_sqft": int(ground_footprint_sqft),
                "recommended_floors": recommended_floors,
                "foundation_type": foundation_type,
            },
            "solar_energy": {
                "solar_capacity_kwp": solar_capacity_kwp,
                "annual_solar_kwh": annual_solar_kwh,
                "annual_co2_offset_tons": annual_co2_offset_tons,
                "monthly_kwh": int(annual_solar_kwh / 12),
                "solar_ghi_kwh_m2": round(local_ghi, 1),
            },
            "construction_cost": {
                "cost_multiplier": round(slope_cost_multiplier, 2),
                "cost_tier": "Standard" if slope_cost_multiplier < 1.10 else ("Moderate" if slope_cost_multiplier < 1.25 else "High Foundation Terrain"),
                "estimated_budget_million_lkr": total_estimated_cost_million_lkr,
                "estimated_budget_formatted": f"LKR {total_estimated_cost_million_lkr:,.1f} Million",
            }
        }


# Quick test
if __name__ == "__main__":
    from real_data_loader import RealDataLoader
    loader = RealDataLoader(data_dir="data", grid_size=200, random_seed=42)
    layers = loader.generate_all()
    
    # Test for School at Matale Town (Row 92, Col 97)
    specs = BuildingFeaturePredictor.predict_features(
        infra_type="school",
        best_row=92.0,
        best_col=97.0,
        elevation_val=470.0,
        slope_deg=6.5,
        population_grid=layers["population_growth"],
    )
    
    print("\n" + "=" * 60)
    print("  PREDICTED BUILDING SPECIFICATIONS (SCHOOL)")
    print("=" * 60)
    print(f"  Catchment Population (3km): {specs['catchment']['population_3km']:,}")
    print(f"  Facility Capacity:          {specs['capacity']['value']:,} {specs['capacity']['unit']}")
    print(f"  Subdivisions:               {specs['capacity']['subdivisions']}")
    print(f"  Total Built-up Floor Area:  {specs['architectural']['total_built_up_m2']:,.0f} m2 ({specs['architectural']['total_built_up_sqft']:,} sq.ft)")
    print(f"  Recommended Floors:         {specs['architectural']['recommended_floors']} Stories")
    print(f"  Ground Footprint:           {specs['architectural']['ground_footprint_m2']:,.0f} m2")
    print(f"  Foundation Engineering:     {specs['architectural']['foundation_type']}")
    print(f"  Solar PV Capacity:          {specs['solar_energy']['solar_capacity_kwp']} kWp")
    print(f"  Annual Solar Generation:    {specs['solar_energy']['annual_solar_kwh']:,} kWh/year")
    print(f"  Est. Construction Budget:   {specs['construction_cost']['estimated_budget_formatted']}")
    print("=" * 60)
