"""
Interactive Spatial Optimization Dashboard with Google Maps / Satellite Map Integration.

Enables planners to:
    1. Select infrastructure type (School, Hospital, Transport Hub, Market)
    2. View REAL Matale District on Interactive Leaflet Map (Satellite / Street / Dark)
    3. Toggle 118 Existing Schools & 43 Hospitals as pins on the real map
    4. Select ANY sub-region / square / bounding box by clicking presets OR drawing directly on the map
    5. Run GA + PSO to find the localized optimum site
    6. Overlay the AHP Suitability heatmap directly onto Google/Satellite map with opacity control
    7. One-click "Open in Google Maps" button for navigation and street verification!

Usage:
    python interactive_dashboard.py
    Open http://localhost:8050
"""

import http.server
import socketserver
import json
import base64
import os
import io
import time
import webbrowser
import threading
import urllib.parse
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from PIL import Image

import geopandas as gpd
from real_data_loader import RealDataLoader, MATALE_BOUNDS
from ahp_engine import AHPEngine
from suitability_surface import SuitabilitySurface
from genetic_algorithm import SpatialGeneticAlgorithm
from pso_optimizer import SpatialPSO
from building_feature_predictor import BuildingFeaturePredictor

PORT = 8050
DATA_DIR = "data"
GRID_SIZE = 200

# 8 criteria names
CRITERIA_8 = [
    "Road Proximity",
    "Slope Safety",
    "Population Density",
    "School Distance",
    "Healthcare Proximity",
    "Land Use Suitability",
    "Transport Access",
    "Amenity Buffer",
]

# Map criteria name -> data layer key
CRITERIA_LAYER_MAP = {
    "Road Proximity": "road_proximity",
    "Slope Safety": "slope",
    "Population Density": "population_growth",
    "School Distance": "school_distance",
    "Healthcare Proximity": "healthcare_proximity",
    "Land Use Suitability": "landuse_suitability",
    "Transport Access": "transport_access",
    "Amenity Buffer": None,
}

# Pre-configured 8x8 AHP matrices for infrastructure types
INFRASTRUCTURE_PRESETS = {
    "school": {
        "label": "School (Pasala)",
        "emoji": "🏫",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     3,     3,     3,      5,      3,      1,     5],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1/5,   1/3,   1/3,   1/3,    1,      1/3,    1/5,   1],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1,     3,     3,     3,      5,      3,      1,     5],
            [1/5,   1/3,   1/3,   1/3,    1,      1/3,    1/5,   1],
        ],
        "description": "Safe road access, high transit connectivity, balanced demographic catchment",
    },
    "national_school": {
        "label": "National School / Central College (Type 1AB)",
        "emoji": "⭐",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     3,     3,     3,      5,      3,      1,     5],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1/5,   1/3,   1/3,   1/3,    1,      1/3,    1/5,   1],
            [1/3,   1,     1,     1,      3,      1,      1/3,   3],
            [1,     3,     3,     3,      5,      3,      1,     5],
            [1/5,   1/3,   1/3,   1/3,    1,      1/3,    1/5,   1],
        ],
        "description": "District-wide catchment (10km), priority on major transport corridors, avoids nearby National Schools, large campus extent",
    },
    "primary_school": {
        "label": "Primary School (Grade 1-5)",
        "emoji": "🧒",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     1/3,   1/3,   3,      3,      1,      1,     3],
            [3,     1,     1,     5,      5,      3,      3,     5],
            [3,     1,     1,     5,      5,      3,      3,     5],
            [1/3,   1/5,   1/5,   1,      1,      1/3,    1/3,   1],
            [1/3,   1/5,   1/5,   1,      1,      1/3,    1/3,   1],
            [1,     1/3,   1/3,   3,      3,      1,      1,     3],
            [1,     1/3,   1/3,   3,      3,      1,      1,     3],
            [1/3,   1/5,   1/5,   1,      1,      1/3,    1/3,   1],
        ],
        "description": "Neighborhood catchment (1.5-2.5km), high priority on slope safety & residential density, avoids fast highway corridors",
    },
    "secondary_school": {
        "label": "Provincial Secondary School (Type 1C)",
        "emoji": "📚",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     3,     3,     5,      3,      3,      3,     5],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1/3,    1/3,   1],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1/3,    1/3,   1],
        ],
        "description": "Regional feeder catchment (4-6km), balanced transport access and residential coverage, standard 200m track",
    },
    "girls_school": {
        "label": "Girls' High School",
        "emoji": "🌸",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     3,     3,     5,      5,      3,      3,     5],
            [1/3,   1,     1,     3,      3,      1,      1,     3],
            [1/3,   1,     1,     3,      3,      1,      1,     3],
            [1/5,   1/3,   1/3,   1,      1,      1/3,    1/3,   1],
            [1/5,   1/3,   1/3,   1,      1,      1/3,    1/3,   1],
            [1/3,   1,     1,     3,      3,      1,      1,     3],
            [1/3,   1,     1,     3,      3,      1,      1,     3],
            [1/5,   1/3,   1/3,   1,      1,      1/3,    1/3,   1],
        ],
        "description": "Dedicated female demographic cohort, safe access corridors, secure campus boundary, direct transit links",
    },
    "boys_school": {
        "label": "Boys' College",
        "emoji": "🦁",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     3,     3,     5,      3,      3,      3,     5],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1/3,    1/3,   1],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/3,   1,     1,     3,      1,      1,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1/3,    1/3,   1],
        ],
        "description": "Dedicated male demographic cohort, expansive campus for rugby/cricket grounds & cadet drills, arterial road access",
    },
    "hospital": {
        "label": "Hospital (Rohala)",
        "emoji": "🏥",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     5,     3,     7,      7,      3,      5,     9],
            [1/5,   1,     1/3,   3,      3,      1/3,    1,     5],
            [1/3,   3,     1,     5,      5,      1,      3,     7],
            [1/7,   1/3,   1/5,   1,      1,      1/5,    1/3,   3],
            [1/7,   1/3,   1/5,   1,      1,      1/5,    1/3,   3],
            [1/3,   3,     1,     5,      5,      1,      3,     7],
            [1/5,   1,     1/3,   3,      3,      1/3,    1,     5],
            [1/9,   1/5,   1/7,   1/3,    1/3,    1/7,    1/5,   1],
        ],
        "description": "Road access is critical for ambulances, high population coverage, emergency corridor",
    },
    "transport_hub": {
        "label": "Transport Hub (Nagarigam Madyasthanaya)",
        "emoji": "🚌",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     7,     3,     5,      3,      3,      5,     7],
            [1/7,   1,     1/5,   1/3,    1/5,    1/5,    1/3,   1],
            [1/3,   5,     1,     3,      1,      1,      3,     5],
            [1/5,   3,     1/3,   1,      1/3,    1/3,    1,     3],
            [1/3,   5,     1,     3,      1,      1,      3,     5],
            [1/3,   5,     1,     3,      1,      1,      3,     5],
            [1/5,   3,     1/3,   1,      1/3,    1/3,    1,     3],
            [1/7,   1,     1/5,   1/3,    1/5,    1/5,    1/3,   1],
        ],
        "description": "Major highway intersections, population catchment, multi-modal transport links",
    },
    "market": {
        "label": "Market / Commercial Center (Kadei)",
        "emoji": "🏪",
        "criteria": CRITERIA_8,
        "matrix": [
            [1,     3,     3,     5,      3,      5,      3,     5],
            [1/3,   1,     1,     3,      1,      3,      1,     3],
            [1/3,   1,     1,     3,      1,      3,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1,      1/3,   1],
            [1/3,   1,     1,     3,      1,      3,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1,      1/3,   1],
            [1/3,   1,     1,     3,      1,      3,      1,     3],
            [1/5,   1/3,   1/3,   1,      1/3,    1,      1/3,   1],
        ],
        "description": "Balanced commercial land use, customer density, accessible transport terminals",
    },
}

# Regional presets within Full Matale District
REGION_PRESETS = {
    "all": {
        "name": "Full Matale District",
        "bounds": [0, 200, 0, 200],  # [min_row, max_row, min_col, max_col]
        "desc": "Entire Matale District search space (40,000 locations)"
    },
    "dambulla": {
        "name": "Dambulla & Sigiriya (North)",
        "bounds": [10, 75, 40, 140],
        "desc": "Northern economic corridor, cultural triangle & tourism belt"
    },
    "galewela": {
        "name": "Galewela (Western Zone)",
        "bounds": [55, 115, 10, 75],
        "desc": "Flat agricultural terrain, developing rural population"
    },
    "naula": {
        "name": "Naula & Central Zone",
        "bounds": [80, 135, 45, 115],
        "desc": "Central connectivity corridor linking North and South"
    },
    "rattota": {
        "name": "Rattota & Knuckles Belt",
        "bounds": [120, 180, 65, 190],
        "desc": "Hilly terrain, environmentally sensitive buffer areas"
    },
    "matale_urban": {
        "name": "Matale Town & Urban South",
        "bounds": [140, 198, 30, 105],
        "desc": "High population density, commercial hubs, high school demand"
    },
}

# In-memory data caching
CACHED_LAYERS = None
CACHED_SLOPE_RAW = None
CACHED_FACILITIES = None
CACHED_WATERWAYS = None


def get_spatial_data():
    """Load spatial layers once and cache in memory."""
    global CACHED_LAYERS, CACHED_SLOPE_RAW
    if CACHED_LAYERS is None:
        print("  [INIT] Loading real spatial layers into memory...")
        loader = RealDataLoader(data_dir=DATA_DIR, grid_size=GRID_SIZE, random_seed=42)
        CACHED_LAYERS = loader.generate_all()
        CACHED_SLOPE_RAW = loader.calculate_slope(CACHED_LAYERS["elevation"])
        print("  [INIT] Spatial layers loaded and cached.")
    return CACHED_LAYERS, CACHED_SLOPE_RAW


def get_facilities_data():
    """Load real existing schools, hospitals, and critical utilities coordinates."""
    global CACHED_FACILITIES
    if CACHED_FACILITIES is None:
        cache_path = os.path.join("outputs", "facilities_geojson.json")
        sch_list = []
        hosp_list = []
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached = json.load(f)
                    sch_list = cached.get("schools", [])
                    hosp_list = cached.get("hospitals", [])
            except Exception:
                pass
        
        if not sch_list or not hosp_list:
            try:
                shp_path = os.path.join(DATA_DIR, "gis_osm_pois_free_1.shp")
                pois = gpd.read_file(shp_path, bbox=(MATALE_BOUNDS["min_lon"], MATALE_BOUNDS["min_lat"], MATALE_BOUNDS["max_lon"], MATALE_BOUNDS["max_lat"]))
                schools = pois[pois["fclass"] == "school"]
                hospitals = pois[pois["fclass"] == "hospital"]
                sch_list = [{"name": str(r.get("name") or "School"), "lat": round(float(r.geometry.y), 4), "lon": round(float(r.geometry.x), 4)} for _, r in schools.iterrows() if r.geometry is not None]
                hosp_list = [{"name": str(r.get("name") or "Hospital"), "lat": round(float(r.geometry.y), 4), "lon": round(float(r.geometry.x), 4)} for _, r in hospitals.iterrows() if r.geometry is not None]
            except Exception as e:
                print("  [WARN] Failed to load facilities shapefile:", e)

        # Load enhanced 225 critical utilities (Police, Fuel, Banks, Pharmacies)
        util_path = os.path.join(DATA_DIR, "matale_critical_utilities.json")
        util_list = []
        if os.path.exists(util_path):
            try:
                with open(util_path, "r", encoding="utf-8") as f:
                    util_list = json.load(f)
            except Exception as e:
                print("  [WARN] Failed to load utilities json:", e)

        CACHED_FACILITIES = {"schools": sch_list, "hospitals": hosp_list, "utilities": util_list}
    return CACHED_FACILITIES


def get_waterways_data():
    """Load real waterways and stream lines for interactive Leaflet map overlay."""
    global CACHED_WATERWAYS
    if CACHED_WATERWAYS is None:
        cache_path = os.path.join("outputs", "matale_waterways.geojson")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    CACHED_WATERWAYS = json.load(f)
            except Exception:
                pass
        if CACHED_WATERWAYS is None:
            try:
                shp_path = os.path.join(DATA_DIR, "gis_osm_waterways_free_1.shp")
                if os.path.exists(shp_path):
                    gdf = gpd.read_file(shp_path, bbox=(MATALE_BOUNDS["min_lon"], MATALE_BOUNDS["min_lat"], MATALE_BOUNDS["max_lon"], MATALE_BOUNDS["max_lat"]))
                    gdf_simple = gdf.copy()
                    gdf_simple["geometry"] = gdf.geometry.simplify(0.0008)
                    geojson_dict = json.loads(gdf_simple[["name", "fclass", "geometry"]].to_json())
                    CACHED_WATERWAYS = geojson_dict
                    os.makedirs("outputs", exist_ok=True)
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(geojson_dict, f)
                else:
                    CACHED_WATERWAYS = {"type": "FeatureCollection", "features": []}
            except Exception as e:
                print("  [WARN] Failed to load waterways shapefile:", e)
                CACHED_WATERWAYS = {"type": "FeatureCollection", "features": []}
    return CACHED_WATERWAYS


def grid_to_latlon(row: float, col: float) -> Tuple[float, float]:
    """Convert grid (row, col) to geographic (lat, lon)."""
    lat = MATALE_BOUNDS["max_lat"] - (row / (GRID_SIZE - 1)) * (MATALE_BOUNDS["max_lat"] - MATALE_BOUNDS["min_lat"])
    lon = MATALE_BOUNDS["min_lon"] + (col / (GRID_SIZE - 1)) * (MATALE_BOUNDS["max_lon"] - MATALE_BOUNDS["min_lon"])
    return float(lat), float(lon)


def latlon_to_grid(lat: float, lon: float) -> Tuple[int, int]:
    """Convert geographic (lat, lon) to grid (row, col)."""
    row = int(np.round((MATALE_BOUNDS["max_lat"] - lat) / (MATALE_BOUNDS["max_lat"] - MATALE_BOUNDS["min_lat"]) * (GRID_SIZE - 1)))
    col = int(np.round((lon - MATALE_BOUNDS["min_lon"]) / (MATALE_BOUNDS["max_lon"] - MATALE_BOUNDS["min_lon"]) * (GRID_SIZE - 1)))
    row = int(np.clip(row, 0, GRID_SIZE - 1))
    col = int(np.clip(col, 0, GRID_SIZE - 1))
    return row, col


def generate_transparent_overlay_png(suitability: np.ndarray, constraint_mask: np.ndarray) -> str:
    """Generate transparent PNG image of suitability surface for Leaflet imageOverlay."""
    cmap = plt.colormaps.get_cmap("plasma")
    norm_suit = np.clip(suitability, 0.0, 1.0)
    rgba = cmap(norm_suit)
    # Set transparency: buildable cells get 0.70 opacity, constrained get 0.0
    rgba[..., 3] = np.where(constraint_mask > 0, 0.70, 0.0)
    
    img_bytes = (rgba * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_bytes, mode="RGBA")
    
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_optimization(
    infra_type: str,
    roi: Optional[Tuple[int, int, int, int]] = None,
    zone_label: str = "Full District"
) -> dict:
    """Run optimization pipeline strictly within the selected ROI bounding box."""
    preset = INFRASTRUCTURE_PRESETS[infra_type]
    layers, slope_raw = get_spatial_data()
    
    # 1. Compute AHP weights
    engine = AHPEngine(
        criteria_names=preset["criteria"],
        comparison_matrix=np.array(preset["matrix"]),
    )
    weights, metrics = engine.run()
    
    # 2. Build Criteria Rasters
    criteria_rasters = {}
    for cname in preset["criteria"]:
        layer_key = CRITERIA_LAYER_MAP.get(cname)
        if layer_key is None:
            criteria_rasters[cname] = layers["population_growth"] * 0.8
        else:
            criteria_rasters[cname] = layers[layer_key]
            
    # 3. Create ROI constraint mask
    constraint_mask = layers["constraint_mask"].copy()
    is_sub_region = False
    
    if roi is not None:
        min_x, max_x, min_y, max_y = roi
        min_x = max(0, min(GRID_SIZE - 5, int(min_x)))
        max_x = max(min_x + 5, min(GRID_SIZE, int(max_x)))
        min_y = max(0, min(GRID_SIZE - 5, int(min_y)))
        max_y = max(min_y + 5, min(GRID_SIZE, int(max_y)))
        
        roi_mask = np.zeros_like(constraint_mask)
        roi_mask[min_x:max_x, min_y:max_y] = 1.0
        constraint_mask = constraint_mask * roi_mask
        is_sub_region = (min_x > 0 or max_x < GRID_SIZE or min_y > 0 or max_y < GRID_SIZE)
        clamped_roi = (min_x, max_x, min_y, max_y)
    else:
        clamped_roi = (0, GRID_SIZE, 0, GRID_SIZE)
        
    buildable_pixels = int(constraint_mask.sum())
    if buildable_pixels == 0:
        print("  [WARN] Selected area has no buildable cells. Relaxing ROI.")
        constraint_mask = layers["constraint_mask"].copy()
        clamped_roi = (0, GRID_SIZE, 0, GRID_SIZE)
        is_sub_region = False

    # 4. Build Full Suitability Surface for visualization
    surface_builder = SuitabilitySurface(
        criteria_rasters=criteria_rasters,
        weights=weights,
        constraint_mask=layers["constraint_mask"],
        elevation=layers["elevation"],
    )
    suitability = surface_builder.build_surface()
    
    # 5. Extract optimization grid
    min_x, max_x, min_y, max_y = clamped_roi
    if is_sub_region:
        opt_suit = suitability[min_x:max_x, min_y:max_y]
        opt_elev = layers["elevation"][min_x:max_x, min_y:max_y]
        opt_slope = slope_raw[min_x:max_x, min_y:max_y]
        opt_mask = constraint_mask[min_x:max_x, min_y:max_y]
        offset_x, offset_y = min_x, min_y
    else:
        opt_suit = suitability
        opt_elev = layers["elevation"]
        opt_slope = slope_raw
        opt_mask = constraint_mask
        offset_x, offset_y = 0, 0
    
    # 6. Run GA on active grid
    t0 = time.time()
    ga = SpatialGeneticAlgorithm(
        suitability_matrix=opt_suit,
        elevation=opt_elev,
        slope=opt_slope,
        constraint_mask=opt_mask,
        population_size=50,
        generations=80,
        random_seed=42,
    )
    ga_result = ga.optimize()
    ga_time = time.time() - t0
    
    # 7. Run PSO on active grid
    t0 = time.time()
    pso = SpatialPSO(
        suitability_matrix=opt_suit,
        elevation=opt_elev,
        slope=opt_slope,
        constraint_mask=opt_mask,
        swarm_size=50,
        iterations=80,
        random_seed=42,
    )
    pso_result = pso.optimize()
    pso_time = time.time() - t0
    
    # Map back to global Matale grid coordinates
    ga_result["best_x"] += offset_x
    ga_result["best_y"] += offset_y
    pso_result["best_x"] += offset_x
    pso_result["best_y"] += offset_y
    
    # Pick winner
    ga_fit = ga_result["best_fitness"]
    pso_fit = pso_result["best_fitness"]
    if pso_fit >= ga_fit:
        winner = pso_result
        winner_name = "PSO" if pso_fit > ga_fit else "Tie"
    else:
        winner = ga_result
        winner_name = "GA"
        
    best_x, best_y = winner["best_x"], winner["best_y"]
    best_lat, best_lon = grid_to_latlon(best_x, best_y)
    
    # Calculate Lat/Lon bounding box of ROI
    roi_south, roi_west = grid_to_latlon(clamped_roi[1] - 1, clamped_roi[0])
    roi_north, roi_east = grid_to_latlon(clamped_roi[0], clamped_roi[3] - 1)
    
    # Generate Plots and Leaflet overlay
    ga_history = ga_result.get("fitness_history", [])
    pso_history = pso_result.get("convergence_history", [])
    
    heatmap_b64 = generate_heatmap(suitability, ga_result, pso_result, infra_type, clamped_roi, is_sub_region)
    convergence_b64 = generate_convergence(ga_history, pso_history, infra_type, zone_label)
    leaflet_overlay_b64 = generate_transparent_overlay_png(suitability, layers["constraint_mask"])
    
    # Predict building parameters & architectural sizing
    bx_int = int(np.clip(np.round(best_x), 0, GRID_SIZE - 1))
    by_int = int(np.clip(np.round(best_y), 0, GRID_SIZE - 1))
    slope_deg = float(slope_raw[bx_int, by_int])

    building_specs = BuildingFeaturePredictor.predict_features(
        infra_type=infra_type,
        best_row=best_x,
        best_col=best_y,
        elevation_val=float(winner["best_z"]),
        slope_deg=slope_deg,
        population_grid=layers["population_growth"],
    )
    
    return {
        "infra_type": infra_type,
        "infra_label": preset["label"],
        "infra_emoji": preset["emoji"],
        "zone_label": zone_label,
        "roi": list(clamped_roi),
        "roi_geo": {
            "south": float(roi_south),
            "north": float(roi_north),
            "west": float(roi_west),
            "east": float(roi_east),
        },
        "is_sub_region": is_sub_region,
        "buildable_pixels": buildable_pixels,
        "description": preset["description"],
        "weights": {name: float(w) for name, w in zip(preset["criteria"], weights)},
        "cr": float(metrics["cr"]),
        "cr_pass": metrics["cr"] < 0.10,
        "best_fitness": float(winner["best_fitness"]),
        "best_x": float(best_x),
        "best_y": float(best_y),
        "best_lat": float(best_lat),
        "best_lon": float(best_lon),
        "best_elevation": float(winner["best_z"]),
        "winner": winner_name,
        "ga": {
            "fitness": float(ga_result["best_fitness"]),
            "x": float(ga_result["best_x"]),
            "y": float(ga_result["best_y"]),
            "elevation": float(ga_result["best_z"]),
            "time": float(ga_time),
        },
        "pso": {
            "fitness": float(pso_result["best_fitness"]),
            "x": float(pso_result["best_x"]),
            "y": float(pso_result["best_y"]),
            "elevation": float(pso_result["best_z"]),
            "time": float(pso_time),
        },
        "building_specs": building_specs,
        "elevation_matrix": [[round(float(v), 1) for v in row] for row in layers["elevation"][::2, ::2]],
        "elevation_min": float(layers["elevation"].min()),
        "elevation_max": float(layers["elevation"].max()),
        "heatmap_b64": heatmap_b64,
        "convergence_b64": convergence_b64,
        "leaflet_overlay_b64": leaflet_overlay_b64,
    }
AHP_WEIGHTS_CACHE = {}


def get_ahp_weights(infra_type: str):
    """Cache and return AHP weights for each infrastructure preset."""
    if infra_type not in AHP_WEIGHTS_CACHE:
        preset = INFRASTRUCTURE_PRESETS[infra_type]
        engine = AHPEngine(
            criteria_names=preset["criteria"],
            comparison_matrix=np.array(preset["matrix"]),
        )
        weights, _ = engine.run()
        AHP_WEIGHTS_CACHE[infra_type] = weights
    return AHP_WEIGHTS_CACHE[infra_type]


def evaluate_site_suitability(lat: float, lon: float, facility_name: str = "", facility_type: str = "") -> dict:
    """
    Perform Highest & Best Use (HBU) Reverse Evaluation for any clicked location.
    If facility_name and facility_type are provided, generates a dedicated Retrospective
    Facility Audit assessing how well the existing operational facility conforms to
    Ministry of Education (MoE) / Health guidelines and AHP criteria.
    """
    layers, slope_raw = get_spatial_data()
    
    # Check within district bounds
    if not (MATALE_BOUNDS["min_lat"] <= lat <= MATALE_BOUNDS["max_lat"] and
            MATALE_BOUNDS["min_lon"] <= lon <= MATALE_BOUNDS["max_lon"]):
        return {
            "error": "Location is outside Matale District boundaries.",
            "bounds": MATALE_BOUNDS
        }
        
    row, col = latlon_to_grid(lat, lon)
    elev_val = float(layers["elevation"][row, col])
    slope_deg = float(slope_raw[row, col])
    is_buildable = bool(layers["constraint_mask"][row, col] > 0)
    
    # Extract criteria values at this pixel
    criteria_vals = {}
    for cname, lkey in CRITERIA_LAYER_MAP.items():
        if lkey is None:
            criteria_vals[cname] = float(layers["population_growth"][row, col] * 0.8)
        else:
            criteria_vals[cname] = float(layers[lkey][row, col])
            
    # Environmental / Policy Constraint Checks
    restriction_reasons = []

    waterways_mask = layers.get("waterways_mask")
    water_mask = layers.get("water_mask")
    protected_mask = layers.get("protected_mask")
    pofw_mask = layers.get("pofw_mask")

    is_waterway = bool(waterways_mask[row, col] > 0) if waterways_mask is not None else False
    is_water = bool(water_mask[row, col] > 0) if water_mask is not None else False
    is_protected = bool(protected_mask[row, col] > 0) if protected_mask is not None else False
    is_pofw = bool(pofw_mask[row, col] > 0) if pofw_mask is not None else False
    is_slope_violation = slope_deg > 20.0

    constraint_details = {
        "is_waterway_violation": is_waterway,
        "is_water_violation": is_water,
        "is_protected_violation": is_protected,
        "is_pofw_violation": is_pofw,
        "is_slope_violation": is_slope_violation,
        "primary_hazard": None,
        "hazard_title": "",
        "risk_description": "",
        "school_safety_reason": "",
        "statutory_law": "",
    }

    if is_waterway:
        constraint_details["primary_hazard"] = "RIVER_FLOOD"
        constraint_details["hazard_title"] = "Riparian River Flood Hazard & Statutory Reservation Buffer"
        constraint_details["risk_description"] = (
            "Located within the mandatory statutory 60m river / 30m stream riparian reservation zone "
            "(Sudu Ganga, Amban Ganga, or Mahaweli tributary network). High risk of seasonal flash-flooding, "
            "river bank scour/erosion, and sub-surface liquefaction."
        )
        constraint_details["school_safety_reason"] = (
            "Under Ministry of Education (MoE) Circulars and Child Protection Regulations, schools CANNOT be constructed "
            "within river flood buffers. Sudden monsoonal torrents pose life-threatening drowning hazards to students, "
            "block evacuation pathways, and water table saturation compromises school building foundations."
        )
        constraint_details["statutory_law"] = (
            "Sri Lanka Irrigation Ordinance (Cap. 453), State Lands Ordinance, and UDA Planning & Building Regulations "
            "(Gazette No. 2235/54) - mandatory 60m river reservation where permanent civil construction is strictly illegal."
        )
        restriction_reasons.append("🌊 Riparian River Flood Buffer: Siting inside 60m river reservation zone (Irrigation Ordinance Cap. 453 & UDA Norms)")

    if is_water:
        if not constraint_details["primary_hazard"]:
            constraint_details["primary_hazard"] = "WATER_BODY"
            constraint_details["hazard_title"] = "Open Water Body / Reservoir Inundation Zone"
            constraint_details["risk_description"] = "Site is situated directly within a permanent water body, tank/reservoir (wewa), or retention wetland."
            constraint_details["school_safety_reason"] = "Direct drowning hazard, permanent inundation, and structural impossibility for standard building foundations."
            constraint_details["statutory_law"] = "Central Environmental Authority (CEA) Wetland Protection Act and National Water Resources Board guidelines."
        restriction_reasons.append("💧 Open Water Body / Reservoir: Siting directly within permanent water body or retention wetland")

    if is_protected:
        if not constraint_details["primary_hazard"]:
            constraint_details["primary_hazard"] = "PROTECTED_FOREST"
            constraint_details["hazard_title"] = "Knuckles Conservation Forest / Protected Reserve Sanctuary"
            constraint_details["risk_description"] = "Site is inside the Knuckles Conservation Forest or protected environmental sanctuary in Matale."
            constraint_details["school_safety_reason"] = "Institutional construction strictly prohibited to protect pristine biodiversity corridors, watershed catchments, and endemic wildlife."
            constraint_details["statutory_law"] = "National Environmental Act (NEA) No. 47 of 1980, Forest Conservation Ordinance, and Fauna & Flora Protection Ordinance."
        restriction_reasons.append("🌲 Protected Nature Reserve: Located inside Knuckles Conservation Forest or state protected area")

    if is_pofw:
        if not constraint_details["primary_hazard"]:
            constraint_details["primary_hazard"] = "SACRED_PRECINCT"
            constraint_details["hazard_title"] = "Sacred Religious Precinct Buffer"
            constraint_details["risk_description"] = "Located within the 30m precinct buffer of an active Buddhist temple, Hindu kovil, mosque, or church."
            constraint_details["school_safety_reason"] = "To prevent noise pollution, traffic disruption, and preserve spiritual serenity of places of worship."
            constraint_details["statutory_law"] = "Ministry of Buddhasasana, Religious and Cultural Affairs Guidelines and UDA Heritage Zoning Regulations."
        restriction_reasons.append("🏛️ Sacred Precinct Buffer: Located within 30m buffer of a temple/church/mosque/kovil")

    if is_slope_violation:
        if not constraint_details["primary_hazard"]:
            constraint_details["primary_hazard"] = "LANDSLIDE"
            constraint_details["hazard_title"] = "Severe Landslide Risk Zone (Slope > 20°)"
            constraint_details["risk_description"] = f"Terrain slope is {slope_deg:.1f}°, which exceeds the NBRO 20.0° threshold for safe construction."
            constraint_details["school_safety_reason"] = "High catastrophic slope failure and mass movement hazard during intense monsoons. Ineligible for NBRO landslide clearance."
            constraint_details["statutory_law"] = "National Building Research Organisation (NBRO) Landslide Risk Siting Guidelines & Disaster Management Act No. 13 of 2005."
        restriction_reasons.append(f"⚠️ Steep Landslide Hazard: Terrain slope is {slope_deg:.1f}° (exceeds NBRO 20° safety limit)")

    if not is_buildable and not restriction_reasons:
        constraint_details["primary_hazard"] = "BOUNDARY_MARGIN"
        constraint_details["hazard_title"] = "District Computational Boundary Margin"
        constraint_details["risk_description"] = "Located within the 5-pixel peripheral margin of the Matale computational grid."
        constraint_details["school_safety_reason"] = "Safety edge margin to prevent raster boundary interpolation artifacts."
        constraint_details["statutory_law"] = "District Siting Computational Norm."
        restriction_reasons.append("🗺️ District Boundary Margin: Peripheral computational border zone")
        
    # Evaluate each infrastructure preset
    results = []
    for itype, preset in INFRASTRUCTURE_PRESETS.items():
        weights = get_ahp_weights(itype)
        raw_score = sum(weights[i] * criteria_vals[cname] for i, cname in enumerate(preset["criteria"]))
        final_score = raw_score * (1.0 if is_buildable else 0.0)
        percentage = round(final_score * 100.0, 1)
        
        strengths = []
        weaknesses = []
        
        # Slope assessment
        if slope_deg <= 8.0:
            strengths.append(f"Flat topography ({slope_deg:.1f}°) enables safe low-cost construction")
        elif slope_deg <= 15.0:
            strengths.append(f"Moderate slope ({slope_deg:.1f}°) - buildable with standard grading")
        else:
            weaknesses.append(f"Steep slope ({slope_deg:.1f}°) increases structural foundation cost")
            
        # Road accessibility
        if criteria_vals["Road Proximity"] >= 0.70:
            strengths.append("High accessibility directly on major arterial road network")
        elif criteria_vals["Road Proximity"] < 0.35:
            weaknesses.append("Poor road access / isolated from main transit network")
            
        # Demographic Catchment
        if criteria_vals["Population Density"] >= 0.50:
            strengths.append("Dense demographic catchment ensures strong operational demand")
        elif criteria_vals["Population Density"] < 0.15:
            weaknesses.append("Sparse rural population catchment")
            
        # Infrastructure-specific checks
        if "school" in itype:
            if criteria_vals["School Distance"] >= 0.65:
                strengths.append("Fills an educational service gap without competing with nearby schools")
            elif criteria_vals["School Distance"] < 0.30:
                weaknesses.append("High overlap with existing schools already serving this zone")
                
        if "hospital" in itype or "clinic" in itype:
            if criteria_vals["Healthcare Proximity"] >= 0.60:
                strengths.append("Rapid connectivity for emergency transport and ambulance access")
            else:
                weaknesses.append("Sub-optimal transit link for regional emergency healthcare")
                
        if itype == "transport_hub":
            if criteria_vals["Transport Access"] >= 0.65 and criteria_vals["Road Proximity"] >= 0.65:
                strengths.append("Ideal multi-modal junction connecting key transit routes")
                
        if itype == "market":
            if criteria_vals["Land Use Suitability"] >= 0.50:
                strengths.append("Favorable commercial zoning and high customer footfall")

        results.append({
            "type": itype,
            "label": preset["label"],
            "emoji": preset["emoji"],
            "score": percentage,
            "strengths": strengths[:3],
            "weaknesses": weaknesses[:3],
            "description": preset["description"],
        })
        
    results.sort(key=lambda x: x["score"], reverse=True)
    winner = results[0]
    
    # Predict building parameters for top recommendation
    winner_specs = BuildingFeaturePredictor.predict_features(
        infra_type=winner["type"],
        best_row=row,
        best_col=col,
        elevation_val=elev_val,
        slope_deg=slope_deg,
        population_grid=layers["population_growth"],
    )

    # ─────────────────────────────────────────────────────────────
    # Dedicated Facility Retrospective Audit (when examining an existing school/hospital)
    # ─────────────────────────────────────────────────────────────
    facility_audit = None
    if facility_name and facility_type:
        target_preset_key = "school" if "school" in facility_type.lower() else "hospital" if "hospital" in facility_type.lower() else facility_type.lower()
        if target_preset_key not in INFRASTRUCTURE_PRESETS:
            target_preset_key = "school"
            
        target_preset = INFRASTRUCTURE_PRESETS[target_preset_key]
        target_weights = get_ahp_weights(target_preset_key)
        
        # For existing facilities, treat local anchor role as high catchment service
        adj_vals = dict(criteria_vals)
        if "school" in target_preset_key:
            adj_vals["School Distance"] = max(adj_vals["School Distance"], 0.72)
        elif "hospital" in target_preset_key:
            adj_vals["Healthcare Proximity"] = max(adj_vals["Healthcare Proximity"], 0.75)
            
        target_raw = sum(target_weights[i] * adj_vals[cname] for i, cname in enumerate(target_preset["criteria"]))
        target_score = target_raw * (1.0 if is_buildable else 0.45)
        target_pct = round(target_score * 100.0, 1)
        
        if target_pct >= 75.0:
            grade = "Grade A"
            grade_desc = "Highly Optimal & Resilient Historical Location"
            grade_color = "#4ade80"
        elif target_pct >= 60.0:
            grade = "Grade B"
            grade_desc = "Favorable Location with Manageable Site Constraints"
            grade_color = "#38bdf8"
        elif target_pct >= 45.0:
            grade = "Grade C"
            grade_desc = "Sub-optimal Siting (Moderate Terrain / Transit Limitations)"
            grade_color = "#facc15"
        else:
            grade = "Grade D"
            grade_desc = "High Hazard / Restrictive Siting"
            grade_color = "#f87171"
            
        compliance = [
            {
                "standard": "NBRO Landslide Safety",
                "status": "PASS" if slope_deg <= 20.0 else "FAIL",
                "detail": f"Terrain slope is {slope_deg:.1f}° (Safely below NBRO 20° ceiling)" if slope_deg <= 20.0 else f"High Hazard: Slope {slope_deg:.1f}° exceeds NBRO 20° threshold",
                "passed": slope_deg <= 20.0
            },
            {
                "standard": "Riparian & River Flood Clearance",
                "status": "FAIL" if is_waterway else "PASS",
                "detail": "Inside mandatory 60m river reservation buffer" if is_waterway else "Safely clear of river flood inundation buffers",
                "passed": not is_waterway
            },
            {
                "standard": "CEA Conservation & Wetland Clearance",
                "status": "FAIL" if (is_water or is_protected) else "PASS",
                "detail": "Overlaps protected forest reserve or wetland buffer" if (is_water or is_protected) else "Completely clear of forest reserves and wetland buffers",
                "passed": not (is_water or is_protected)
            },
            {
                "standard": "Arterial Road & Emergency Access",
                "status": "PASS" if criteria_vals["Road Proximity"] >= 0.40 else "WARNING",
                "detail": f"Road Proximity index is {criteria_vals['Road Proximity']*100:.0f}% (Immediate transit connectivity)" if criteria_vals["Road Proximity"] >= 0.40 else "Access via narrow rural secondary lanes",
                "passed": criteria_vals["Road Proximity"] >= 0.40
            },
            {
                "standard": "Demographic Catchment Coverage",
                "status": "PASS" if criteria_vals["Population Density"] >= 0.30 else "MODERATE",
                "detail": f"Population demand catchment index: {criteria_vals['Population Density']*100:.0f}%",
                "passed": criteria_vals["Population Density"] >= 0.30
            }
        ]
        
        audit_strengths = []
        audit_recommendations = []
        
        if slope_deg <= 8.0:
            audit_strengths.append("Safe flat topography (< 8°): Ideal for student playgrounds, assembly grounds, and multi-story expansion.")
        elif slope_deg <= 15.0:
            audit_strengths.append(f"Moderate hillside slope ({slope_deg:.1f}°): Natural stormwater drainage, requires terraced landscaping.")
        else:
            audit_recommendations.append(f"High terrain slope ({slope_deg:.1f}°): Implement reinforced retaining walls and routine slope monitoring.")
            
        if criteria_vals["Road Proximity"] >= 0.60:
            audit_strengths.append("Prime arterial accessibility: Direct connectivity for school buses, staff vehicles, and emergency services.")
            audit_recommendations.append("Traffic Management: Implement pedestrian zebra crossings and 20 km/h school-zone speed restrictions.")
        else:
            audit_recommendations.append("Access Road Upgrade: Widen secondary access roads to allow safe 2-way school bus movement.")
            
        if is_buildable:
            audit_strengths.append("100% compliant with Central Environmental Authority (CEA) and Urban Development Authority (UDA) zoning.")
            
        target_specs = BuildingFeaturePredictor.predict_features(
            infra_type=target_preset_key,
            best_row=row,
            best_col=col,
            elevation_val=elev_val,
            slope_deg=slope_deg,
            population_grid=layers["population_growth"],
        )
        
        facility_audit = {
            "is_facility_audit": True,
            "facility_name": facility_name,
            "facility_type": facility_type,
            "target_key": target_preset_key,
            "label": target_preset["label"],
            "emoji": target_preset["emoji"],
            "score": target_pct,
            "grade": grade,
            "grade_desc": grade_desc,
            "grade_color": grade_color,
            "compliance": compliance,
            "strengths": audit_strengths,
            "recommendations": audit_recommendations,
            "predicted_specs": target_specs,
        }
    
    return {
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "grid_row": row,
        "grid_col": col,
        "elevation": round(elev_val, 1),
        "slope_deg": round(slope_deg, 1),
        "is_buildable": is_buildable,
        "restriction_reasons": restriction_reasons,
        "constraint_details": constraint_details,
        "criteria_breakdown": {k: round(v * 100, 1) for k, v in criteria_vals.items()},
        "top_recommendation": winner,
        "rankings": results,
        "predicted_specs": winner_specs,
        "facility_audit": facility_audit,
    }


def generate_heatmap(
    suitability: np.ndarray,
    ga_result: dict,
    pso_result: dict,
    infra_type: str,
    roi: Tuple[int, int, int, int],
    is_sub_region: bool
) -> str:
    """Generate suitability heatmap with marked optimal site and selected squad bounding box."""
    preset = INFRASTRUCTURE_PRESETS[infra_type]
    fig, ax = plt.subplots(figsize=(9, 7))
    
    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#1a0a2e", "#16213e", "#0f3460", "#e94560", "#f5a623", "#f7f7f7"]
    )
    im = ax.imshow(suitability, cmap=cmap, origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="Suitability Score", fraction=0.046, pad=0.04)
    
    min_x, max_x, min_y, max_y = roi
    
    if is_sub_region:
        rect = patches.Rectangle(
            (min_y, min_x), max_y - min_y, max_x - min_x,
            linewidth=2.8, edgecolor="#00f2fe", facecolor="none",
            linestyle="--", label="Selected Squad / Area"
        )
        ax.add_patch(rect)
        outside_mask = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
        outside_mask[min_x:max_x, min_y:max_y] = False
        ax.imshow(outside_mask, cmap="binary", origin="lower", alpha=0.35, aspect="auto")

    ax.plot(
        ga_result["best_y"], ga_result["best_x"], "o",
        color="#ff4757", markersize=11, markeredgecolor="white", markeredgewidth=1.5,
        label=f"GA Optimal ({ga_result['best_fitness']:.3f})"
    )
    ax.plot(
        pso_result["best_y"], pso_result["best_x"], "*",
        color="#ffd700", markersize=20, markeredgecolor="black", markeredgewidth=1.5,
        label=f"PSO Optimal ({pso_result['best_fitness']:.3f})"
    )
    
    title_suffix = " (Selected Zone)" if is_sub_region else " (Full District)"
    ax.set_title(f"Optimal Site for {preset['label']}{title_suffix}\nMatale District, Sri Lanka", fontweight="bold", fontsize=12)
    ax.set_xlabel("Grid Y (Longitude axis / Columns)", fontsize=10)
    ax.set_ylabel("Grid X (Latitude axis / Rows)", fontsize=10)
    ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#ffffff", labelcolor="white", fontsize=9)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def generate_convergence(ga_history, pso_history, infra_type, zone_label) -> str:
    """Generate convergence comparison plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ga_hist = np.clip(ga_history, 0, 1.05)
    pso_hist = np.clip(pso_history, 0, 1.05)
    
    ax.plot(range(1, len(ga_hist) + 1), ga_hist, color="#ff4757", linewidth=2, label="GA (Fitness)", marker="o", markersize=2.5)
    ax.plot(range(1, len(pso_hist) + 1), pso_hist, color="#1e90ff", linewidth=2, label="PSO (Convergence)", marker="s", markersize=2.5)
    
    ax.set_xlabel("Iteration / Generation", fontsize=11)
    ax.set_ylabel("Suitability Score", fontsize=11)
    ax.set_title(f"Convergence: GA vs PSO — {zone_label}", fontweight="bold", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_main_page() -> str:
    """Build modern HTML UI with Google/Leaflet Map integration."""
    PRIMARY_INFRA = [
        ("school", "School", "🏫"),
        ("hospital", "Hospital", "🏥"),
        ("transport_hub", "Transport Hub", "🚌"),
        ("market", "Market / Commercial", "🏪"),
    ]
    infra_buttons_html = ""
    for key, label, emoji in PRIMARY_INFRA:
        infra_buttons_html += f"""
        <button class="infra-btn" onclick="selectPrimaryInfra('{key}')" id="btn-{key}">
            <span class="btn-emoji">{emoji}</span>
            <span class="btn-label">{label}</span>
        </button>
        """

    region_buttons_html = ""
    for key, reg in REGION_PRESETS.items():
        b = reg["bounds"]
        region_buttons_html += f"""
        <button class="zone-btn" onclick="applyZonePreset('{key}', {b[0]}, {b[1]}, {b[2]}, {b[3]})" id="zone-{key}">
            <span class="zone-title">{reg['name']}</span>
            <span class="zone-desc">{reg['desc']}</span>
        </button>
        """

    facilities = get_facilities_data()
    facilities_json_str = json.dumps(facilities)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Maps &amp; AI Spatial Site Selector - Matale District</title>
    <!-- Leaflet.js CSS & JS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <!-- Three.js & OrbitControls for 3D WebGL Terrain -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #090814, #141228, #16152b);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding: 22px 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 1.85rem;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #4ade80, #38bdf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header p {{ color: #94a3b8; font-size: 0.95rem; margin-top: 6px; }}
        .container {{ max-width: 1320px; margin: 0 auto; padding: 25px 20px; }}
        
        .control-panel {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 960px) {{ .control-panel {{ grid-template-columns: 1fr; }} }}
        
        .panel-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 22px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .panel-card h2 {{
            font-size: 1.15rem;
            margin-bottom: 6px;
            color: #f1f5f9;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .panel-card .subtitle {{
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 16px;
        }}
        
        /* Step 1 Infra buttons */
        .infra-buttons {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }}
        .infra-btn {{
            background: rgba(255,255,255,0.06);
            border: 2px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 14px 10px;
            cursor: pointer;
            transition: all 0.25s;
            color: #f1f5f9;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 6px;
        }}
        .infra-btn:hover {{
            background: rgba(56,189,248,0.15);
            border-color: #38bdf8;
            transform: translateY(-2px);
        }}
        .infra-btn.active {{
            background: rgba(56,189,248,0.22);
            border-color: #38bdf8;
            box-shadow: 0 0 15px rgba(56,189,248,0.3);
        }}
        .btn-emoji {{ font-size: 1.8rem; }}
        .btn-label {{ font-size: 0.85rem; font-weight: 600; text-align: center; }}
        
        /* School Subtypes Container */
        .school-subtypes-container {{
            margin-top: 14px;
            background: rgba(0,0,0,0.28);
            border-radius: 12px;
            padding: 12px;
            border: 1px dashed rgba(56,189,248,0.35);
            transition: all 0.3s ease;
        }}
        .subtype-header {{
            font-size: 0.82rem;
            color: #38bdf8;
            font-weight: 600;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .subtype-pills {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .subtype-pill {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            color: #cbd5e1;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 0.82rem;
            cursor: pointer;
            transition: all 0.2s;
            text-align: left;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .subtype-pill:hover {{
            background: rgba(56,189,248,0.15);
            border-color: #38bdf8;
            color: #fff;
            transform: translateX(2px);
        }}
        .subtype-pill.active {{
            background: rgba(56,189,248,0.22);
            border-color: #38bdf8;
            color: #38bdf8;
            font-weight: 600;
            box-shadow: 0 0 10px rgba(56,189,248,0.25);
        }}
        
        /* Step 2 Squad / Area Selection */
        .zone-presets {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 16px;
        }}
        .zone-btn {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 10px;
            padding: 10px 14px;
            cursor: pointer;
            text-align: left;
            transition: all 0.2s;
            color: #e2e8f0;
            display: flex;
            flex-direction: column;
        }}
        .zone-btn:hover {{
            background: rgba(74,222,128,0.12);
            border-color: #4ade80;
        }}
        .zone-btn.active {{
            background: rgba(74,222,128,0.2);
            border-color: #4ade80;
            box-shadow: 0 0 12px rgba(74,222,128,0.25);
        }}
        .zone-title {{ font-size: 0.9rem; font-weight: 600; color: #f8fafc; }}
        .zone-desc {{ font-size: 0.75rem; color: #94a3b8; margin-top: 2px; }}
        
        .coords-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .coord-field {{ display: flex; flex-direction: column; gap: 4px; }}
        .coord-field label {{ font-size: 0.75rem; color: #94a3b8; }}
        .coord-field input {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 6px;
            color: #fff;
            padding: 6px 10px;
            font-size: 0.85rem;
            width: 100%;
        }}
        
        .run-btn {{
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            color: #0b091a;
            border: none;
            border-radius: 12px;
            padding: 14px 32px;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 20px rgba(56,189,248,0.3);
            width: 100%;
            margin-top: 15px;
        }}
        .run-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(56,189,248,0.5);
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
        }}
        
        /* Map Section */
        .map-section {{
            background: rgba(255,255,255,0.04);
            border-radius: 18px;
            padding: 22px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 25px;
            box-shadow: 0 10px 35px rgba(0,0,0,0.4);
        }}
        .map-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .map-header h2 {{
            font-size: 1.25rem;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .map-controls {{
            display: flex;
            align-items: center;
            gap: 14px;
            flex-wrap: wrap;
        }}
        .map-ctrl-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.82rem;
            color: #cbd5e1;
        }}
        .map-ctrl-item input[type="range"] {{
            width: 90px;
        }}
        
        #leaflet-map {{
            height: 540px;
            width: 100%;
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.12);
            z-index: 1;
        }}
        
        /* 2D / 3D Mode Switcher */
        .map-view-tabs {{
            display: inline-flex;
            background: rgba(0,0,0,0.4);
            padding: 4px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            gap: 4px;
        }}
        .view-tab {{
            background: transparent;
            border: none;
            color: #94a3b8;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 0.82rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .view-tab:hover {{
            color: #fff;
            background: rgba(255,255,255,0.06);
        }}
        .view-tab.active {{
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            color: #0b091a;
            box-shadow: 0 2px 10px rgba(56,189,248,0.3);
        }}
        
        /* 3D WebGL Toolbar */
        .three-toolbar {{
            position: absolute;
            top: 14px;
            left: 14px;
            right: 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(11, 9, 26, 0.88);
            backdrop-filter: blur(10px);
            padding: 10px 16px;
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.14);
            z-index: 10;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .three-ctrl {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            color: #cbd5e1;
        }}
        .three-ctrl select {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.18);
            color: #fff;
            padding: 5px 8px;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
        }}
        .three-ctrl select option {{
            background: #141228;
            color: #fff;
        }}
        .three-ctrl input[type="range"] {{ width: 90px; }}
        .three-btn {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            color: #f1f5f9;
            padding: 5px 12px;
            border-radius: 6px;
            font-size: 0.8rem;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.2s;
        }}
        .three-btn:hover {{
            background: rgba(56,189,248,0.22);
            border-color: #38bdf8;
            color: #38bdf8;
        }}
        .three-help {{
            font-size: 0.75rem;
            color: #94a3b8;
        }}
        
        .gmaps-link-btn {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(90deg, #4285F4, #34A853);
            color: white !important;
            padding: 10px 20px;
            border-radius: 25px;
            text-decoration: none;
            font-weight: 700;
            font-size: 0.95rem;
            box-shadow: 0 4px 15px rgba(66,133,244,0.4);
            transition: all 0.25s;
            margin-top: 10px;
        }}
        .gmaps-link-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(66,133,244,0.6);
            color: white;
        }}
        
        /* Site Recommendation Modal & Pins */
        .site-rec-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(4, 2, 14, 0.78);
            backdrop-filter: blur(8px);
            z-index: 9999;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }}
        .site-rec-card {{
            background: linear-gradient(145deg, rgba(22, 17, 44, 0.98), rgba(13, 10, 28, 0.98));
            border: 1px solid rgba(56, 189, 248, 0.45);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.85), 0 0 35px rgba(56, 189, 248, 0.2);
            border-radius: 20px;
            max-width: 840px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            padding: 28px;
            color: #f1f5f9;
            position: relative;
            animation: modalFadeIn 0.25s ease-out;
        }}
        @keyframes modalFadeIn {{
            from {{ opacity: 0; transform: scale(0.96) translateY(10px); }}
            to {{ opacity: 1; transform: scale(1) translateY(0); }}
        }}
        .rec-winner-box {{
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(129, 140, 248, 0.15));
            border: 1px solid rgba(56, 189, 248, 0.5);
            border-radius: 14px;
            padding: 20px;
            margin-top: 16px;
            display: flex;
            align-items: flex-start;
            gap: 18px;
        }}
        .rec-score-pill {{
            font-size: 1.55rem;
            font-weight: 800;
            color: #38bdf8;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid rgba(56, 189, 248, 0.4);
            padding: 8px 14px;
            border-radius: 12px;
            text-align: center;
            min-width: 85px;
        }}
        .rec-rank-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 10px;
            padding: 10px 14px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .rec-rank-item:hover {{
            background: rgba(56, 189, 248, 0.08);
            border-color: rgba(56, 189, 248, 0.3);
            transform: translateX(3px);
        }}
        .rec-bar-bg {{
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            margin: 0 14px;
            overflow: hidden;
        }}
        .rec-bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            border-radius: 4px;
        }}
        .inspect-pulse-pin {{
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            animation: pinBounce 0.9s infinite alternate;
        }}
        @keyframes pinBounce {{
            from {{ transform: translateY(0); }}
            to {{ transform: translateY(-8px); }}
        }}
        
        /* Loading */
        .loading {{ display: none; text-align: center; padding: 45px; }}
        .loading.show {{ display: block; }}
        .spinner {{
            width: 46px; height: 46px;
            border: 4px solid rgba(255,255,255,0.1);
            border-top: 4px solid #38bdf8;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px;
        }}
        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
        
        /* Results */
        #results {{ display: none; }}
        #results.show {{ display: block; animation: fadeIn 0.4s ease-in; }}
        @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: translateY(0); }} }}
        
        .result-header {{
            background: linear-gradient(135deg, rgba(56,189,248,0.14), rgba(129,140,248,0.14));
            border: 1px solid rgba(56,189,248,0.35);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            margin-bottom: 25px;
        }}
        .location-badge {{
            display: inline-block;
            background: rgba(255,215,0,0.18);
            border: 1px solid #ffd700;
            color: #ffd700;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 1.15rem;
            font-weight: 700;
            margin: 8px 0;
        }}
        
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 16px;
            margin-bottom: 25px;
        }}
        .card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 18px;
            text-align: center;
        }}
        .card h3 {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px; }}
        .card .value {{ font-size: 1.45rem; font-weight: 700; }}
        .card .value.green {{ color: #4ade80; }}
        .card .value.blue {{ color: #38bdf8; }}
        .card .value.yellow {{ color: #facc15; }}
        .card .sub {{ font-size: 0.78rem; color: #64748b; margin-top: 4px; }}
        
        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        @media (max-width: 900px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
        
        .section {{
            background: rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 22px;
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 25px;
        }}
        .section h2 {{ font-size: 1.15rem; margin-bottom: 15px; color: #f1f5f9; }}
        
        .plot-container {{ text-align: center; }}
        .plot-container img {{
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 6px 25px rgba(0,0,0,0.5);
        }}
        
        .weight-bar {{ display: flex; align-items: center; margin: 7px 0; }}
        .weight-bar .label {{ width: 170px; font-size: 0.85rem; color: #cbd5e1; }}
        .weight-bar .bar-bg {{
            flex: 1; height: 22px; background: rgba(255,255,255,0.08);
            border-radius: 11px; overflow: hidden; margin: 0 10px;
        }}
        .weight-bar .bar-fill {{
            height: 100%; border-radius: 11px;
            background: linear-gradient(90deg, #facc15, #f87171);
            display: flex; align-items: center; justify-content: flex-end;
            padding-right: 8px; font-size: 0.75rem; font-weight: 700; color: #0b091a;
        }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.06); }}
        th {{ color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
        
        .footer {{ text-align: center; padding: 25px; color: #64748b; font-size: 0.82rem; }}
        
        /* Custom Marker pulse */
        .optimal-pulse {{
            background: rgba(255, 215, 0, 0.7);
            border: 3px solid #fff;
            border-radius: 50%;
            animation: pulse 1.5s infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.7); }}
            70% {{ box-shadow: 0 0 0 20px rgba(255, 215, 0, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }}
        }}
        
        /* AI Building Prediction Section */
        .building-specs-section {{
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.5), rgba(15, 23, 42, 0.7));
            border: 1px solid rgba(129, 140, 248, 0.35);
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
        }}
        .specs-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
            flex-wrap: wrap;
            gap: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            padding-bottom: 12px;
        }}
        .specs-header h2 {{
            font-size: 1.25rem;
            color: #a5b4fc;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .specs-tag {{
            background: rgba(129, 140, 248, 0.18);
            border: 1px solid #818cf8;
            color: #c7d2fe;
            font-size: 0.78rem;
            padding: 5px 14px;
            border-radius: 20px;
            font-weight: 600;
        }}
        .specs-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 16px;
        }}
        .spec-card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 18px;
            position: relative;
            overflow: hidden;
            transition: all 0.25s;
        }}
        .spec-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(129, 140, 248, 0.4);
            box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        }}
        .spec-card .icon-title {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.82rem;
            color: #94a3b8;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        .spec-card .main-val {{
            font-size: 1.55rem;
            font-weight: 700;
            color: #f8fafc;
            line-height: 1.2;
        }}
        .spec-card .main-val span {{
            font-size: 0.82rem;
            font-weight: 500;
            color: #94a3b8;
            margin-left: 4px;
        }}
        .spec-card .detail-list {{
            margin-top: 12px;
            font-size: 0.8rem;
            color: #cbd5e1;
            line-height: 1.55;
            border-top: 1px dashed rgba(255,255,255,0.08);
            padding-top: 10px;
        }}
        .spec-card .detail-item {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            gap: 8px;
        }}
        .spec-card .detail-label {{ color: #94a3b8; }}
        .spec-card .detail-val {{ font-weight: 600; color: #f1f5f9; text-align: right; }}
    </style>
</head>
<body>

<div class="header">
    <h1>Multi-Criteria Spatial Optimization &amp; Geospatial Siting System</h1>
    <p>Matale District, Sri Lanka | Real-Time Google &amp; Satellite Map Integration | 8 AHP Criteria</p>
</div>

<div class="container">
    
    <!-- Step 1 & 2 Controls -->
    <div class="control-panel">
        <div class="panel-card">
            <h2><span>1.</span> What do you want to build?</h2>
            <div class="subtitle">Select facility to dynamically compute 8x8 AHP criteria weights</div>
            <div class="infra-buttons">
                {infra_buttons_html}
            </div>
            
            <!-- School Subtypes Panel -->
            <div id="school-subtypes" class="school-subtypes-container">
                <div class="subtype-header">
                    <span>🎓 School Classification:</span>
                </div>
                <div class="subtype-pills">
                    <button class="subtype-pill active" onclick="selectSchoolSubtype('national_school')" id="sub-national_school">
                        ⭐ National School / Central College (Type 1AB)
                    </button>
                    <button class="subtype-pill" onclick="selectSchoolSubtype('primary_school')" id="sub-primary_school">
                        🧒 Primary School (Grade 1-5)
                    </button>
                    <button class="subtype-pill" onclick="selectSchoolSubtype('secondary_school')" id="sub-secondary_school">
                        📚 Provincial Secondary School (Type 1C)
                    </button>
                    <button class="subtype-pill" onclick="selectSchoolSubtype('girls_school')" id="sub-girls_school">
                        🌸 Girls' High School
                    </button>
                    <button class="subtype-pill" onclick="selectSchoolSubtype('boys_school')" id="sub-boys_school">
                        🦁 Boys' College
                    </button>
                </div>
            </div>
        </div>
        
        <div class="panel-card">
            <h2><span>2.</span> Select Geographic Sub-Zone</h2>
            <div class="subtitle">Select administrative zone OR drag custom square on the map below</div>
            
            <div class="zone-presets">
                {region_buttons_html}
            </div>
            
            <div style="background: rgba(0,0,0,0.25); border-radius: 12px; padding: 12px; border: 1px dashed rgba(255,255,255,0.15);">
                <div style="font-size:0.8rem; color:#38bdf8; margin-bottom:8px;">Custom Grid Square Bounds (0 - 200):</div>
                <div class="coords-grid">
                    <div class="coord-field">
                        <label>Min X (Row):</label>
                        <input type="number" id="roi-min-x" value="0" min="0" max="190">
                    </div>
                    <div class="coord-field">
                        <label>Max X (Row):</label>
                        <input type="number" id="roi-max-x" value="200" min="10" max="200">
                    </div>
                    <div class="coord-field">
                        <label>Min Y (Col):</label>
                        <input type="number" id="roi-min-y" value="0" min="0" max="190">
                    </div>
                    <div class="coord-field">
                        <label>Max Y (Col):</label>
                        <input type="number" id="roi-max-y" value="200" min="10" max="200">
                    </div>
                </div>
            </div>
            
            <button class="run-btn" onclick="executeSearch()">
                ⚡ Find Optimal Location in Selected Area
            </button>
        </div>
    </div>
    
    <!-- Real Interactive Google / Satellite Map & 3D WebGL Terrain -->
    <div class="map-section">
        <div class="map-header">
            <div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">
                <h2>🗺️ Geospatial Terrain &amp; Site Exploration</h2>
                <!-- 2D / 3D View Switcher Tabs -->
                <div class="map-view-tabs">
                    <button class="view-tab active" onclick="switchMapView('google-hybrid')" id="tab-google-hybrid">
                        🟢 Google Satellite &amp; Roads
                    </button>
                    <button class="view-tab" onclick="switchMapView('google-streets')" id="tab-google-streets">
                        🟡 Google Streets
                    </button>
                    <button class="view-tab" onclick="switchMapView('esri-satellite')" id="tab-esri-satellite">
                        🛰️ Esri Satellite
                    </button>
                    <button class="view-tab" onclick="switchMapView('3d')" id="tab-3d">
                        🏔️ 3D WebGL Terrain View
                    </button>
                </div>
                <span style="background:rgba(56,189,248,0.12); border:1px solid rgba(56,189,248,0.35); color:#38bdf8; font-size:0.78rem; font-weight:600; padding:5px 12px; border-radius:20px; display:inline-flex; align-items:center; gap:6px;">
                    🎯 Click anywhere on 2D or 3D map to recommend best infrastructure!
                </span>
            </div>
            <div class="map-controls" id="map-2d-controls">
                <div class="map-ctrl-item">
                    <label>Heatmap Opacity:</label>
                    <input type="range" id="opacity-slider" min="0" max="1" step="0.05" value="0.65" oninput="updateOverlayOpacity(this.value)">
                </div>
                <div class="map-ctrl-item">
                    <input type="checkbox" id="toggle-schools" checked onchange="toggleFacilityLayer('schools', this.checked)">
                    <label for="toggle-schools" style="cursor:pointer;">🏫 Existing Schools (118)</label>
                </div>
                <div class="map-ctrl-item">
                    <input type="checkbox" id="toggle-hospitals" checked onchange="toggleFacilityLayer('hospitals', this.checked)">
                    <label for="toggle-hospitals" style="cursor:pointer;">🏥 Existing Hospitals (43)</label>
                </div>
                <div class="map-ctrl-item">
                    <input type="checkbox" id="toggle-utilities" checked onchange="toggleFacilityLayer('utilities', this.checked)">
                    <label for="toggle-utilities" style="cursor:pointer;">⛽ Public Utilities (225)</label>
                </div>
                <div class="map-ctrl-item">
                    <input type="checkbox" id="toggle-waterways" checked onchange="toggleFacilityLayer('waterways', this.checked)">
                    <label for="toggle-waterways" style="cursor:pointer; color:#38bdf8; font-weight:600;">🌊 River Flood Buffers (749)</label>
                </div>
            </div>
        </div>
        
        <!-- 2D Leaflet Map -->
        <div id="leaflet-map"></div>
        
        <!-- 3D Three.js Terrain Container -->
        <div id="threejs-container" style="display:none; position:relative; width:100%; height:540px; border-radius:14px; overflow:hidden; background:#070512; border:1px solid rgba(255,255,255,0.12);">
            <div id="threejs-canvas" style="width:100%; height:100%;"></div>
            
            <!-- 3D Floating Control Bar -->
            <div class="three-toolbar">
                <div class="three-ctrl">
                    <label>🏔️ Z-Exaggeration:</label>
                    <input type="range" id="z-scale-slider" min="0.5" max="4.0" step="0.2" value="1.8" oninput="update3DScale(this.value)">
                    <span id="z-scale-val">1.8x</span>
                </div>
                <div class="three-ctrl">
                    <label>🎨 Surface:</label>
                    <select id="three-texture-mode" onchange="change3DTexture(this.value)">
                        <option value="suitability">Suitability Heatmap</option>
                        <option value="topography">NASA SRTM Topo Gradient</option>
                        <option value="wireframe">3D Wireframe Mesh</option>
                    </select>
                </div>
                <button class="three-btn" onclick="reset3DCamera()">🔄 Reset 3D View</button>
                <div class="three-help">🖱️ Left Drag: Rotate 360° | Right Drag: Pan | Scroll: Zoom</div>
            </div>
        </div>
    </div>
    
    <!-- Loading Spinner -->
    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div style="font-size:1.1rem; color:#38bdf8;">Running Metaheuristic Optimization...</div>
        <div style="font-size:0.85rem; margin-top:8px; color:#94a3b8;">
            Evaluating terrain, roads, schools, hospitals, transport → Localized GA &amp; PSO search
        </div>
    </div>
    
    <!-- Results -->
    <div id="results">
        <div class="result-header">
            <h2 id="result-title">Optimal Location Identified!</h2>
            <div class="location-badge" id="result-badge">Grid (0.0, 0.0)</div>
            <div style="color:#e2e8f0; font-size:1rem; margin-top:6px;" id="result-coords"></div>
            <div style="color:#94a3b8; font-size:0.9rem; margin-top:6px;" id="result-details"></div>
            
            <!-- Direct Google Maps link -->
            <div style="margin-top: 14px;">
                <a id="gmaps-btn" href="#" target="_blank" class="gmaps-link-btn">
                    🗺️ Open Location in Google Maps
                </a>
            </div>
        </div>
        
        <div class="cards" id="result-cards"></div>
        
        <!-- Building Specs Prediction Section -->
        <div class="building-specs-section">
            <div class="specs-header">
                <h2>🏗️ AI Predicted Building Specifications &amp; Sizing</h2>
                <span class="specs-tag">Micro-Level Architectural &amp; Capacity AI</span>
            </div>
            <div class="specs-grid" id="specs-grid">
                <!-- Dynamically populated via JS -->
            </div>
        </div>
        
        <div class="two-col">
            <div class="section">
                <h2>Suitability Heatmap &amp; Selected Squad</h2>
                <div class="plot-container">
                    <img id="heatmap-img" src="" alt="Suitability Heatmap">
                </div>
            </div>
            <div class="section">
                <h2>Convergence Curves (GA vs PSO)</h2>
                <div class="plot-container">
                    <img id="convergence-img" src="" alt="Convergence Curves">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>8 Criteria AHP Weights for this Facility</h2>
            <div id="weights-bars"></div>
        </div>
        
        <div class="section">
            <h2>Algorithm Benchmark Comparison (Inside Selected Zone)</h2>
            <table>
                <thead>
                    <tr><th>Metric</th><th style="color:#f87171">Genetic Algorithm (GA)</th><th style="color:#38bdf8">Particle Swarm Optimization (PSO)</th></tr>
                </thead>
                <tbody id="benchmark-table"></tbody>
            </table>
        </div>
    </div>
    
</div>

<div class="footer">
    Component 4: Multi-Criteria Spatial Optimization &amp; Google Maps GIS Integration | Matale District, Sri Lanka
</div>

<!-- Interactive Site Suitability & Highest/Best Use (HBU) Modal -->
<div id="site-recommendation-modal" class="site-rec-modal" onclick="if(event.target===this) closeSiteModal()">
    <div class="site-rec-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid rgba(255,255,255,0.12); padding-bottom:14px; margin-bottom:16px;">
            <div>
                <div style="display:inline-flex; align-items:center; gap:8px; font-size:0.8rem; font-weight:700; color:#38bdf8; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">
                    🎯 Highest &amp; Best Use (HBU) Analysis | Reverse Siting Engine
                </div>
                <h2 style="font-size:1.35rem; margin:0; color:#fff;" id="rec-site-title">
                    📍 Analyzing Selected Site...
                </h2>
            </div>
            <button onclick="closeSiteModal()" style="background:rgba(255,255,255,0.08); border:none; color:#cbd5e1; font-size:1.4rem; border-radius:50%; width:36px; height:36px; cursor:pointer; display:flex; align-items:center; justify-content:center; transition:0.2s;" onmouseover="this.style.background='rgba(239,68,68,0.25)'" onmouseout="this.style.background='rgba(255,255,255,0.08)'">&times;</button>
        </div>

        <!-- Site Context Badges -->
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:18px;" id="rec-site-badges">
            <span class="location-badge" style="font-size:0.9rem; padding:4px 12px;">Loading terrain telemetry...</span>
        </div>

        <!-- Content Area: Winner + Specs + Ranked Alternatives -->
        <div id="rec-modal-content">
            <div style="text-align:center; padding:40px;">
                <div class="spinner"></div>
                <p style="color:#94a3b8; font-weight:600;">Evaluating multi-criteria suitability (NASA SRTM, WorldPop, OSM)...</p>
            </div>
        </div>
    </div>
</div>

<script>
let currentPrimaryInfra = 'school';
let currentSchoolSubtype = 'national_school';
let currentInfra = 'national_school';
let currentZoneLabel = 'Full Matale District';
let leafletMap = null;
let currentOverlayLayer = null;
let currentOptimalMarker = null;
let currentRoiRectangle = null;
let schoolsLayerGroup = null;
let hospitalsLayerGroup = null;
let utilitiesLayerGroup = null;
let waterwaysLayerGroup = null;
let googleHybridLayer = null;
let googleStreetLayer = null;
let googleTerrainLayer = null;
let satelliteLayer = null;
let streetLayer = null;
let darkLayer = null;

// 3D WebGL Terrain State
let currentViewMode = '2d';
let lastOptimizationData = null;
let threeScene = null, threeCamera = null, threeRenderer = null, threeControls = null;
let terrainMesh = null, optimal3DPin = null;
let currentZScale = 1.8;
let current3DTextureMode = 'suitability';

// Reverse Siting & Highest/Best Use (HBU) State
let inspectMarker = null;
let inspect3DPin = null;
let lastRecommendationData = null;

function closeSiteModal() {{
    const modal = document.getElementById('site-recommendation-modal');
    if (modal) modal.style.display = 'none';
}}

function inspectFacility(type, idx) {{
    let fac = null;
    if (type === 'school' && FACILITIES_DATA.schools) fac = FACILITIES_DATA.schools[idx];
    else if (type === 'hospital' && FACILITIES_DATA.hospitals) fac = FACILITIES_DATA.hospitals[idx];
    else if (type === 'utility' && FACILITIES_DATA.utilities) fac = FACILITIES_DATA.utilities[idx];
    
    if (fac) {{
        inspectLocation(fac.lat, fac.lon, null, fac.name, type);
    }}
}}

function inspectLocation(lat, lon, p3d, facilityName, facilityType) {{
    if (!leafletMap) return;
    facilityName = facilityName || '';
    facilityType = facilityType || '';
    
    // 1. Move/place 2D inspect marker
    if (inspectMarker) {{
        leafletMap.removeLayer(inspectMarker);
    }}
    inspectMarker = L.marker([lat, lon], {{
        icon: L.divIcon({{
            className: '',
            html: '<div class="inspect-pulse-pin">📍</div>',
            iconSize: [32, 32],
            iconAnchor: [16, 30]
        }})
    }}).addTo(leafletMap);
    
    // 2. Move/place 3D inspect pin if Three.js active
    if (threeScene) {{
        if (inspect3DPin) {{
            threeScene.remove(inspect3DPin);
            inspect3DPin = null;
        }}
        if (p3d) {{
            inspect3DPin = new THREE.Group();
            const poleGeo = new THREE.CylinderGeometry(0.6, 0.6, 20, 16);
            const poleMat = new THREE.MeshBasicMaterial({{ color: 0x38bdf8 }});
            const pole = new THREE.Mesh(poleGeo, poleMat);
            pole.position.y = 10;
            inspect3DPin.add(pole);
            
            const headGeo = new THREE.SphereGeometry(3.5, 20, 20);
            const headMat = new THREE.MeshStandardMaterial({{
                color: 0x38bdf8,
                emissive: 0x0284c7,
                emissiveIntensity: 0.9
            }});
            const head = new THREE.Mesh(headGeo, headMat);
            head.position.y = 20;
            inspect3DPin.add(head);
            
            inspect3DPin.position.set(p3d.x, p3d.y, p3d.z);
            threeScene.add(inspect3DPin);
            if (threeRenderer && threeScene && threeCamera) threeRenderer.render(threeScene, threeCamera);
        }}
    }}
    
    // 3. Show modal with loading state
    const modal = document.getElementById('site-recommendation-modal');
    modal.style.display = 'flex';
    
    if (facilityName) {{
        document.getElementById('rec-site-title').innerHTML = `🔬 Operational Facility Audit: <span style="color:#38bdf8;">${{facilityName}}</span>`;
        document.getElementById('rec-site-badges').innerHTML = `
            <span class="location-badge" style="background:rgba(56,189,248,0.2); border-color:#38bdf8; color:#38bdf8; font-size:0.85rem; padding:4px 12px;">🏛️ Retrospective GIS Audit</span>
            <span class="location-badge" style="font-size:0.85rem; padding:4px 12px;">🛰️ Querying NASA DEM &amp; WorldPop...</span>
        `;
    }} else {{
        document.getElementById('rec-site-title').innerHTML = `📍 Site: ${{lat.toFixed(4)}}°N, ${{lon.toFixed(4)}}°E`;
        document.getElementById('rec-site-badges').innerHTML = `
            <span class="location-badge" style="font-size:0.85rem; padding:4px 12px;">🛰️ Querying NASA SRTM &amp; WorldPop...</span>
        `;
    }}
    
    document.getElementById('rec-modal-content').innerHTML = `
        <div style="text-align:center; padding:40px;">
            <div class="spinner"></div>
            <p style="color:#94a3b8; font-weight:600; font-size:1rem;">Evaluating ${{facilityName ? `"${{facilityName}}" against MoE &amp; NBRO criteria` : 'site suitability across 8 criteria &amp; 9 infrastructure types'}}...</p>
            <p style="color:#64748b; font-size:0.85rem;">Checking NBRO landslide thresholds, CEA environmental reserves, and road connectivity</p>
        </div>
    `;
    
    // 4. Fetch /api/recommend-site with name & type
    const qUrl = `/api/recommend-site?lat=${{lat}}&lon=${{lon}}&name=${{encodeURIComponent(facilityName)}}&type=${{encodeURIComponent(facilityType)}}`;
    fetch(qUrl)
        .then(res => res.json())
        .then(data => {{
            lastRecommendationData = data;
            renderSiteRecommendationModal(data);
        }})
        .catch(err => {{
            document.getElementById('rec-modal-content').innerHTML = `
                <div style="background:rgba(239,68,68,0.15); border:1px solid #ef4444; border-radius:12px; padding:18px; color:#fca5a5;">
                    ⚠️ <strong>Evaluation Failed:</strong> ${{err.message || err}}
                </div>
            `;
        }});
}}

function renderSiteRecommendationModal(data) {{
    if (data.error) {{
        document.getElementById('rec-site-title').innerText = 'Location Warning';
        document.getElementById('rec-modal-content').innerHTML = `
            <div style="background:rgba(239,68,68,0.15); border:1px solid #ef4444; border-radius:12px; padding:18px; color:#fca5a5;">
                ⚠️ ${{data.error}}
            </div>
        `;
        return;
    }}
    
    const latStr = data.lat.toFixed(4);
    const lonStr = data.lon.toFixed(4);
    
    if (data.facility_audit) {{
        document.getElementById('rec-site-title').innerHTML = `🔬 Operational Facility Audit: <span style="color:#38bdf8;">${{data.facility_audit.facility_name}}</span>`;
    }} else {{
        document.getElementById('rec-site-title').innerHTML = `📍 Selected Site: ${{latStr}}°N, ${{lonStr}}°E`;
    }}
    
    // Badges
    const slopeClass = data.slope_deg <= 8.0 ? 'background:rgba(34,197,94,0.18); border-color:#22c55e; color:#4ade80;' :
                       data.slope_deg <= 15.0 ? 'background:rgba(234,179,8,0.18); border-color:#eab308; color:#facc15;' :
                       'background:rgba(239,68,68,0.18); border-color:#ef4444; color:#f87171;';
    const slopeLabel = data.slope_deg <= 8.0 ? 'Gentle / Safe' : data.slope_deg <= 15.0 ? 'Moderate Slope' : 'Steep / High Hazard';
    
    let buildableBadge = '';
    if (data.is_buildable && data.slope_deg <= 20.0) {{
        buildableBadge = `<span class="location-badge" style="background:rgba(34,197,94,0.18); border-color:#22c55e; color:#4ade80; font-size:0.85rem; padding:4px 12px;">✅ NBRO Buildable Area</span>`;
    }} else {{
        const hazard = (data.constraint_details && data.constraint_details.primary_hazard) ? data.constraint_details.primary_hazard : null;
        let bText = '⚠️ Restricted by Regulations';
        if (hazard === 'RIVER_FLOOD') bText = '🌊 60m River Flood Buffer (Prohibited)';
        else if (hazard === 'WATER_BODY') bText = '💧 Water Body / Wetland (Prohibited)';
        else if (hazard === 'PROTECTED_FOREST') bText = '🌲 Protected Forest Reserve (Prohibited)';
        else if (hazard === 'SACRED_PRECINCT') bText = '🏛️ Sacred Precinct Buffer (Prohibited)';
        else if (hazard === 'LANDSLIDE') bText = '⚠️ Severe Landslide Risk (Slope > 20°)';
        buildableBadge = `<span class="location-badge" style="background:rgba(239,68,68,0.25); border-color:#ef4444; color:#f87171; font-size:0.85rem; padding:4px 12px;">${{bText}}</span>`;
    }}
        
    const bldgBadge = (data.predicted_specs && data.predicted_specs.neighborhood_buildings) ? 
        `<span class="location-badge" style="background:rgba(168,85,247,0.18); border-color:#a855f7; color:#c084fc; font-size:0.85rem; padding:4px 12px;">🏢 <strong>${{data.predicted_specs.neighborhood_buildings}}</strong> Buildings in Cell</span>` : '';
    const solarBadge = (data.predicted_specs && data.predicted_specs.empirical_solar_ghi_kwh_m2_yr) ? 
        `<span class="location-badge" style="background:rgba(250,204,21,0.18); border-color:#facc15; color:#fde047; font-size:0.85rem; padding:4px 12px;">☀️ Solar: <strong>${{data.predicted_specs.empirical_solar_ghi_kwh_m2_yr}}</strong> kWh/m²/yr</span>` : '';

    document.getElementById('rec-site-badges').innerHTML = `
        <span class="location-badge" style="font-size:0.85rem; padding:4px 12px;">🏔️ Elevation: <strong>${{data.elevation}}m</strong></span>
        <span class="location-badge" style="${{slopeClass}} font-size:0.85rem; padding:4px 12px;">📐 Slope: <strong>${{data.slope_deg}}° (${{slopeLabel}})</strong></span>
        ${{buildableBadge}}
        ${{bldgBadge}}
        ${{solarBadge}}
        <a href="https://www.google.com/maps?q=${{data.lat}},${{data.lon}}" target="_blank" class="gmaps-link-btn" style="padding:4px 12px; font-size:0.8rem; margin:0; border-radius:14px;">🗺️ Google Maps ↗</a>
    `;
    
    // Warning box if restricted
    let restrictionHtml = '';
    if ((!data.is_buildable || data.slope_deg > 20.0) && data.restriction_reasons && data.restriction_reasons.length > 0) {{
        restrictionHtml = `
            <div style="background:rgba(239,68,68,0.14); border:1px solid #ef4444; border-radius:12px; padding:14px 18px; margin-bottom:18px;">
                <div style="display:flex; align-items:center; gap:8px; font-weight:800; color:#fca5a5; font-size:0.95rem; margin-bottom:6px;">
                    <span>🚫</span> <span>National Statutory &amp; Physical Restriction Checklist:</span>
                </div>
                <ul style="margin:0; padding-left:20px; color:#fecaca; font-size:0.86rem; line-height:1.5;">
                    ${{data.restriction_reasons.map(r => `<li><strong>${{r}}</strong></li>`).join('')}}
                </ul>
            </div>
        `;
    }}
    
    let mainCardHtml = '';
    
    // Check if this is a dedicated retrospective facility audit
    if (data.facility_audit) {{
        const audit = data.facility_audit;
        const auditSpecs = audit.predicted_specs;
        
        mainCardHtml = `
            <div class="rec-winner-box" style="border:1px solid ${{audit.grade_color}}; background:rgba(15,23,42,0.75);">
                <div class="rec-score-pill" style="border-color:${{audit.grade_color}}; min-width:95px;">
                    <div style="font-size:1.45rem; font-weight:800; color:${{audit.grade_color}};">${{audit.score}}%</div>
                    <div style="font-size:0.75rem; color:${{audit.grade_color}}; font-weight:700; text-transform:uppercase;">${{audit.grade}}</div>
                </div>
                <div style="flex:1;">
                    <div style="display:inline-flex; align-items:center; gap:6px; background:rgba(56,189,248,0.18); border:1px solid #38bdf8; color:#38bdf8; font-size:0.75rem; font-weight:800; padding:2px 10px; border-radius:12px; text-transform:uppercase; margin-bottom:6px;">
                        🏛️ Retrospective Operational Facility Audit
                    </div>
                    <h3 style="font-size:1.35rem; margin:0 0 4px 0; color:#fff;">
                        ${{audit.emoji}} ${{audit.facility_name}}
                    </h3>
                    <p style="color:#94a3b8; font-size:0.85rem; margin:0 0 10px 0;">
                        Operational Siting Diagnostic: <strong style="color:${{audit.grade_color}};">${{audit.grade_desc}}</strong>
                    </p>
                    
                    <!-- National Standards & NBRO Compliance Checklist -->
                    <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:6px; margin-bottom:12px;">
                        ${{audit.compliance.map(c => `
                            <div style="background:rgba(255,255,255,0.03); border:1px solid ${{c.passed ? 'rgba(74,222,128,0.35)' : 'rgba(239,68,68,0.35)'}}; border-radius:8px; padding:6px 10px; font-size:0.8rem;">
                                <div style="display:flex; justify-content:space-between; align-items:center; font-weight:700; color:${{c.passed ? '#4ade80' : '#f87171'}};">
                                    <span>${{c.standard}}</span>
                                    <span>${{c.passed ? '✅ ' + c.status : '⚠️ ' + c.status}}</span>
                                </div>
                                <div style="color:#cbd5e1; font-size:0.75rem; margin-top:2px;">${{c.detail}}</div>
                            </div>
                        `).join('')}}
                    </div>

                    <!-- Strengths and Operational Recommendations -->
                    <div style="display:flex; flex-direction:column; gap:4px; margin-bottom:12px;">
                        ${{audit.strengths.map(s => `<div style="color:#4ade80; font-size:0.82rem; display:flex; align-items:flex-start; gap:6px;"><span>✅</span> <span>${{s}}</span></div>`).join('')}}
                        ${{audit.recommendations.map(r => `<div style="color:#38bdf8; font-size:0.82rem; display:flex; align-items:flex-start; gap:6px;"><span>💡</span> <span>${{r}}</span></div>`).join('')}}
                    </div>
                    
                    <!-- Asset Potential (Solar & Capacity) -->
                    ${{auditSpecs ? `
                        <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:10px; display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:8px;">
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Population Demand:</span><br><strong style="color:#38bdf8; font-size:0.92rem;">${{data.criteria_breakdown['Population Density']}}% Demand</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Road Access Index:</span><br><strong style="color:#fff; font-size:0.92rem;">${{data.criteria_breakdown['Road Proximity']}}% Access</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Rooftop Solar PV:</span><br><strong style="color:#facc15; font-size:0.92rem;">${{auditSpecs.solar_energy.solar_capacity_kwp}} kWp</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Est. Replacement Value:</span><br><strong style="color:#4ade80; font-size:0.92rem;">LKR ${{auditSpecs.construction_cost.estimated_budget_million_lkr}}M</strong></div>
                        </div>
                    ` : ''}}
                </div>
            </div>
        `;
    }} else if (!data.is_buildable || data.slope_deg > 20.0) {{
        const cd = data.constraint_details || {{}};
        const hazardType = cd.primary_hazard || 'REGULATORY_RESTRICTION';
        let hazardEmoji = '🚫';
        let hazardTitle = cd.hazard_title || 'Strictly Prohibited Construction Zone';
        if (hazardType === 'RIVER_FLOOD') hazardEmoji = '🌊';
        else if (hazardType === 'WATER_BODY') hazardEmoji = '💧';
        else if (hazardType === 'PROTECTED_FOREST') hazardEmoji = '🌲';
        else if (hazardType === 'SACRED_PRECINCT') hazardEmoji = '🏛️';
        else if (hazardType === 'LANDSLIDE') hazardEmoji = '⚠️';

        mainCardHtml = `
            <div class="rec-winner-box" style="border: 2px solid #ef4444; background: linear-gradient(135deg, rgba(239,68,68,0.20) 0%, rgba(15,23,42,0.94) 100%);">
                <div class="rec-score-pill" style="border-color:#ef4444; min-width:105px; padding:12px 14px;">
                    <div style="font-size:1.6rem; font-weight:800; color:#ef4444; line-height:1;">0.0%</div>
                    <div style="font-size:0.72rem; color:#fca5a5; font-weight:700; text-transform:uppercase; margin-top:4px;">DISQUALIFIED</div>
                    <div style="font-size:0.65rem; color:#ef4444; font-weight:600; margin-top:2px;">Penalty: -999.0</div>
                </div>
                <div style="flex:1;">
                    <div style="display:inline-flex; align-items:center; gap:6px; background:rgba(239,68,68,0.25); border:1px solid #ef4444; color:#fca5a5; font-size:0.75rem; font-weight:800; padding:3px 12px; border-radius:12px; text-transform:uppercase; margin-bottom:8px;">
                        ${{hazardEmoji}} ${{hazardTitle}}
                    </div>
                    <h3 style="font-size:1.35rem; margin:0 0 6px 0; color:#fee2e2;">
                        🚫 School / Public Infrastructure Prohibited at this Site
                    </h3>
                    <p style="color:#cbd5e1; font-size:0.86rem; margin:0 0 12px 0; line-height:1.45;">
                        This coordinate is <strong>permanently disqualified</strong> from new educational or public building construction. Developing permanent civil structures here violates statutory safety regulations and environmental conservation laws in Sri Lanka.
                    </p>
                    
                    <!-- Detailed Justification Grid -->
                    <div style="background:rgba(0,0,0,0.5); border:1px solid rgba(239,68,68,0.35); border-radius:10px; padding:14px; display:flex; flex-direction:column; gap:10px; font-size:0.83rem;">
                        <div>
                            <span style="color:#f87171; font-weight:700; display:flex; align-items:center; gap:5px;">
                                <span>🌊</span> <span>Physical &amp; Environmental Hazard:</span>
                            </span>
                            <div style="color:#e2e8f0; font-size:0.82rem; margin-top:3px; line-height:1.4;">
                                ${{cd.risk_description || 'Active physical or geological hazard zone.'}}
                            </div>
                        </div>
                        <div>
                            <span style="color:#f87171; font-weight:700; display:flex; align-items:center; gap:5px;">
                                <span>🏫</span> <span>School Siting &amp; Student Life Safety Justification:</span>
                            </span>
                            <div style="color:#e2e8f0; font-size:0.82rem; margin-top:3px; line-height:1.4;">
                                ${{cd.school_safety_reason || 'Student life safety and evacuation hazard.'}}
                            </div>
                        </div>
                        <div>
                            <span style="color:#f87171; font-weight:700; display:flex; align-items:center; gap:5px;">
                                <span>⚖️</span> <span>Statutory Ordinances &amp; Regulations:</span>
                            </span>
                            <div style="color:#fca5a5; font-size:0.82rem; margin-top:3px; line-height:1.4;">
                                ${{cd.statutory_law || 'Sri Lanka statutory planning guidelines.'}}
                            </div>
                        </div>
                        <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:8px;">
                            <span style="color:#38bdf8; font-weight:700; display:flex; align-items:center; gap:5px;">
                                <span>🤖</span> <span>Algorithmic Optimization Handling (GA &amp; PSO):</span>
                            </span>
                            <div style="color:#93c5fd; font-size:0.82rem; margin-top:3px; line-height:1.4;">
                                Hard Constraint Mask <code>C(x, y) = 0.0</code> zeroes out the suitability score (<code>S = S_AHP &times; 0.0 = 0.0%</code>). In both the <strong>Genetic Algorithm</strong> and <strong>Particle Swarm Optimizer</strong>, any chromosome or particle landing in this zone receives a lethal <strong>-999.0 Death Penalty</strong>, mathematically guaranteeing the AI will never select it.
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }} else {{
        const winner = data.top_recommendation;
        const specs = data.predicted_specs;
        
        mainCardHtml = `
            <div class="rec-winner-box">
                <div class="rec-score-pill">
                    <div>${{winner.score}}%</div>
                    <div style="font-size:0.7rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Match</div>
                </div>
                <div style="flex:1;">
                    <div style="display:inline-flex; align-items:center; gap:6px; background:rgba(255,215,0,0.18); border:1px solid #ffd700; color:#ffd700; font-size:0.75rem; font-weight:800; padding:2px 10px; border-radius:12px; text-transform:uppercase; margin-bottom:6px;">
                        🏆 #1 Best Recommended Infrastructure
                    </div>
                    <h3 style="font-size:1.4rem; margin:0 0 6px 0; color:#fff;">
                        ${{winner.emoji}} ${{winner.label}}
                    </h3>
                    <p style="color:#94a3b8; font-size:0.85rem; margin:0 0 10px 0;">${{winner.description}}</p>
                    
                    <!-- Strengths & Weaknesses -->
                    <div style="display:flex; flex-direction:column; gap:4px; margin-bottom:12px;">
                        ${{winner.strengths.map(s => `<div style="color:#4ade80; font-size:0.82rem; display:flex; align-items:center; gap:6px;"><span>✅</span> <span>${{s}}</span></div>`).join('')}}
                        ${{winner.weaknesses.map(w => `<div style="color:#f87171; font-size:0.82rem; display:flex; align-items:center; gap:6px;"><span>⚠️</span> <span>${{w}}</span></div>`).join('')}}
                    </div>
                    
                    <!-- Predicted Architectural & Capacity Specs -->
                    ${{specs && specs.capacity ? `
                        <div style="background:rgba(0,0,0,0.3); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:12px; display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:8px;">
                            <div><span style="font-size:0.72rem; color:#94a3b8;">${{specs.capacity.unit || 'Capacity'}}:</span><br><strong style="color:#38bdf8; font-size:0.95rem;">${{Number(specs.capacity.value).toLocaleString()}}</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Building Layout:</span><br><strong style="color:#fff; font-size:0.95rem;">${{specs.architectural.recommended_floors}} Floors</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Ground Footprint:</span><br><strong style="color:#fff; font-size:0.95rem;">${{specs.architectural.ground_footprint_m2}} m²</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Solar Energy:</span><br><strong style="color:#facc15; font-size:0.95rem;">${{specs.solar_energy.solar_capacity_kwp}} kWp</strong></div>
                            <div><span style="font-size:0.72rem; color:#94a3b8;">Est. Budget:</span><br><strong style="color:#4ade80; font-size:0.95rem;">LKR ${{specs.construction_cost.estimated_budget_million_lkr}}M</strong></div>
                        </div>
                    ` : ''}}
                </div>
            </div>
        `;
    }}
    
    // Ranked List HTML
    let rankingsHtml = '';
    if (!data.is_buildable || data.slope_deg > 20.0) {{
        rankingsHtml = `
            <div style="margin-top:22px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                    <h4 style="margin:0; font-size:1.05rem; color:#fca5a5;">📊 All Infrastructure Types Disqualified (0.0% Match):</h4>
                    <span style="font-size:0.75rem; color:#94a3b8;">Click any row to inspect criteria</span>
                </div>
                <div style="background:rgba(239,68,68,0.08); border-left:3px solid #ef4444; padding:8px 12px; border-radius:0 6px 6px 0; margin-bottom:12px; font-size:0.82rem; color:#fecaca;">
                    💡 <strong>Multi-Criteria Constraint Override:</strong> Under Saaty AHP &amp; GIS spatial logic, when the Hard Constraint Mask is 0 (<code>C = 0</code>), total suitability <code>S = S_AHP &times; C = 0.0%</code> across all infrastructure types regardless of accessibility.
                </div>
                <div id="rec-rankings-container">
                    ${{data.rankings.map((item, idx) => `
                        <div class="rec-rank-item" onclick="toggleRankDetails(${{idx}})" style="opacity:0.85;">
                            <div style="display:flex; align-items:center; gap:10px; min-width:220px;">
                                <span style="font-size:1.15rem;">${{item.emoji}}</span>
                                <span style="font-weight:600; font-size:0.88rem; color:#cbd5e1;">${{item.label}}</span>
                            </div>
                            <div class="rec-bar-bg">
                                <div class="rec-bar-fill" style="width:0%; background:#ef4444;"></div>
                            </div>
                            <div style="font-weight:700; font-size:0.88rem; color:#f87171; min-width:95px; text-align:right;">
                                0.0% (Barred)
                            </div>
                        </div>
                        <div id="rec-details-${{idx}}" style="display:none; background:rgba(0,0,0,0.35); border-left:3px solid #ef4444; padding:10px 14px; margin:-4px 0 8px 10px; border-radius:0 8px 8px 0; font-size:0.82rem;">
                            <p style="color:#cbd5e1; margin:0 0 6px 0;">${{item.description}}</p>
                            <div style="color:#f87171; font-weight:700; margin-bottom:4px;">❌ Disqualified: Location overlaps statutory safety constraint buffer.</div>
                            ${{item.weaknesses.map(w => `<div style="color:#fca5a5;">⚠️ ${{w}}</div>`).join('')}}
                        </div>
                    `).join('')}}
                </div>
            </div>
        `;
    }} else {{
        rankingsHtml = `
            <div style="margin-top:22px;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <h4 style="margin:0; font-size:1.05rem; color:#e2e8f0;">${{data.facility_audit ? '📊 Comparative Multi-Sector Suitability at this Site:' : '📊 All Infrastructure Suitability Scores at this Site:'}}</h4>
                    <span style="font-size:0.75rem; color:#94a3b8;">Click any row to inspect why</span>
                </div>
                <div id="rec-rankings-container">
                    ${{data.rankings.map((item, idx) => `
                        <div class="rec-rank-item" onclick="toggleRankDetails(${{idx}})">
                            <div style="display:flex; align-items:center; gap:10px; min-width:220px;">
                                <span style="font-size:1.15rem;">${{item.emoji}}</span>
                                <span style="font-weight:600; font-size:0.88rem; color:${{idx === 0 ? '#38bdf8' : '#e2e8f0'}};">${{item.label}}</span>
                            </div>
                            <div class="rec-bar-bg">
                                <div class="rec-bar-fill" style="width:${{Math.max(4, item.score)}}%; ${{idx === 0 ? 'background:linear-gradient(90deg, #38bdf8, #818cf8);' : 'background:rgba(255,255,255,0.4);'}}"></div>
                            </div>
                            <div style="font-weight:700; font-size:0.92rem; color:${{idx === 0 ? '#38bdf8' : '#cbd5e1'}}; min-width:55px; text-align:right;">
                                ${{item.score}}%
                            </div>
                        </div>
                        <div id="rec-details-${{idx}}" style="display:none; background:rgba(0,0,0,0.25); border-left:3px solid #38bdf8; padding:10px 14px; margin:-4px 0 8px 10px; border-radius:0 8px 8px 0; font-size:0.82rem;">
                            <p style="color:#cbd5e1; margin:0 0 6px 0;">${{item.description}}</p>
                            ${{item.strengths.map(s => `<div style="color:#4ade80;">✅ ${{s}}</div>`).join('')}}
                            ${{item.weaknesses.map(w => `<div style="color:#f87171;">⚠️ ${{w}}</div>`).join('')}}
                        </div>
                    `).join('')}}
                </div>
            </div>
        `;
    }}
    
    document.getElementById('rec-modal-content').innerHTML = restrictionHtml + mainCardHtml + rankingsHtml;
}}

function toggleRankDetails(idx) {{
    const el = document.getElementById('rec-details-' + idx);
    if (el) {{
        el.style.display = (el.style.display === 'none' || el.style.display === '') ? 'block' : 'none';
    }}
}}

function setBaseTileLayer(targetLayer) {{
    const allLayers = [googleHybridLayer, googleStreetLayer, googleTerrainLayer, satelliteLayer, streetLayer, darkLayer];
    allLayers.forEach(l => {{
        if (l && leafletMap && leafletMap.hasLayer(l)) {{
            leafletMap.removeLayer(l);
        }}
    }});
    if (targetLayer && leafletMap) {{
        targetLayer.addTo(leafletMap);
    }}
}}

function switchMapView(mode) {{
    currentViewMode = mode;
    document.querySelectorAll('.view-tab').forEach(b => b.classList.remove('active'));
    const activeTab = document.getElementById('tab-' + mode);
    if (activeTab) activeTab.classList.add('active');
    
    const leafletElem = document.getElementById('leaflet-map');
    const threeContainer = document.getElementById('threejs-container');
    const controls2D = document.getElementById('map-2d-controls');
    
    if (mode === '3d') {{
        leafletElem.style.display = 'none';
        threeContainer.style.display = 'block';
        if (controls2D) controls2D.style.visibility = 'hidden';
        if (!threeScene) {{
            initThreeJS();
        }}
        if (lastOptimizationData) {{
            render3DTerrain(lastOptimizationData);
        }}
        onWindowResize3D();
    }} else {{
        threeContainer.style.display = 'none';
        leafletElem.style.display = 'block';
        if (controls2D) controls2D.style.visibility = 'visible';
        
        if (mode === 'google-hybrid' && googleHybridLayer) {{
            setBaseTileLayer(googleHybridLayer);
        }} else if (mode === 'google-streets' && googleStreetLayer) {{
            setBaseTileLayer(googleStreetLayer);
        }} else if (mode === 'esri-satellite' && satelliteLayer) {{
            setBaseTileLayer(satelliteLayer);
        }}
        
        if (leafletMap) leafletMap.invalidateSize();
    }}
}}

function initThreeJS() {{
    const container = document.getElementById('threejs-canvas');
    if (!container) return;
    
    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x070512);
    
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 540;
    threeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    threeCamera.position.set(0, 140, 180);
    
    threeRenderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(window.devicePixelRatio || 1);
    threeRenderer.shadowMap.enabled = true;
    container.innerHTML = '';
    container.appendChild(threeRenderer.domElement);
    
    threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.05;
    threeControls.maxPolarAngle = Math.PI / 2.05;
    threeControls.minDistance = 25;
    threeControls.maxDistance = 450;
    threeControls.target.set(0, 10, 0);
    
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    threeScene.add(ambientLight);
    
    const sunLight = new THREE.DirectionalLight(0xfffaed, 0.9);
    sunLight.position.set(-100, 180, 80);
    threeScene.add(sunLight);
    
    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x141228, 0.4);
    threeScene.add(hemiLight);
    
    const grid = new THREE.GridHelper(220, 22, 0x1e293b, 0x0f172a);
    grid.position.y = -0.5;
    threeScene.add(grid);
    
    function animate() {{
        requestAnimationFrame(animate);
        if (threeControls) threeControls.update();
        if (optimal3DPin) {{
            const time = Date.now() * 0.003;
            optimal3DPin.scale.set(1 + 0.08 * Math.sin(time), 1, 1 + 0.08 * Math.sin(time));
        }}
        if (threeRenderer && threeScene && threeCamera) {{
            threeRenderer.render(threeScene, threeCamera);
        }}
    }}
    animate();
    
    // Click Listener for Three.js 3D Terrain (Raycast)
    container.addEventListener('click', function(event) {{
        if (!terrainMesh || !threeCamera || !threeScene) return;
        const rect = container.getBoundingClientRect();
        const mouseX = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        const mouseY = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(new THREE.Vector2(mouseX, mouseY), threeCamera);
        const intersects = raycaster.intersectObject(terrainMesh);
        if (intersects.length > 0) {{
            const p = intersects[0].point;
            // planeW = 140, planeH = 175
            const colFrac = Math.max(0, Math.min(1, (p.x / 140.0) + 0.5));
            const rowFrac = Math.max(0, Math.min(1, (p.z / 175.0) + 0.5));
            const lat = MATALE_BOUNDS[1][0] - rowFrac * (MATALE_BOUNDS[1][0] - MATALE_BOUNDS[0][0]);
            const lon = MATALE_BOUNDS[0][1] + colFrac * (MATALE_BOUNDS[1][1] - MATALE_BOUNDS[0][1]);
            inspectLocation(lat, lon, p);
        }}
    }});
    
    window.addEventListener('resize', onWindowResize3D);
}}

function onWindowResize3D() {{
    const container = document.getElementById('threejs-canvas');
    if (!container || !threeCamera || !threeRenderer) return;
    const width = container.clientWidth;
    const height = container.clientHeight;
    if (width === 0 || height === 0) return;
    threeCamera.aspect = width / height;
    threeCamera.updateProjectionMatrix();
    threeRenderer.setSize(width, height);
}}

function render3DTerrain(data) {{
    if (!threeScene || !data.elevation_matrix) return;
    
    if (terrainMesh) {{
        threeScene.remove(terrainMesh);
        if (terrainMesh.geometry) terrainMesh.geometry.dispose();
        if (terrainMesh.material) {{
            if (Array.isArray(terrainMesh.material)) terrainMesh.material.forEach(m => m.dispose());
            else terrainMesh.material.dispose();
        }}
        terrainMesh = null;
    }}
    
    if (optimal3DPin) {{
        threeScene.remove(optimal3DPin);
        optimal3DPin = null;
    }}
    
    const elev = data.elevation_matrix;
    const rows = elev.length;
    const cols = elev[0].length;
    const elevMin = data.elevation_min !== undefined ? data.elevation_min : 70;
    const elevMax = data.elevation_max !== undefined ? data.elevation_max : 1864;
    const elevRange = Math.max(1, elevMax - elevMin);
    
    // Matale District true aspect ratio: Lon range ~0.55 deg (140), Lat range ~0.70 deg (175)
    const planeW = 140;
    const planeH = 175;
    const geometry = new THREE.PlaneGeometry(planeW, planeH, cols - 1, rows - 1);
    geometry.rotateX(-Math.PI / 2);
    
    const positions = geometry.attributes.position;
    const colors = [];
    
    for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
            const idx = r * cols + c;
            const val = elev[r][c];
            const normY = Math.max(0, Math.min(1, (val - elevMin) / elevRange));
            const height = normY * 32.0 * currentZScale;
            positions.setY(idx, height);
            
            // Topographic gradient based on real altitude across full district
            const color = new THREE.Color();
            if (normY < 0.15) {{
                color.setHSL(0.32, 0.60, 0.35); // Dambulla/Sigiriya northern plain green
            }} else if (normY < 0.35) {{
                color.setHSL(0.20, 0.55, 0.42); // foothills / midland valley
            }} else if (normY < 0.65) {{
                color.setHSL(0.08, 0.50, 0.46); // mountain slope / rock
            }} else {{
                color.setHSL(0.0, 0.15, 0.85); // Knuckles / Riverston misty summit
            }}
            colors.push(color.r, color.g, color.b);
        }}
    }}
    
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeVertexNormals();
    
    let material;
    if (current3DTextureMode === 'suitability' && data.leaflet_overlay_b64) {{
        const texture = new THREE.TextureLoader().load('data:image/png;base64,' + data.leaflet_overlay_b64, () => {{
            if (threeRenderer && threeScene && threeCamera) threeRenderer.render(threeScene, threeCamera);
        }});
        texture.flipY = false;
        material = new THREE.MeshStandardMaterial({{
            map: texture,
            roughness: 0.65,
            metalness: 0.15,
            wireframe: false,
        }});
    }} else if (current3DTextureMode === 'wireframe') {{
        material = new THREE.MeshBasicMaterial({{
            color: 0x38bdf8,
            wireframe: true,
        }});
    }} else {{
        material = new THREE.MeshStandardMaterial({{
            vertexColors: true,
            roughness: 0.75,
            metalness: 0.1,
            wireframe: false,
        }});
    }}
    
    terrainMesh = new THREE.Mesh(geometry, material);
    threeScene.add(terrainMesh);
    
    // Add 3D Optimal Site Beacon / Pin
    optimal3DPin = new THREE.Group();
    
    const bestYFrac = (data.best_x / 199.0);
    const bestXFrac = (data.best_y / 199.0);
    const pinX = (bestXFrac - 0.5) * planeW;
    const pinZ = (bestYFrac - 0.5) * planeH;
    const normSiteElev = (data.best_elevation - elevMin) / elevRange;
    const pinY = normSiteElev * 30.0 * currentZScale;
    
    const poleGeo = new THREE.CylinderGeometry(0.5, 0.5, 16, 16);
    const poleMat = new THREE.MeshBasicMaterial({{ color: 0xffd700 }});
    const pole = new THREE.Mesh(poleGeo, poleMat);
    pole.position.y = 8;
    optimal3DPin.add(pole);
    
    const headGeo = new THREE.SphereGeometry(3.2, 24, 24);
    const headMat = new THREE.MeshStandardMaterial({{
        color: 0xff3b30,
        emissive: 0xff453a,
        emissiveIntensity: 0.9
    }});
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 16;
    optimal3DPin.add(head);
    
    const ringGeo = new THREE.RingGeometry(2.5, 5.0, 32);
    ringGeo.rotateX(-Math.PI / 2);
    const ringMat = new THREE.MeshBasicMaterial({{ color: 0xffd700, side: THREE.DoubleSide }});
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.y = 0.5;
    optimal3DPin.add(ring);
    
    optimal3DPin.position.set(pinX, pinY, pinZ);
    threeScene.add(optimal3DPin);
    
    if (threeControls) {{
        threeControls.target.set(pinX * 0.4, pinY * 0.4, pinZ * 0.4);
    }}
}}

function update3DScale(val) {{
    currentZScale = parseFloat(val);
    document.getElementById('z-scale-val').textContent = val + 'x';
    if (lastOptimizationData) {{
        render3DTerrain(lastOptimizationData);
    }}
}}

function change3DTexture(mode) {{
    current3DTextureMode = mode;
    if (lastOptimizationData) {{
        render3DTerrain(lastOptimizationData);
    }}
}}

function reset3DCamera() {{
    if (!threeCamera || !threeControls) return;
    threeCamera.position.set(0, 140, 180);
    threeControls.target.set(0, 10, 0);
    threeControls.update();
}}

const FACILITIES_DATA = {facilities_json_str};

// Matale District Center & Bounds (Full District: Sigiriya & Dambulla down to Matale & Knuckles)
const MATALE_LAT = 7.70;
const MATALE_LON = 80.72;
const MATALE_BOUNDS = [[7.35, 80.45], [8.05, 81.00]];

function initLeafletMap() {{
    // 1. Initialize Leaflet Map
    leafletMap = L.map('leaflet-map').setView([MATALE_LAT, MATALE_LON], 10);
    
    // 2. Base Tile Layers
    googleHybridLayer = L.tileLayer('https://{{s}}.google.com/vt/lyrs=y&x={{x}}&y={{y}}&z={{z}}', {{
        subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
        attribution: '&copy; Google Maps | Hybrid (Satellite & Roads)',
        maxZoom: 20
    }});
    
    googleStreetLayer = L.tileLayer('https://{{s}}.google.com/vt/lyrs=m&x={{x}}&y={{y}}&z={{z}}', {{
        subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
        attribution: '&copy; Google Maps | Street Map',
        maxZoom: 20
    }});
    
    googleTerrainLayer = L.tileLayer('https://{{s}}.google.com/vt/lyrs=p&x={{x}}&y={{y}}&z={{z}}', {{
        subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
        attribution: '&copy; Google Maps | Terrain Relief',
        maxZoom: 20
    }});

    satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
        attribution: 'Esri Satellite | Matale District GIS',
        maxZoom: 19
    }});
    
    streetLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{y}}/{{x}}.png', {{
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    }});
    
    darkLayer = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{y}}/{{x}}{{r}}.png', {{
        attribution: '&copy; CartoDB DarkMatter',
        maxZoom: 19
    }});
    
    // Default to Google Hybrid (High-res satellite + clear road & town labels)
    googleHybridLayer.addTo(leafletMap);
    
    const baseLayers = {{
        "🟢 Google Hybrid (Satellite + Roads/Labels)": googleHybridLayer,
        "🟡 Google Streets (Google Maps)": googleStreetLayer,
        "🏔️ Google Terrain (Topographic Relief)": googleTerrainLayer,
        "🛰️ Esri Satellite (World Imagery)": satelliteLayer,
        "🗺️ OpenStreetMap (Standard)": streetLayer,
        "🌙 Dark Mode (CartoDB)": darkLayer
    }};
    
    L.control.layers(baseLayers, null, {{ position: 'topright' }}).addTo(leafletMap);
    
    // 3. Add Existing Facilities Pins
    schoolsLayerGroup = L.layerGroup();
    hospitalsLayerGroup = L.layerGroup();
    
    if (FACILITIES_DATA.schools) {{
        const schoolIcon = L.divIcon({{
            html: '<div style="background:#3b82f6; color:white; width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; border:2px solid white; box-shadow:0 2px 5px rgba(0,0,0,0.4);">🏫</div>',
            className: '',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        }});
        FACILITIES_DATA.schools.forEach((s, idx) => {{
            const popupHtml = `
                <div style="font-family:sans-serif; min-width:190px; padding:2px;">
                    <b style="color:#2563eb; font-size:0.95rem;">🏫 Existing School</b><br>
                    <strong style="color:#0f172a; font-size:0.9rem;">${{s.name}}</strong><br>
                    <span style="font-size:0.75rem; color:#64748b;">Lat: ${{s.lat.toFixed(4)}}, Lon: ${{s.lon.toFixed(4)}}</span><br>
                    <button onclick="inspectFacility('school', ${{idx}})" style="margin-top:8px; width:100%; padding:6px 10px; background:#2563eb; color:white; border:none; border-radius:6px; font-weight:600; cursor:pointer; font-size:0.8rem; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
                        🔬 Model Suitability Audit
                    </button>
                </div>
            `;
            L.marker([s.lat, s.lon], {{ icon: schoolIcon }})
                .bindPopup(popupHtml)
                .addTo(schoolsLayerGroup);
        }});
        schoolsLayerGroup.addTo(leafletMap);
    }}
    
    if (FACILITIES_DATA.hospitals) {{
        const hospitalIcon = L.divIcon({{
            html: '<div style="background:#22c55e; color:white; width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; border:2px solid white; box-shadow:0 2px 5px rgba(0,0,0,0.4);">🏥</div>',
            className: '',
            iconSize: [22, 22],
            iconAnchor: [11, 11]
        }});
        FACILITIES_DATA.hospitals.forEach((h, idx) => {{
            const popupHtml = `
                <div style="font-family:sans-serif; min-width:190px; padding:2px;">
                    <b style="color:#16a34a; font-size:0.95rem;">🏥 Existing Hospital</b><br>
                    <strong style="color:#0f172a; font-size:0.9rem;">${{h.name}}</strong><br>
                    <span style="font-size:0.75rem; color:#64748b;">Lat: ${{h.lat.toFixed(4)}}, Lon: ${{h.lon.toFixed(4)}}</span><br>
                    <button onclick="inspectFacility('hospital', ${{idx}})" style="margin-top:8px; width:100%; padding:6px 10px; background:#16a34a; color:white; border:none; border-radius:6px; font-weight:600; cursor:pointer; font-size:0.8rem; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
                        🔬 Model Suitability Audit
                    </button>
                </div>
            `;
            L.marker([h.lat, h.lon], {{ icon: hospitalIcon }})
                .bindPopup(popupHtml)
                .addTo(hospitalsLayerGroup);
        }});
        hospitalsLayerGroup.addTo(leafletMap);
    }}
    
    utilitiesLayerGroup = L.layerGroup();
    if (FACILITIES_DATA.utilities && FACILITIES_DATA.utilities.length > 0) {{
        FACILITIES_DATA.utilities.forEach((u, idx) => {{
            const utilIcon = L.divIcon({{
                html: `<div style="background:${{u.color || '#f97316'}}; color:white; width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:12px; border:2px solid white; box-shadow:0 2px 5px rgba(0,0,0,0.4);">${{u.emoji || '📍'}}</div>`,
                className: '',
                iconSize: [22, 22],
                iconAnchor: [11, 11]
            }});
            const popupHtml = `
                <div style="font-family:sans-serif; min-width:190px; padding:2px;">
                    <b style="color:#ea580c; font-size:0.95rem;">${{u.emoji || '📍'}} ${{u.category}}</b><br>
                    <strong style="color:#0f172a; font-size:0.9rem;">${{u.name}}</strong><br>
                    <span style="font-size:0.75rem; color:#64748b;">Lat: ${{u.lat.toFixed(4)}}, Lon: ${{u.lon.toFixed(4)}}</span><br>
                    <button onclick="inspectFacility('utility', ${{idx}})" style="margin-top:8px; width:100%; padding:6px 10px; background:#ea580c; color:white; border:none; border-radius:6px; font-weight:600; cursor:pointer; font-size:0.8rem; box-shadow:0 2px 4px rgba(0,0,0,0.2);">
                        🔬 Model Suitability Audit
                    </button>
                </div>
            `;
            L.marker([u.lat, u.lon], {{ icon: utilIcon }})
                .bindPopup(popupHtml)
                .addTo(utilitiesLayerGroup);
        }});
        utilitiesLayerGroup.addTo(leafletMap);
    }}

    // 4. Add Waterways & River Riparian Flood Buffers (749 streams)
    waterwaysLayerGroup = L.layerGroup();
    fetch('/api/waterways')
        .then(res => res.json())
        .then(geojson => {{
            if (geojson && geojson.features && geojson.features.length > 0) {{
                L.geoJSON(geojson, {{
                    style: function(feature) {{
                        const isRiver = (feature.properties && feature.properties.fclass === 'river');
                        return {{
                            color: isRiver ? '#0284c7' : '#06b6d4',
                            weight: isRiver ? 3.5 : 2.0,
                            opacity: 0.85,
                            dashArray: isRiver ? null : '4, 4'
                        }};
                    }},
                    onEachFeature: function(feature, layer) {{
                        const props = feature.properties || {{}};
                        const name = props.name || (props.fclass === 'river' ? 'Major River' : 'Stream / Canal');
                        const bufferDist = (props.fclass === 'river') ? '60m Mandatory Flood Buffer' : '30m Stream Flood Buffer';
                        layer.bindTooltip(`<b>🌊 ${{name}}</b><br><span style="font-size:0.75rem; color:#f87171;">⚠️ ${{bufferDist}} (No School Siting)</span>`, {{ sticky: true }});
                        layer.on('click', function(e) {{
                            L.DomEvent.stopPropagation(e);
                            inspectLocation(e.latlng.lat, e.latlng.lng);
                        }});
                    }}
                }}).addTo(waterwaysLayerGroup);
                waterwaysLayerGroup.addTo(leafletMap);
            }}
        }})
        .catch(err => console.warn('Waterways load error:', err));
    
    // Fit to Matale
    leafletMap.fitBounds(MATALE_BOUNDS);
    
    // 5. Click Listener for Site Recommendation & Highest/Best Use (HBU)
    leafletMap.on('click', function(e) {{
        inspectLocation(e.latlng.lat, e.latlng.lng);
    }});
}}

function updateOverlayOpacity(val) {{
    if (currentOverlayLayer) {{
        currentOverlayLayer.setOpacity(parseFloat(val));
    }}
}}

function toggleFacilityLayer(layerType, isChecked) {{
    if (layerType === 'schools' && schoolsLayerGroup) {{
        if (isChecked) schoolsLayerGroup.addTo(leafletMap);
        else leafletMap.removeLayer(schoolsLayerGroup);
    }}
    if (layerType === 'hospitals' && hospitalsLayerGroup) {{
        if (isChecked) hospitalsLayerGroup.addTo(leafletMap);
        else leafletMap.removeLayer(hospitalsLayerGroup);
    }}
    if (layerType === 'utilities' && utilitiesLayerGroup) {{
        if (isChecked) utilitiesLayerGroup.addTo(leafletMap);
        else leafletMap.removeLayer(utilitiesLayerGroup);
    }}
    if (layerType === 'waterways' && waterwaysLayerGroup) {{
        if (isChecked) waterwaysLayerGroup.addTo(leafletMap);
        else leafletMap.removeLayer(waterwaysLayerGroup);
    }}
}}

function selectPrimaryInfra(type) {{
    currentPrimaryInfra = type;
    document.querySelectorAll('.infra-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('btn-' + type).classList.add('active');
    
    const subContainer = document.getElementById('school-subtypes');
    if (type === 'school') {{
        subContainer.style.display = 'block';
        currentInfra = currentSchoolSubtype;
    }} else {{
        subContainer.style.display = 'none';
        currentInfra = type;
    }}
    executeSearch();
}}

function selectSchoolSubtype(subtype) {{
    currentSchoolSubtype = subtype;
    currentInfra = subtype;
    document.querySelectorAll('.subtype-pill').forEach(b => b.classList.remove('active'));
    document.getElementById('sub-' + subtype).classList.add('active');
    executeSearch();
}}

function applyZonePreset(key, minX, maxX, minY, maxY) {{
    document.getElementById('roi-min-x').value = minX;
    document.getElementById('roi-max-x').value = maxX;
    document.getElementById('roi-min-y').value = minY;
    document.getElementById('roi-max-y').value = maxY;
    
    document.querySelectorAll('.zone-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('zone-' + key).classList.add('active');
    
    currentZoneLabel = document.querySelector('#zone-' + key + ' .zone-title').textContent;
    executeSearch();
}}

function executeSearch() {{
    const minX = parseInt(document.getElementById('roi-min-x').value);
    const maxX = parseInt(document.getElementById('roi-max-x').value);
    const minY = parseInt(document.getElementById('roi-min-y').value);
    const maxY = parseInt(document.getElementById('roi-max-y').value);
    
    document.getElementById('loading').classList.add('show');
    document.getElementById('results').classList.remove('show');
    
    const url = `/api/optimize?type=${{currentInfra}}&min_x=${{minX}}&max_x=${{maxX}}&min_y=${{minY}}&max_y=${{maxY}}&zone=${{encodeURIComponent(currentZoneLabel)}}`;
    
    fetch(url)
        .then(r => r.json())
        .then(data => {{
            document.getElementById('loading').classList.remove('show');
            if (data.error) {{
                alert('Error: ' + data.error);
                return;
            }}
            lastOptimizationData = data;
            displayResults(data);
            updateLeafletMap(data);
            if (currentViewMode === '3d') {{
                if (!threeScene) initThreeJS();
                render3DTerrain(data);
            }}
        }})
        .catch(err => {{
            document.getElementById('loading').classList.remove('show');
            alert('Error running optimization: ' + err.message);
        }});
}}

function updateLeafletMap(data) {{
    if (!leafletMap) return;
    
    // 1. Update Heatmap Overlay
    if (currentOverlayLayer) {{
        leafletMap.removeLayer(currentOverlayLayer);
    }}
    const opacity = parseFloat(document.getElementById('opacity-slider').value);
    currentOverlayLayer = L.imageOverlay('data:image/png;base64,' + data.leaflet_overlay_b64, MATALE_BOUNDS, {{
        opacity: opacity,
        interactive: false
    }}).addTo(leafletMap);
    
    // 2. Update Bounding Box Rectangle (if sub-region)
    if (currentRoiRectangle) {{
        leafletMap.removeLayer(currentRoiRectangle);
    }}
    if (data.is_sub_region && data.roi_geo) {{
        const rectBounds = [
            [data.roi_geo.south, data.roi_geo.west],
            [data.roi_geo.north, data.roi_geo.east]
        ];
        currentRoiRectangle = L.rectangle(rectBounds, {{
            color: '#00f2fe',
            weight: 3,
            dashArray: '6, 6',
            fillColor: '#00f2fe',
            fillOpacity: 0.12
        }}).addTo(leafletMap);
        leafletMap.fitBounds(rectBounds, {{ padding: [30, 30] }});
    }} else {{
        leafletMap.fitBounds(MATALE_BOUNDS);
    }}
    
    // 3. Place Optimal Site Marker
    if (currentOptimalMarker) {{
        leafletMap.removeLayer(currentOptimalMarker);
    }}
    
    const starIcon = L.divIcon({{
        html: `<div class="optimal-pulse" style="width:36px; height:36px; display:flex; align-items:center; justify-content:center; font-size:22px; cursor:pointer;">${{data.infra_emoji}}</div>`,
        className: '',
        iconSize: [36, 36],
        iconAnchor: [18, 18]
    }});
    
    const gmapsUrl = `https://www.google.com/maps?q=${{data.best_lat}},${{data.best_lon}}&t=k`;
    
    currentOptimalMarker = L.marker([data.best_lat, data.best_lon], {{ icon: starIcon }}).addTo(leafletMap);
    
    const popupContent = `
        <div style="font-family:'Segoe UI', sans-serif; min-width:210px;">
            <div style="font-size:1.05rem; font-weight:700; color:#0284c7; margin-bottom:4px;">
                ${{data.infra_emoji}} Recommended ${{data.infra_label}}
            </div>
            <div style="font-size:0.85rem; color:#475569; margin-bottom:6px;">
                <strong>Suitability:</strong> <span style="color:#16a34a; font-weight:700;">${{(data.best_fitness * 100).toFixed(1)}}%</span><br>
                <strong>Elevation:</strong> ${{data.best_elevation.toFixed(0)}} m<br>
                <strong>Coordinates:</strong> ${{data.best_lat.toFixed(4)}}°N, ${{data.best_lon.toFixed(4)}}°E
            </div>
            <a href="${{gmapsUrl}}" target="_blank" style="display:inline-block; background:#4285F4; color:white; padding:6px 12px; border-radius:14px; text-decoration:none; font-weight:600; font-size:0.8rem; margin-top:4px;">
                🗺️ Open in Google Maps ↗
            </a>
        </div>
    `;
    currentOptimalMarker.bindPopup(popupContent).openPopup();
}}

function displayResults(data) {{
    const subNote = data.is_sub_region ? ` (Inside ${{data.zone_label}})` : '';
    document.getElementById('result-title').textContent = `${{data.infra_emoji}} Best Location for ${{data.infra_label}}${{subNote}}`;
    document.getElementById('result-badge').textContent = `Grid (${{data.best_x.toFixed(1)}}, ${{data.best_y.toFixed(1)}}) | Elevation: ${{data.best_elevation.toFixed(0)}}m`;
    document.getElementById('result-coords').innerHTML = `<strong>GPS Coordinates:</strong> Latitude ${{data.best_lat.toFixed(4)}}°N, Longitude ${{data.best_lon.toFixed(4)}}°E`;
    document.getElementById('result-details').textContent = data.description;
    
    // Update Google Maps Link button
    const gmapsUrl = `https://www.google.com/maps?q=${{data.best_lat}},${{data.best_lon}}&t=k`;
    document.getElementById('gmaps-btn').href = gmapsUrl;
    
    // Cards
    document.getElementById('result-cards').innerHTML = `
        <div class="card">
            <h3>AHP Consistency</h3>
            <div class="value yellow">CR = ${{data.cr.toFixed(4)}}</div>
            <div class="sub">${{data.cr_pass ? 'PASS (Consistent)' : 'FAIL'}}</div>
        </div>
        <div class="card">
            <h3>Best Suitability Score</h3>
            <div class="value green">${{data.best_fitness.toFixed(4)}}</div>
            <div class="sub">${{(data.best_fitness * 100).toFixed(1)}}% Match</div>
        </div>
        <div class="card">
            <h3>GA Result</h3>
            <div class="value" style="color:#f87171">${{data.ga.fitness.toFixed(4)}}</div>
            <div class="sub">Time: ${{data.ga.time.toFixed(3)}}s</div>
        </div>
        <div class="card">
            <h3>PSO Result</h3>
            <div class="value blue">${{data.pso.fitness.toFixed(4)}}</div>
            <div class="sub">Time: ${{data.pso.time.toFixed(3)}}s</div>
        </div>
    `;
    
    // Building Specs Cards
    if (data.building_specs) {{
        const b = data.building_specs;
        const penaltyPct = Math.round((b.construction_cost.cost_multiplier - 1.0) * 100);
        document.getElementById('specs-grid').innerHTML = `
            <div class="spec-card">
                <div class="icon-title">👥 Demographics &amp; Capacity</div>
                <div class="main-val" style="color:#38bdf8;">${{b.capacity.value.toLocaleString()}} <span>${{b.capacity.unit}}</span></div>
                <div class="detail-list">
                    <div class="detail-item"><span class="detail-label">Catchment (3km):</span><span class="detail-val">${{b.catchment.population_3km.toLocaleString()}} residents</span></div>
                    <div class="detail-item"><span class="detail-label">Existing Buildings:</span><span class="detail-val" style="color:#c084fc;">${{b.neighborhood_buildings ? b.neighborhood_buildings + ' in grid cell' : 'Rural / Sparsely Built'}}</span></div>
                    <div class="detail-item"><span class="detail-label">Facility Sizing:</span><span class="detail-val" style="color:#facc15;">${{b.capacity.subdivisions}}</span></div>
                </div>
            </div>

            <div class="spec-card">
                <div class="icon-title">🏢 Floor Area &amp; Sizing</div>
                <div class="main-val" style="color:#4ade80;">${{b.architectural.recommended_floors}} <span>Stories / Floors</span></div>
                <div class="detail-list">
                    <div class="detail-item"><span class="detail-label">Total Built Area:</span><span class="detail-val">${{b.architectural.total_built_up_m2.toLocaleString()}} m² (${{b.architectural.total_built_up_sqft.toLocaleString()}} sq.ft)</span></div>
                    <div class="detail-item"><span class="detail-label">Ground Footprint:</span><span class="detail-val">${{b.architectural.ground_footprint_m2.toLocaleString()}} m²</span></div>
                    <div class="detail-item"><span class="detail-label">Story Sizing:</span><span class="detail-val">Adaptive for ${{b.slope_deg}}° slope</span></div>
                </div>
            </div>

            <div class="spec-card">
                <div class="icon-title">📐 Foundation &amp; Terrain Sizing</div>
                <div class="main-val" style="color:#fb923c; font-size:1.25rem; margin-top:2px;">${{b.construction_cost.cost_tier}}</div>
                <div class="detail-list">
                    <div class="detail-item"><span class="detail-label">Foundation Eng:</span><span class="detail-val" style="font-size:0.75rem;">${{b.architectural.foundation_type}}</span></div>
                    <div class="detail-item"><span class="detail-label">Terrain Slope:</span><span class="detail-val">${{b.slope_deg}}° gradient</span></div>
                    <div class="detail-item"><span class="detail-label">Cost Multiplier:</span><span class="detail-val">${{b.construction_cost.cost_multiplier}}x Base</span></div>
                </div>
            </div>

            <div class="spec-card">
                <div class="icon-title">☀️ Rooftop Solar PV Potential</div>
                <div class="main-val" style="color:#facc15;">${{b.solar_energy.solar_capacity_kwp}} <span>kWp PV</span></div>
                <div class="detail-list">
                    <div class="detail-item"><span class="detail-label">Local Solar GHI:</span><span class="detail-val" style="color:#fde047;">${{b.solar_energy.solar_ghi_kwh_m2 || 1910}} kWh/m²/yr</span></div>
                    <div class="detail-item"><span class="detail-label">Annual Solar Power:</span><span class="detail-val">${{b.solar_energy.annual_solar_kwh.toLocaleString()}} kWh/yr</span></div>
                    <div class="detail-item"><span class="detail-label">Carbon Offset:</span><span class="detail-val" style="color:#4ade80;">${{b.solar_energy.annual_co2_offset_tons}} t CO₂/yr</span></div>
                </div>
            </div>

            <div class="spec-card">
                <div class="icon-title">🏟️ Campus &amp; Infrastructure Extent</div>
                <div class="main-val" style="color:#a78bfa; font-size:1.2rem;">${{b.campus_specs ? b.campus_specs.land_requirement : 'Standard Campus Parcel'}}</div>
                <div class="detail-list">
                    <div class="detail-item"><span class="detail-label">Sports Complex:</span><span class="detail-val" style="color:#38bdf8; font-size:0.75rem;">${{b.campus_specs ? b.campus_specs.sports_facilities : 'Standard Sports Playground'}}</span></div>
                    <div class="detail-item"><span class="detail-label">Traffic Management:</span><span class="detail-val" style="color:#4ade80; font-size:0.75rem;">${{b.campus_specs ? b.campus_specs.traffic_guidance : 'Standard Access Road'}}</span></div>
                </div>
            </div>

            <div class="spec-card">
                <div class="icon-title">💰 Est. Construction Budget</div>
                <div class="main-val" style="color:#ec4899; font-size:1.35rem;">${{b.construction_cost.estimated_budget_formatted}}</div>
                <div class="detail-list">
                    <div class="detail-item"><span class="detail-label">Base Cost Rate:</span><span class="detail-val">LKR 135,000 / m²</span></div>
                    <div class="detail-item"><span class="detail-label">Terrain Penalty:</span><span class="detail-val">+${{penaltyPct}}% foundation</span></div>
                    <div class="detail-item"><span class="detail-label">Feasibility:</span><span class="detail-val" style="color:#38bdf8;">Standard Feasible</span></div>
                </div>
            </div>
        `;
    }}
    
    // Weights
    let barsHtml = '';
    for (const [name, weight] of Object.entries(data.weights)) {{
        const pct = (weight * 100).toFixed(1);
        barsHtml += `
            <div class="weight-bar">
                <span class="label">${{name}}</span>
                <div class="bar-bg"><div class="bar-fill" style="width:${{pct}}%">${{pct}}%</div></div>
            </div>
        `;
    }}
    document.getElementById('weights-bars').innerHTML = barsHtml;
    
    // Images
    document.getElementById('heatmap-img').src = 'data:image/png;base64,' + data.heatmap_b64;
    document.getElementById('convergence-img').src = 'data:image/png;base64,' + data.convergence_b64;
    
    // Benchmark table
    document.getElementById('benchmark-table').innerHTML = `
        <tr><td>Suitability Score</td><td>${{data.ga.fitness.toFixed(6)}}</td><td>${{data.pso.fitness.toFixed(6)}}</td></tr>
        <tr><td>Optimal X (Row)</td><td>${{data.ga.x.toFixed(4)}}</td><td>${{data.pso.x.toFixed(4)}}</td></tr>
        <tr><td>Optimal Y (Col)</td><td>${{data.ga.y.toFixed(4)}}</td><td>${{data.pso.y.toFixed(4)}}</td></tr>
        <tr><td>Elevation Z (m)</td><td>${{data.ga.elevation.toFixed(2)}} m</td><td>${{data.pso.elevation.toFixed(2)}} m</td></tr>
        <tr><td>Runtime (seconds)</td><td>${{data.ga.time.toFixed(4)}} s</td><td>${{data.pso.time.toFixed(4)}} s</td></tr>
        <tr><td><strong>Algorithm Winner</strong></td><td colspan="2" style="text-align:center;color:#4ade80;font-weight:700">${{data.winner}}</td></tr>
    `;
    
    document.getElementById('results').classList.add('show');
}}

// Initialize on window load
window.onload = function() {{
    document.getElementById('btn-school').classList.add('active');
    const subNat = document.getElementById('sub-national_school');
    if (subNat) subNat.classList.add('active');
    document.getElementById('zone-all').classList.add('active');
    initLeafletMap();
    executeSearch();
}};
</script>

</body>
</html>"""


class GoogleMapsDashboardHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler supporting Google Maps / Leaflet GIS integration."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/optimize":
            self._handle_optimize(parsed)
        elif parsed.path == "/api/facilities":
            self._handle_facilities()
        elif parsed.path == "/api/waterways":
            self._handle_waterways()
        elif parsed.path == "/api/recommend-site":
            self._handle_recommend_site(parsed)
        else:
            self._serve_main_page()

    def _serve_main_page(self):
        html = build_main_page()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def _handle_facilities(self):
        facilities = get_facilities_data()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(facilities).encode("utf-8"))

    def _handle_waterways(self):
        waterways = get_waterways_data()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(waterways).encode("utf-8"))

    def _handle_optimize(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        infra_type = params.get("type", ["school"])[0]
        if infra_type not in INFRASTRUCTURE_PRESETS:
            infra_type = "school"

        try:
            min_x = int(params.get("min_x", [0])[0])
            max_x = int(params.get("max_x", [GRID_SIZE])[0])
            min_y = int(params.get("min_y", [0])[0])
            max_y = int(params.get("max_y", [GRID_SIZE])[0])
            roi = (min_x, max_x, min_y, max_y)
        except (ValueError, IndexError):
            roi = None

        zone_label = params.get("zone", ["Full District"])[0]

        try:
            result = run_optimization(infra_type, roi=roi, zone_label=zone_label)
            response = json.dumps(result)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = json.dumps({"error": str(e)})
            self.wfile.write(error.encode("utf-8"))

    def _handle_recommend_site(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        try:
            lat = float(params.get("lat", [0.0])[0])
            lon = float(params.get("lon", [0.0])[0])
            facility_name = params.get("name", [""])[0]
            facility_type = params.get("type", [""])[0]
            result = evaluate_site_suitability(lat, lon, facility_name=facility_name, facility_type=facility_type)
            response = json.dumps(result)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response.encode("utf-8"))
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error = json.dumps({"error": str(e)})
            self.wfile.write(error.encode("utf-8"))

    def log_message(self, format, *args):
        pass


def main():
    # Pre-cache data
    get_spatial_data()
    get_facilities_data()
    get_waterways_data()
    print(f"Starting Google Maps & GIS Spatial Optimization Dashboard on port {PORT}...")

    with socketserver.TCPServer(("", PORT), GoogleMapsDashboardHandler) as httpd:
        url = f"http://localhost:{PORT}"
        print(f"\n  Dashboard running at: {url}")
        print("  Press Ctrl+C to stop.\n")
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


if __name__ == "__main__":
    main()
