"""
Real Data Loader for Matale District, Sri Lanka.

Loads actual geospatial data (OSM shapefiles + WorldPop population raster),
clips to Matale District bounds, and produces normalized raster layers
compatible with the spatial optimization pipeline.

Falls back to synthetic elevation/slope when DEM is not available.
"""

import os
import sys
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

# Matale District approximate bounding box (WGS84 / EPSG:4326)
MATALE_BOUNDS = {
    "min_lon": 80.45,
    "max_lon": 81.00,
    "min_lat": 7.35,
    "max_lat": 8.05,
}


class RealDataLoader:
    """
    Loads real geospatial data for Matale District and produces
    normalized raster layers for the optimization pipeline.

    Produces the same output dict as SyntheticRasterGenerator.generate_all():
        - 'elevation': 2D array (meters)
        - 'slope': 2D array (normalized, inverted — flat=high)
        - 'road_proximity': 2D array (normalized — close=high)
        - 'population_growth': 2D array (normalized 0-1)
        - 'constraint_mask': 2D binary array (1=buildable, 0=restricted)

    Args:
        data_dir: Path to directory containing data files.
        grid_size: Output raster dimensions (grid_size x grid_size).
        bounds: Dict with min_lon, max_lon, min_lat, max_lat.
        dem_path: Optional path to DEM GeoTIFF. If None, synthetic elevation used.
    """

    # Filenames expected in data_dir
    POPULATION_FILE = "lka_pop_2020_CN_100m_R2025A_v1.tif"
    ROADS_FILE = "gis_osm_roads_free_1.shp"
    WATER_FILE = "gis_osm_water_a_free_1.shp"
    PROTECTED_FILE = "gis_osm_protected_areas_a_free_1.shp"
    POIS_FILE = "gis_osm_pois_a_free_1.shp"
    POIS_POINT_FILE = "gis_osm_pois_free_1.shp"
    LANDUSE_FILE = "gis_osm_landuse_a_free_1.shp"
    TRANSPORT_FILE = "gis_osm_transport_free_1.shp"

    # Elevation constants for synthetic fallback (Matale terrain)
    ELEV_MIN = 250.0
    ELEV_MAX = 850.0
    MAX_SLOPE_DEG = 45.0
    EDGE_BUFFER_PX = 5

    def __init__(
        self,
        data_dir: str = "data",
        grid_size: int = 200,
        bounds: Optional[Dict[str, float]] = None,
        dem_path: Optional[str] = None,
        random_seed: int = 42,
    ):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.bounds = bounds or MATALE_BOUNDS
        self.dem_path = dem_path
        self.rng = np.random.RandomState(random_seed)

        # Lazy imports — check availability
        self._rasterio = None
        self._gpd = None
        self._check_dependencies()

        # Dedicated sub-constraint diagnostic masks
        self.water_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.protected_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.waterways_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.pofw_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

    def _check_dependencies(self) -> None:
        """Check and import required GIS libraries."""
        try:
            import rasterio
            self._rasterio = rasterio
        except ImportError:
            print("WARNING: rasterio not installed. Population raster loading disabled.")
            print("  Install with: pip install rasterio")

        try:
            import geopandas as gpd
            self._gpd = gpd
        except ImportError:
            print("WARNING: geopandas not installed. Shapefile loading disabled.")
            print("  Install with: pip install geopandas")

    def _file_path(self, filename: str) -> str:
        """Get full path to a data file."""
        return os.path.join(self.data_dir, filename)

    def _file_exists(self, filename: str) -> bool:
        """Check if a data file exists."""
        path = self._file_path(filename)
        exists = os.path.exists(path)
        if not exists:
            print(f"  [SKIP] File not found: {filename}")
        return exists

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1]."""
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin < 1e-10:
            return np.zeros_like(arr, dtype=np.float64)
        return (arr - vmin) / (vmax - vmin)

    # ──────────────────────────────────────────────────────────────
    # Population Raster (WorldPop GeoTIFF)
    # ──────────────────────────────────────────────────────────────

    def _load_population(self) -> np.ndarray:
        """Load and resample WorldPop population raster to grid."""
        print("\n  Loading population raster...")

        if self._rasterio is None or not self._file_exists(self.POPULATION_FILE):
            print("  [FALLBACK] Using synthetic population data")
            return self._synthetic_population()

        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        filepath = self._file_path(self.POPULATION_FILE)

        with rasterio.open(filepath) as src:
            # Calculate the window for Matale bounds
            window = from_bounds(
                self.bounds["min_lon"],
                self.bounds["min_lat"],
                self.bounds["max_lon"],
                self.bounds["max_lat"],
                src.transform,
            )

            # Read the window and resample to grid_size
            data = src.read(
                1,
                window=window,
                out_shape=(self.grid_size, self.grid_size),
                resampling=Resampling.bilinear,
            )

        # Handle nodata values
        data = np.nan_to_num(data, nan=0.0)
        data = np.clip(data, 0, None)  # No negative population

        # Normalize
        result = self._normalize(data)

        # Enhanced Demographic Modeling: Fuse WorldPop with 63,860 real building footprints
        bldg_density_path = os.path.join(self.data_dir, "matale_building_density.npy")
        if os.path.exists(bldg_density_path):
            try:
                bldg_density = np.load(bldg_density_path)
                if bldg_density.shape == (self.grid_size, self.grid_size):
                    norm_bldg = self._normalize(bldg_density)
                    # 60% macro census population + 40% micro physical building footprint density
                    fused_pop = 0.60 * result + 0.40 * norm_bldg
                    result = self._normalize(fused_pop)
                    print("  [ENHANCED] Demographic demand fused with 63,860 real OSM building footprints!")
            except Exception as e:
                print(f"  [WARN] Building density fusion skipped: {e}")

        print(f"  [OK] Population loaded: shape={result.shape}, "
              f"range=[{data.min():.1f}, {data.max():.1f}] people/pixel")
        return result

    # ──────────────────────────────────────────────────────────────
    # Road Proximity (OSM Shapefile → Rasterize → EDT)
    # ──────────────────────────────────────────────────────────────

    def _load_roads(self) -> np.ndarray:
        """Load roads, rasterize, and compute proximity."""
        print("\n  Loading road network...")

        if self._gpd is None or not self._file_exists(self.ROADS_FILE):
            print("  [FALLBACK] Using synthetic road proximity")
            return self._synthetic_road_proximity()

        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        filepath = self._file_path(self.ROADS_FILE)

        # Read and clip to Matale bounds
        gdf = gpd.read_file(
            filepath,
            bbox=(
                self.bounds["min_lon"],
                self.bounds["min_lat"],
                self.bounds["max_lon"],
                self.bounds["max_lat"],
            ),
        )

        if gdf.empty:
            print("  [WARN] No roads found in Matale bounds. Using synthetic.")
            return self._synthetic_road_proximity()

        print(f"  Found {len(gdf)} road segments in Matale area")

        # Create transform for rasterization
        transform = from_bounds(
            self.bounds["min_lon"],
            self.bounds["min_lat"],
            self.bounds["max_lon"],
            self.bounds["max_lat"],
            self.grid_size,
            self.grid_size,
        )

        # Rasterize roads: 1 = road pixel, 0 = no road
        road_raster = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=(self.grid_size, self.grid_size),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Compute Euclidean distance from roads
        # Invert: pixels that are no-road get distance; road pixels = 0
        no_road_mask = road_raster == 0
        distance = distance_transform_edt(no_road_mask)

        # Invert and normalize: close to road = high score
        result = 1.0 - self._normalize(distance)
        print(f"  [OK] Road proximity computed: {int(road_raster.sum())} road pixels")
        return result

    # ──────────────────────────────────────────────────────────────
    # Constraint Mask (Water + Protected Areas)
    # ──────────────────────────────────────────────────────────────

    def _load_constraints(self) -> np.ndarray:
        """Load water bodies and protected areas as constraint mask."""
        print("\n  Loading constraint zones...")

        if self._gpd is None:
            print("  [FALLBACK] Using synthetic constraints")
            return self._synthetic_constraints()

        from rasterio.features import rasterize
        from rasterio.transform import from_bounds
        import geopandas as gpd

        transform = from_bounds(
            self.bounds["min_lon"],
            self.bounds["min_lat"],
            self.bounds["max_lon"],
            self.bounds["max_lat"],
            self.grid_size,
            self.grid_size,
        )

        # Start with all buildable
        constraint_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float64)
        self.water_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.protected_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.waterways_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.pofw_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        # Water bodies → restricted
        if self._file_exists(self.WATER_FILE):
            water_gdf = gpd.read_file(
                self._file_path(self.WATER_FILE),
                bbox=(
                    self.bounds["min_lon"],
                    self.bounds["min_lat"],
                    self.bounds["max_lon"],
                    self.bounds["max_lat"],
                ),
            )
            if not water_gdf.empty:
                water_raster = rasterize(
                    [(geom, 1) for geom in water_gdf.geometry if geom is not None],
                    out_shape=(self.grid_size, self.grid_size),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                self.water_mask = (water_raster == 1).astype(np.uint8)
                constraint_mask[self.water_mask == 1] = 0.0
                water_count = int(self.water_mask.sum())
                print(f"  [OK] Water bodies: {water_count} restricted pixels")
            else:
                print("  [INFO] No water bodies in Matale bounds")

        # Protected areas → restricted
        if self._file_exists(self.PROTECTED_FILE):
            prot_gdf = gpd.read_file(
                self._file_path(self.PROTECTED_FILE),
                bbox=(
                    self.bounds["min_lon"],
                    self.bounds["min_lat"],
                    self.bounds["max_lon"],
                    self.bounds["max_lat"],
                ),
            )
            if not prot_gdf.empty:
                prot_raster = rasterize(
                    [(geom, 1) for geom in prot_gdf.geometry if geom is not None],
                    out_shape=(self.grid_size, self.grid_size),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                self.protected_mask = (prot_raster == 1).astype(np.uint8)
                constraint_mask[self.protected_mask == 1] = 0.0
                prot_count = int(self.protected_mask.sum())
                print(f"  [OK] Protected areas: {prot_count} restricted pixels")
            else:
                print("  [INFO] No protected areas in Matale bounds")

        # Waterways & River Flood Protection Buffers (Irrigation Dept & UDA Norms)
        waterways_file = "gis_osm_waterways_free_1.shp"
        if self._file_exists(waterways_file):
            ww_gdf = gpd.read_file(
                self._file_path(waterways_file),
                bbox=(
                    self.bounds["min_lon"],
                    self.bounds["min_lat"],
                    self.bounds["max_lon"],
                    self.bounds["max_lat"],
                ),
            )
            if not ww_gdf.empty:
                if ww_gdf.crs is None:
                    ww_gdf = ww_gdf.set_crs(epsg=4326)
                ww_metric = ww_gdf.to_crs(epsg=3857)
                # Apply 60m buffer for rivers, 30m for streams/canals
                buffer_dists = ww_metric["fclass"].apply(lambda fc: 60.0 if fc == "river" else 30.0)
                buffered_ww = ww_metric.geometry.buffer(buffer_dists).to_crs(epsg=4326)

                ww_raster = rasterize(
                    [(geom, 1) for geom in buffered_ww if geom is not None and not geom.is_empty],
                    out_shape=(self.grid_size, self.grid_size),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                self.waterways_mask = (ww_raster == 1).astype(np.uint8)
                constraint_mask[self.waterways_mask == 1] = 0.0
                ww_count = int(self.waterways_mask.sum())
                print(f"  [ENHANCED] Waterways & River Flood Buffers: {len(ww_gdf)} segments ({ww_count} restricted pixels)")

        # Places of Worship & Cultural Heritage Buffers (Temples, Kovils, Mosques, Churches)
        pofw_file = "gis_osm_pofw_a_free_1.shp"
        if self._file_exists(pofw_file):
            pofw_gdf = gpd.read_file(
                self._file_path(pofw_file),
                bbox=(
                    self.bounds["min_lon"],
                    self.bounds["min_lat"],
                    self.bounds["max_lon"],
                    self.bounds["max_lat"],
                ),
            )
            if not pofw_gdf.empty:
                if pofw_gdf.crs is None:
                    pofw_gdf = pofw_gdf.set_crs(epsg=4326)
                pofw_buffered = pofw_gdf.to_crs(epsg=3857).geometry.buffer(30.0).to_crs(epsg=4326)

                pofw_raster = rasterize(
                    [(geom, 1) for geom in pofw_buffered if geom is not None and not geom.is_empty],
                    out_shape=(self.grid_size, self.grid_size),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                )
                self.pofw_mask = (pofw_raster == 1).astype(np.uint8)
                constraint_mask[self.pofw_mask == 1] = 0.0
                pofw_count = int(self.pofw_mask.sum())
                print(f"  [ENHANCED] Places of Worship & Cultural Buffers: {len(pofw_gdf)} sites ({pofw_count} restricted pixels)")

        # Edge buffer
        constraint_mask[:self.EDGE_BUFFER_PX, :] = 0.0
        constraint_mask[-self.EDGE_BUFFER_PX:, :] = 0.0
        constraint_mask[:, :self.EDGE_BUFFER_PX] = 0.0
        constraint_mask[:, -self.EDGE_BUFFER_PX:] = 0.0

        buildable = constraint_mask.sum() / constraint_mask.size * 100
        print(f"  [OK] Constraint mask: {buildable:.1f}% buildable area")
        return constraint_mask

    # ──────────────────────────────────────────────────────────────
    # Elevation & Slope (DEM or Synthetic Fallback)
    # ──────────────────────────────────────────────────────────────

    def _load_elevation(self) -> np.ndarray:
        dem_default = os.path.join(self.data_dir, "matale_dem.tif")
        target_dem = self.dem_path if (self.dem_path and os.path.exists(self.dem_path)) else (dem_default if os.path.exists(dem_default) else None)
        if target_dem and self._rasterio:
            print(f"\n  Loading Real NASA SRTM DEM elevation from {target_dem}...")
            import rasterio
            from rasterio.windows import from_bounds
            from rasterio.enums import Resampling

            with rasterio.open(target_dem) as src:
                window = from_bounds(
                    self.bounds["min_lon"],
                    self.bounds["min_lat"],
                    self.bounds["max_lon"],
                    self.bounds["max_lat"],
                    src.transform,
                )
                data = src.read(
                    1,
                    window=window,
                    out_shape=(self.grid_size, self.grid_size),
                    resampling=Resampling.bilinear,
                )
            data = np.nan_to_num(data, nan=self.ELEV_MIN)
            print(f"  [OK] DEM loaded: elevation {data.min():.0f}m - {data.max():.0f}m")
            return data.astype(np.float64)
        else:
            print("\n  [FALLBACK] Using synthetic elevation (DEM not available)")
            return self._synthetic_elevation()

    def _calculate_slope(self, elevation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate slope from elevation in degrees using real geographic metric pixel spacing.
        
        Returns:
            slope_norm: normalized and inverted (flat = 1.0, steep = 0.0)
            slope_deg: raw slope angle in degrees [0, MAX_SLOPE_DEG]
        """
        # Calculate real physical distance per grid pixel in meters (WGS84 ellipsoid approx)
        delta_lat = abs(self.bounds["max_lat"] - self.bounds["min_lat"])
        delta_lon = abs(self.bounds["max_lon"] - self.bounds["min_lon"])
        mid_lat = (self.bounds["max_lat"] + self.bounds["min_lat"]) / 2.0

        # 1 deg latitude ≈ 111,139 meters; 1 deg longitude ≈ 111,320 * cos(mid_lat)
        dy_meters = max(1.0, (delta_lat * 111139.0) / float(self.grid_size))
        dx_meters = max(1.0, (delta_lon * 111320.0 * np.cos(np.radians(mid_lat))) / float(self.grid_size))

        dy, dx = np.gradient(elevation, dy_meters, dx_meters)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)
        slope_deg = np.clip(slope_deg, 0.0, self.MAX_SLOPE_DEG)

        # Normalized and inverted (flat = high suitability: 1.0 at 0 deg, 0.0 at MAX_SLOPE_DEG)
        slope_norm = 1.0 - np.clip(slope_deg / self.MAX_SLOPE_DEG, 0.0, 1.0)
        return slope_norm, slope_deg

    # ──────────────────────────────────────────────────────────────
    # School Proximity (Distance to existing schools — want FAR)
    # ──────────────────────────────────────────────────────────────

    def _load_school_proximity(self) -> np.ndarray:
        """Load existing school locations and compute distance.
        
        Sites farther from existing schools are preferred (avoid duplication).
        Returns: normalized array where far_from_school = high score.
        """
        print("\n  Loading existing school locations...")

        if self._gpd is None or not self._file_exists(self.POIS_POINT_FILE):
            print("  [FALLBACK] Using synthetic school proximity")
            return self._synthetic_poi_proximity("school")

        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        gdf = gpd.read_file(
            self._file_path(self.POIS_POINT_FILE),
            bbox=(self.bounds["min_lon"], self.bounds["min_lat"],
                  self.bounds["max_lon"], self.bounds["max_lat"]),
        )

        schools = gdf[gdf["fclass"] == "school"]
        if schools.empty:
            print("  [WARN] No schools found. Using synthetic.")
            return self._synthetic_poi_proximity("school")

        print(f"  Found {len(schools)} existing schools in Matale")

        transform = from_bounds(
            self.bounds["min_lon"], self.bounds["min_lat"],
            self.bounds["max_lon"], self.bounds["max_lat"],
            self.grid_size, self.grid_size,
        )

        # Buffer points in metric CRS (250m radius) to avoid geographic CRS distortion
        if schools.crs is None:
            schools = schools.set_crs(epsg=4326)
        schools_proj = schools.to_crs(epsg=3857)
        schools_buffered = schools_proj.copy()
        schools_buffered["geometry"] = schools_proj.geometry.buffer(250.0)
        schools_buffered = schools_buffered.to_crs(epsg=4326)

        school_raster = rasterize(
            [(geom, 1) for geom in schools_buffered.geometry if geom is not None],
            out_shape=(self.grid_size, self.grid_size),
            transform=transform, fill=0, dtype=np.uint8,
        )

        # Distance from schools — FAR = high score (avoid duplication)
        no_school = school_raster == 0
        distance = distance_transform_edt(no_school)
        result = self._normalize(distance)  # NOT inverted: far = high
        print(f"  [OK] School distance: {int(school_raster.sum())} school pixels")
        return result

    # ──────────────────────────────────────────────────────────────
    # Healthcare Proximity (Distance to hospitals — want NEAR)
    # ──────────────────────────────────────────────────────────────

    def _load_healthcare_proximity(self) -> np.ndarray:
        """Load hospital locations and compute proximity.
        
        Sites closer to hospitals are preferred (emergency access).
        Returns: normalized array where near_hospital = high score.
        """
        print("\n  Loading healthcare facility locations...")

        if self._gpd is None or not self._file_exists(self.POIS_POINT_FILE):
            print("  [FALLBACK] Using synthetic healthcare proximity")
            return self._synthetic_poi_proximity("hospital")

        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        gdf = gpd.read_file(
            self._file_path(self.POIS_POINT_FILE),
            bbox=(self.bounds["min_lon"], self.bounds["min_lat"],
                  self.bounds["max_lon"], self.bounds["max_lat"]),
        )

        hospitals = gdf[gdf["fclass"] == "hospital"]
        if hospitals.empty:
            print("  [WARN] No hospitals found. Using synthetic.")
            return self._synthetic_poi_proximity("hospital")

        print(f"  Found {len(hospitals)} hospitals in Matale")

        transform = from_bounds(
            self.bounds["min_lon"], self.bounds["min_lat"],
            self.bounds["max_lon"], self.bounds["max_lat"],
            self.grid_size, self.grid_size,
        )

        # Buffer points in metric CRS (250m radius) to avoid geographic CRS distortion
        if hospitals.crs is None:
            hospitals = hospitals.set_crs(epsg=4326)
        hospitals_proj = hospitals.to_crs(epsg=3857)
        hospitals_buffered = hospitals_proj.copy()
        hospitals_buffered["geometry"] = hospitals_proj.geometry.buffer(250.0)
        hospitals_buffered = hospitals_buffered.to_crs(epsg=4326)

        hospital_raster = rasterize(
            [(geom, 1) for geom in hospitals_buffered.geometry if geom is not None],
            out_shape=(self.grid_size, self.grid_size),
            transform=transform, fill=0, dtype=np.uint8,
        )

        # Distance from hospitals — NEAR = high score (inverted)
        no_hospital = hospital_raster == 0
        distance = distance_transform_edt(no_hospital)
        result = 1.0 - self._normalize(distance)  # Inverted: close = high
        print(f"  [OK] Healthcare proximity: {int(hospital_raster.sum())} hospital pixels")
        return result

    # ──────────────────────────────────────────────────────────────
    # Land Use Suitability (residential/farmland=good, forest=bad)
    # ──────────────────────────────────────────────────────────────

    def _load_landuse(self) -> np.ndarray:
        """Load land use types and assign suitability scores.
        
        Scoring: residential=1.0, farmland=0.8, grass/meadow=0.6,
                 commercial=0.5, industrial=0.3, forest=0.1
        Returns: normalized suitability array.
        """
        print("\n  Loading land use classification...")

        if self._gpd is None or not self._file_exists(self.LANDUSE_FILE):
            print("  [FALLBACK] Using synthetic land use")
            return self._synthetic_landuse()

        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        gdf = gpd.read_file(
            self._file_path(self.LANDUSE_FILE),
            bbox=(self.bounds["min_lon"], self.bounds["min_lat"],
                  self.bounds["max_lon"], self.bounds["max_lat"]),
        )

        if gdf.empty:
            print("  [WARN] No land use data found. Using synthetic.")
            return self._synthetic_landuse()

        # Suitability scores by land use type
        lu_scores = {
            "residential": 1.0, "farmland": 0.8, "grass": 0.7,
            "meadow": 0.7, "farmyard": 0.7, "orchard": 0.6,
            "commercial": 0.5, "retail": 0.5, "recreation_ground": 0.5,
            "park": 0.4, "scrub": 0.3, "industrial": 0.2,
            "heath": 0.2, "forest": 0.1, "cemetery": 0.0,
            "military": 0.0, "quarry": 0.0, "landfill": 0.0,
        }

        transform = from_bounds(
            self.bounds["min_lon"], self.bounds["min_lat"],
            self.bounds["max_lon"], self.bounds["max_lat"],
            self.grid_size, self.grid_size,
        )

        shapes = []
        for _, row in gdf.iterrows():
            if row.geometry is not None:
                score = lu_scores.get(row["fclass"], 0.5)
                shapes.append((row.geometry, score))

        lu_raster = rasterize(
            shapes,
            out_shape=(self.grid_size, self.grid_size),
            transform=transform,
            fill=0.5,  # Default: moderate suitability for unclassified
            dtype=np.float64,
        )

        lu_counts = {k: int((gdf["fclass"] == k).sum()) for k in lu_scores if (gdf["fclass"] == k).sum() > 0}
        top3 = sorted(lu_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{k}({v})" for k, v in top3)
        print(f"  [OK] Land use loaded: {len(gdf)} polygons. Top: {top_str}")
        return lu_raster

    # ──────────────────────────────────────────────────────────────
    # Transport Access (Distance to bus stops — want NEAR)
    # ──────────────────────────────────────────────────────────────

    def _load_transport_access(self) -> np.ndarray:
        """Load transport points (bus stops, stations) and compute proximity.
        
        Sites closer to public transport are preferred.
        Returns: normalized array where near_transport = high score.
        """
        print("\n  Loading transport access points...")

        if self._gpd is None or not self._file_exists(self.TRANSPORT_FILE):
            print("  [FALLBACK] Using synthetic transport access")
            return self._synthetic_poi_proximity("transport")

        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import from_bounds

        gdf = gpd.read_file(
            self._file_path(self.TRANSPORT_FILE),
            bbox=(self.bounds["min_lon"], self.bounds["min_lat"],
                  self.bounds["max_lon"], self.bounds["max_lat"]),
        )

        if gdf.empty:
            print("  [WARN] No transport points found. Using synthetic.")
            return self._synthetic_poi_proximity("transport")

        transport_types = ["bus_stop", "bus_station", "railway_station", "railway_halt"]
        transport = gdf[gdf["fclass"].isin(transport_types)]
        if transport.empty:
            transport = gdf  # Use all transport points

        print(f"  Found {len(transport)} transport points in Matale")

        transform = from_bounds(
            self.bounds["min_lon"], self.bounds["min_lat"],
            self.bounds["max_lon"], self.bounds["max_lat"],
            self.grid_size, self.grid_size,
        )

        # Buffer points in metric CRS (250m radius) to avoid geographic CRS distortion
        if transport.crs is None:
            transport = transport.set_crs(epsg=4326)
        transport_proj = transport.to_crs(epsg=3857)
        transport_buffered = transport_proj.copy()
        transport_buffered["geometry"] = transport_proj.geometry.buffer(250.0)
        transport_buffered = transport_buffered.to_crs(epsg=4326)

        transport_raster = rasterize(
            [(geom, 1) for geom in transport_buffered.geometry if geom is not None],
            out_shape=(self.grid_size, self.grid_size),
            transform=transform, fill=0, dtype=np.uint8,
        )

        # Distance from transport — NEAR = high score (inverted)
        no_transport = transport_raster == 0
        distance = distance_transform_edt(no_transport)
        result = 1.0 - self._normalize(distance)  # Inverted: close = high
        print(f"  [OK] Transport access: {int(transport_raster.sum())} transport pixels")
        return result

    # ──────────────────────────────────────────────────────────────
    # Synthetic Fallbacks
    # ──────────────────────────────────────────────────────────────

    def _synthetic_elevation(self) -> np.ndarray:
        """Generate synthetic elevation for Matale (hilly terrain)."""
        gs = self.grid_size
        x = np.linspace(0, 4 * np.pi, gs)
        y = np.linspace(0, 4 * np.pi, gs)
        X, Y = np.meshgrid(x, y)
        terrain = (
            np.sin(X) * np.cos(Y) * 0.4
            + np.sin(2 * X + 1) * np.cos(1.5 * Y + 0.5) * 0.3
            + self.rng.normal(0, 0.05, (gs, gs))
        )
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
        elevation = self.ELEV_MIN + terrain * (self.ELEV_MAX - self.ELEV_MIN)
        print(f"  Synthetic elevation: {elevation.min():.0f}m - {elevation.max():.0f}m")
        return elevation

    def _synthetic_population(self) -> np.ndarray:
        """Generate synthetic population distribution."""
        gs = self.grid_size
        x = np.linspace(-1, 1, gs)
        y = np.linspace(-1, 1, gs)
        X, Y = np.meshgrid(x, y)
        pop = np.exp(-((X - 0.2)**2 + (Y + 0.1)**2) / 0.3)
        pop += 0.5 * np.exp(-((X + 0.3)**2 + (Y - 0.4)**2) / 0.2)
        return self._normalize(pop)

    def _synthetic_road_proximity(self) -> np.ndarray:
        """Generate synthetic road proximity."""
        gs = self.grid_size
        road_mask = np.zeros((gs, gs), dtype=bool)
        road_mask[gs // 3, :] = True
        road_mask[:, gs // 2] = True
        road_mask[2 * gs // 3, :] = True
        distance = distance_transform_edt(~road_mask)
        return 1.0 - self._normalize(distance)

    def _synthetic_constraints(self) -> np.ndarray:
        """Generate synthetic constraint mask."""
        gs = self.grid_size
        mask = np.ones((gs, gs), dtype=np.float64)
        # Circular water body
        cx, cy, r = gs // 3, gs // 3, gs // 6
        Y, X = np.ogrid[:gs, :gs]
        circle = (X - cx)**2 + (Y - cy)**2 <= r**2
        mask[circle] = 0.0
        self.water_mask = circle.astype(np.uint8)
        self.protected_mask = np.zeros((gs, gs), dtype=np.uint8)
        self.waterways_mask = np.zeros((gs, gs), dtype=np.uint8)
        self.pofw_mask = np.zeros((gs, gs), dtype=np.uint8)
        # Edge buffer
        mask[:self.EDGE_BUFFER_PX, :] = 0.0
        mask[-self.EDGE_BUFFER_PX:, :] = 0.0
        mask[:, :self.EDGE_BUFFER_PX] = 0.0
        mask[:, -self.EDGE_BUFFER_PX:] = 0.0
        return mask

    def _synthetic_poi_proximity(self, poi_type: str = "school") -> np.ndarray:
        """Generate synthetic POI proximity layer."""
        gs = self.grid_size
        poi_mask = np.zeros((gs, gs), dtype=bool)
        # Scatter random POI points
        n_pois = {"school": 15, "hospital": 8, "transport": 25}.get(poi_type, 10)
        for _ in range(n_pois):
            px, py = self.rng.randint(10, gs - 10, size=2)
            poi_mask[max(0, px-2):px+3, max(0, py-2):py+3] = True
        distance = distance_transform_edt(~poi_mask)
        if poi_type == "school":
            return self._normalize(distance)  # far = high
        else:
            return 1.0 - self._normalize(distance)  # near = high

    def _synthetic_landuse(self) -> np.ndarray:
        """Generate synthetic land use suitability."""
        gs = self.grid_size
        # Gradient: residential (high) at bottom, forest (low) at top
        y = np.linspace(0.8, 0.3, gs)
        x = np.linspace(0.5, 0.7, gs)
        X, Y = np.meshgrid(x, y)
        lu = (X + Y) / 2 + self.rng.normal(0, 0.05, (gs, gs))
        return np.clip(lu, 0, 1)

    # ──────────────────────────────────────────────────────────────
    # Main Entry Point
    # ──────────────────────────────────────────────────────────────

    def generate_all(self) -> Dict[str, np.ndarray]:
        """
        Load all data layers and return normalized rasters.

        Returns dict with 9 layers:
            - elevation, slope, road_proximity, population_growth,
              constraint_mask (original 5)
            - school_distance, healthcare_proximity, landuse_suitability,
              transport_access (4 new criteria)
        """
        print("=" * 60)
        print("  Loading Real Geospatial Data — Matale District")
        print("=" * 60)
        print(f"  Bounds: Lat [{self.bounds['min_lat']}, {self.bounds['max_lat']}]")
        print(f"          Lon [{self.bounds['min_lon']}, {self.bounds['max_lon']}]")
        print(f"  Grid:   {self.grid_size} x {self.grid_size}")

        # Original layers
        elevation = self._load_elevation()
        slope_norm, slope_raw = self._calculate_slope(elevation)
        road_proximity = self._load_roads()
        population = self._load_population()
        constraint_mask = self._load_constraints()

        # New criteria layers
        school_distance = self._load_school_proximity()
        healthcare_proximity = self._load_healthcare_proximity()
        landuse_suitability = self._load_landuse()
        transport_access = self._load_transport_access()

        layers = {
            "elevation": elevation,
            "slope": slope_norm,
            "road_proximity": road_proximity,
            "population_growth": population,
            "constraint_mask": constraint_mask,
            # Sub-constraint diagnostic masks
            "water_mask": self.water_mask,
            "protected_mask": self.protected_mask,
            "waterways_mask": self.waterways_mask,
            "pofw_mask": self.pofw_mask,
            # New layers
            "school_distance": school_distance,
            "healthcare_proximity": healthcare_proximity,
            "landuse_suitability": landuse_suitability,
            "transport_access": transport_access,
        }

        # Print summary
        print("\n" + "-" * 60)
        print("  Data Layer Summary:")
        print("-" * 60)
        for name, arr in layers.items():
            print(
                f"  {name:25s} | Min: {arr.min():8.3f} | Max: {arr.max():8.3f} "
                f"| Mean: {arr.mean():8.3f} | Shape: {arr.shape}"
            )

        return layers

    def calculate_slope(self, elevation: np.ndarray) -> np.ndarray:
        """Calculate raw slope in degrees (for GA/PSO constraint checking)."""
        _, slope_deg = self._calculate_slope(elevation)
        return slope_deg


# ──────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader = RealDataLoader(data_dir="data", grid_size=200)
    layers = loader.generate_all()

    # Plot all layers (8 criteria + constraints + combined)
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    plot_items = [
        ("elevation", "Elevation (m)", "terrain"),
        ("slope", "Slope Safety (flat=high)", "viridis"),
        ("road_proximity", "Road Proximity", "viridis"),
        ("population_growth", "Population Density", "viridis"),
        ("school_distance", "School Distance (far=high)", "viridis"),
        ("healthcare_proximity", "Healthcare Proximity", "viridis"),
        ("landuse_suitability", "Land Use Suitability", "viridis"),
        ("transport_access", "Transport Access", "viridis"),
        ("constraint_mask", "Constraint Mask (1=buildable)", "gray"),
    ]

    for ax, (name, title, cmap_name) in zip(axes.flat[:9], plot_items):
        im = ax.imshow(layers[name], cmap=cmap_name, origin="lower")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Combined suitability (8 criteria with AHP weights)
    combined = (
        0.3436 * layers["road_proximity"]
        + 0.1553 * layers["slope"]
        + 0.0653 * layers["population_growth"]
        + 0.0300 * layers["school_distance"]
        + 0.1553 * layers["healthcare_proximity"]
        + 0.0653 * layers["landuse_suitability"]
        + 0.1553 * layers["transport_access"]
        + 0.0300 * (layers["population_growth"] * 0.8)
    ) * layers["constraint_mask"]

    ax_last = axes.flat[9]
    im_last = ax_last.imshow(combined, cmap="hot", origin="lower")
    ax_last.set_title("8-Criteria Combined Suitability", fontsize=10, fontweight="bold", color="darkred")
    ax_last.axis("off")
    plt.colorbar(im_last, ax=ax_last, fraction=0.046, pad=0.04)

    plt.suptitle("8 Multi-Criteria Geospatial Layers — Matale District, Sri Lanka", fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/real_data_preview.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nSaved preview to outputs/real_data_preview.png")
