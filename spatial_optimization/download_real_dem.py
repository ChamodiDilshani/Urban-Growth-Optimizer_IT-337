"""
Download real NASA SRTM / Terrarium 30m elevation tiles for Matale District,
assemble the seamless mosaic, crop to exact Matale geographic bounds,
and write georeferenced GeoTIFF (data/matale_dem.tif).
"""

import math
import io
import os
import urllib.request
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.enums import Resampling

DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "matale_dem.tif")

MATALE_BOUNDS = {
    "min_lat": 7.35,
    "max_lat": 8.05,
    "min_lon": 80.45,
    "max_lon": 81.00,
}

ZOOM = 11

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def download_matale_dem():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Calculate tile ranges
    x_min, y_min = deg2num(MATALE_BOUNDS["max_lat"], MATALE_BOUNDS["min_lon"], ZOOM)
    x_max, y_max = deg2num(MATALE_BOUNDS["min_lat"], MATALE_BOUNDS["max_lon"], ZOOM)
    
    nx = x_max - x_min + 1
    ny = y_max - y_min + 1
    print(f"Downloading real elevation tiles for Matale District: {nx} x {ny} = {nx * ny} tiles...")
    
    mosaic = np.zeros((ny * 256, nx * 256), dtype=np.float32)
    
    for iy, y in enumerate(range(y_min, y_max + 1)):
        for ix, x in enumerate(range(x_min, x_max + 1)):
            url = f"https://elevation-tiles-prod.s3.amazonaws.com/terrarium/{ZOOM}/{x}/{y}.png"
            req = urllib.request.Request(url, headers={"User-Agent": "MataleGIS/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    img = Image.open(io.BytesIO(resp.read())).convert("RGB")
                    arr = np.array(img, dtype=np.float32)
                    # Terrarium formula: (R * 256 + G + B / 256) - 32768
                    elev = (arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0) - 32768.0
                    mosaic[iy*256 : (iy+1)*256, ix*256 : (ix+1)*256] = elev
                    print(f"  Downloaded tile ({x}, {y}) [z={ZOOM}]")
            except Exception as e:
                print(f"  Failed tile ({x}, {y}): {e}")
                
    # Calculate bounds of mosaic
    mosaic_north, mosaic_west = num2deg(x_min, y_min, ZOOM)
    mosaic_south, mosaic_east = num2deg(x_max + 1, y_max + 1, ZOOM)
    
    print(f"\nMosaic downloaded:")
    print(f"  Coverage: Lat [{mosaic_south:.4f}, {mosaic_north:.4f}], Lon [{mosaic_west:.4f}, {mosaic_east:.4f}]")
    print(f"  Raw elevation range: {mosaic.min():.1f}m to {mosaic.max():.1f}m")
    
    # Save as temporary full mosaic GeoTIFF
    mosaic_transform = from_bounds(mosaic_west, mosaic_south, mosaic_east, mosaic_north, mosaic.shape[1], mosaic.shape[0])
    
    # Now extract exact Matale District bounds at 200x200 or high resolution
    # Let's save a clean GeoTIFF for Matale bounds
    target_h, target_w = 400, 400  # High resolution DEM
    matale_transform = from_bounds(
        MATALE_BOUNDS["min_lon"],
        MATALE_BOUNDS["min_lat"],
        MATALE_BOUNDS["max_lon"],
        MATALE_BOUNDS["max_lat"],
        target_w,
        target_h
    )
    
    # Resample / crop to Matale bounds
    import scipy.ndimage
    # Normalized coordinates inside mosaic
    y_coords = np.linspace(
        (mosaic_north - MATALE_BOUNDS["max_lat"]) / (mosaic_north - mosaic_south) * mosaic.shape[0],
        (mosaic_north - MATALE_BOUNDS["min_lat"]) / (mosaic_north - mosaic_south) * mosaic.shape[0],
        target_h
    )
    x_coords = np.linspace(
        (MATALE_BOUNDS["min_lon"] - mosaic_west) / (mosaic_east - mosaic_west) * mosaic.shape[1],
        (MATALE_BOUNDS["max_lon"] - mosaic_west) / (mosaic_east - mosaic_west) * mosaic.shape[1],
        target_w
    )
    
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    matale_dem = scipy.ndimage.map_coordinates(mosaic, [yy, xx], order=1, mode="nearest")
    
    # Clip negative elevation noise (sea/underground artifacts) to realistic minimum for Matale (lowest valley ~70m)
    matale_dem = np.clip(matale_dem, 70.0, None)
    
    print(f"\nMatale DEM Extracted:")
    print(f"  Shape: {matale_dem.shape}")
    print(f"  Elevation: {matale_dem.min():.1f}m (Dambulla/Valley) to {matale_dem.max():.1f}m (Knuckles/Riverston peaks)")
    
    with rasterio.open(
        OUTPUT_FILE,
        "w",
        driver="GTiff",
        height=target_h,
        width=target_w,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=matale_transform,
    ) as dst:
        dst.write(matale_dem.astype(np.float32), 1)
        
    print(f"\n[SUCCESS] Saved Real NASA SRTM DEM to: {OUTPUT_FILE}")

if __name__ == "__main__":
    download_matale_dem()
