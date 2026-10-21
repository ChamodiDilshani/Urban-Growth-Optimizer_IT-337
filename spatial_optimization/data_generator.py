import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from typing import Dict


class SyntheticRasterGenerator:
    """
    Generates synthetic geospatial raster data for the Matale District, Sri Lanka.
    Simulates realistic terrain and geospatial variables for spatial optimization modeling.
    
    Attributes:
        grid_size (int): The height and width of the square raster grid.
        random_seed (int): The seed for the random number generator.
        rng (np.random.RandomState): The random state used for generating data.
    """
    
    # Constants for elevation generation (in meters above sea level for Matale area)
    ELEV_MIN: float = 250.0
    ELEV_MAX: float = 850.0
    
    # Constants for slope generation
    MAX_SLOPE_DEG: float = 45.0
    
    # Constants for spatial constraints
    EDGE_BUFFER_PX: int = 10
    
    def __init__(self, grid_size: int = 200, random_seed: int = 42):
        """
        Initialize the synthetic raster generator.
        
        Args:
            grid_size (int): Size of the square grid (N x N). Defaults to 200.
            random_seed (int): Seed for reproducibility. Defaults to 42.
        """
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.rng = np.random.RandomState(self.random_seed)

    def generate_elevation(self) -> np.ndarray:
        """
        Generate a realistic DEM grid simulating terrain using overlapping sine/cosine waves.
        
        Returns:
            np.ndarray: Elevation array of shape (grid_size, grid_size) ranging from 80m to 350m.
        """
        try:
            x = np.linspace(0, 10, self.grid_size)
            y = np.linspace(0, 10, self.grid_size)
            X, Y = np.meshgrid(x, y)
            
            # Combine multiple frequencies to create natural-looking terrain
            base_terrain = (
                np.sin(X * 0.8) + 
                np.cos(Y * 0.7) + 
                np.sin(X * 2.5 + Y * 1.5) * 0.4 + 
                np.cos(X * 1.1 - Y * 2.1) * 0.3
            )
            
            # Normalize to [0, 1]
            base_norm = (base_terrain - base_terrain.min()) / (base_terrain.max() - base_terrain.min())
            
            # Scale to realistic elevations [ELEV_MIN, ELEV_MAX]
            elevation = base_norm * (self.ELEV_MAX - self.ELEV_MIN) + self.ELEV_MIN
            
            # Add micro-terrain noise (Gaussian)
            noise = self.rng.normal(loc=0.0, scale=3.0, size=(self.grid_size, self.grid_size))
            elevation += noise
            
            return np.clip(elevation, self.ELEV_MIN, self.ELEV_MAX)
        except Exception as e:
            raise RuntimeError(f"Failed to generate elevation: {str(e)}")

    def calculate_slope(self, elevation: np.ndarray) -> np.ndarray:
        """
        Calculate slope gradients in degrees from an elevation grid.
        
        Args:
            elevation (np.ndarray): The generated elevation grid.
            
        Returns:
            np.ndarray: Slope array of shape (grid_size, grid_size) in degrees, clipped to [0, 45].
        """
        try:
            # np.gradient returns (dy, dx) differences
            dy, dx = np.gradient(elevation)
            
            # Calculate slope in radians, then convert to degrees
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            
            return np.clip(slope_deg, 0.0, self.MAX_SLOPE_DEG)
        except Exception as e:
            raise RuntimeError(f"Failed to calculate slope: {str(e)}")

    def generate_road_proximity(self) -> np.ndarray:
        """
        Create a synthetic road network and compute Euclidean distance transform.
        Closer to the road gives a higher value (inverted normalized distance).
        
        Returns:
            np.ndarray: Road proximity array of shape (grid_size, grid_size), values [0, 1].
        """
        try:
            road_mask = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            
            # Horizontal road
            road_mask[self.grid_size // 3, :] = True
            # Vertical road
            road_mask[:, self.grid_size // 2] = True
            # Diagonal road
            np.fill_diagonal(road_mask, True)
            
            # Compute distance from roads
            # distance_transform_edt computes distance to nearest zero value, so we invert the mask
            dist = distance_transform_edt(~road_mask)
            
            # Normalize to [0, 1] where 0 is on the road, 1 is farthest
            norm_dist = dist / dist.max()
            
            # Invert: Closer to road = higher value
            road_proximity = 1.0 - norm_dist
            return road_proximity
        except Exception as e:
            raise RuntimeError(f"Failed to generate road proximity: {str(e)}")

    def generate_population_growth(self) -> np.ndarray:
        """
        Create a population/urban growth score concentrated around a city center.
        Uses 2D Gaussians to simulate a main city and a satellite town.
        
        Returns:
            np.ndarray: Population growth array of shape (grid_size, grid_size), values [0, 1].
        """
        try:
            def make_gaussian(cx: int, cy: int, sigma: float) -> np.ndarray:
                x = np.arange(self.grid_size)
                y = np.arange(self.grid_size)
                X, Y = np.meshgrid(x, y)
                return np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
            
            # Main city center
            main_center = make_gaussian(self.grid_size // 2, self.grid_size // 2, self.grid_size / 4)
            
            # Satellite town offset to the top right
            sat_center = make_gaussian(int(self.grid_size * 0.75), int(self.grid_size * 0.25), self.grid_size / 8)
            
            pop_growth = main_center + 0.6 * sat_center
            
            # Normalize to [0, 1]
            return pop_growth / pop_growth.max()
        except Exception as e:
            raise RuntimeError(f"Failed to generate population growth: {str(e)}")

    def generate_constraint_mask(self) -> np.ndarray:
        """
        Create a binary mask identifying buildable (1.0) and restricted (0.0) areas.
        
        Returns:
            np.ndarray: Binary mask array of shape (grid_size, grid_size).
        """
        try:
            mask = np.ones((self.grid_size, self.grid_size), dtype=np.float64)
            
            # Edge buffers
            mask[:self.EDGE_BUFFER_PX, :] = 0.0
            mask[-self.EDGE_BUFFER_PX:, :] = 0.0
            mask[:, :self.EDGE_BUFFER_PX] = 0.0
            mask[:, -self.EDGE_BUFFER_PX:] = 0.0
            
            # Irregular water body (circular region near top-left)
            x = np.arange(self.grid_size)
            y = np.arange(self.grid_size)
            X, Y = np.meshgrid(x, y)
            water_mask = (X - self.grid_size // 4)**2 + (Y - self.grid_size // 4)**2 < (self.grid_size // 6)**2
            mask[water_mask] = 0.0
            
            # Environmental sanctuary (rectangular region near bottom-right)
            sanctuary_start_y = int(self.grid_size * 0.65)
            sanctuary_start_x = int(self.grid_size * 0.70)
            mask[sanctuary_start_y:-self.EDGE_BUFFER_PX, sanctuary_start_x:-self.EDGE_BUFFER_PX] = 0.0
            
            return mask
        except Exception as e:
            raise RuntimeError(f"Failed to generate constraint mask: {str(e)}")

    def generate_all(self) -> Dict[str, np.ndarray]:
        """
        Generate all geospatial layers and compile them into a dictionary.
        Also normalizes and inverts the slope layer for suitability.
        
        Returns:
            Dict[str, np.ndarray]: Dictionary containing all generated synthetic layers.
        """
        print("Generating synthetic geospatial data layers...")
        elevation = self.generate_elevation()
        
        slope_raw = self.calculate_slope(elevation)
        # Normalize to [0, 1] and INVERT so flat terrain (0) has high suitability (1)
        slope_suitability = 1.0 - (slope_raw / self.MAX_SLOPE_DEG)
        
        road_proximity = self.generate_road_proximity()
        population_growth = self.generate_population_growth()
        constraint_mask = self.generate_constraint_mask()
        
        layers = {
            'elevation': elevation,
            'slope': slope_suitability,
            'road_proximity': road_proximity,
            'population_growth': population_growth,
            'constraint_mask': constraint_mask
        }
        
        # Print summary statistics
        print("\nSummary Statistics:")
        for name, data in layers.items():
            print(f"{name:20s} | Min: {data.min():7.3f} | Max: {data.max():7.3f} | Mean: {data.mean():7.3f} | Shape: {data.shape}")
            
        return layers


if __name__ == '__main__':
    # Generate all layers
    generator = SyntheticRasterGenerator(grid_size=200, random_seed=42)
    layers = generator.generate_all()
    
    # Plotting configuration
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    layer_names = list(layers.keys())
    cmaps = ['terrain', 'RdYlGn', 'plasma', 'magma', 'Greys']
    
    for i, (name, cmap) in enumerate(zip(layer_names, cmaps)):
        ax = axes[i]
        im = ax.imshow(layers[name], cmap=cmap)
        ax.set_title(name.replace('_', ' ').title())
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # Create output directory and save
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'synthetic_layers.png')
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved successfully to {out_path}")
