"""Module for building composite 3D suitability surfaces from weighted criteria."""

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


class SuitabilitySurface:
    """Builds and analyzes a composite 3D suitability surface.

    Args:
        criteria_rasters (dict[str, np.ndarray]): Dictionary mapping criterion names to their
            normalized raster grids (values in [0, 1]).
        weights (np.ndarray): AHP-derived weights in the same order as criteria_rasters keys.
        constraint_mask (np.ndarray): Binary mask where 0 = restricted, 1 = buildable.
        elevation (np.ndarray): Raw elevation grid for z-value lookup.
    """

    def __init__(
        self,
        criteria_rasters: Dict[str, np.ndarray],
        weights: np.ndarray,
        constraint_mask: np.ndarray,
        elevation: np.ndarray,
    ):
        """Initializes the SuitabilitySurface."""
        self.criteria_rasters = criteria_rasters
        self.weights = weights if isinstance(weights, np.ndarray) else np.array(weights)
        self.constraint_mask = constraint_mask
        self.elevation = elevation
        self.suitability_matrix: np.ndarray | None = None

    def build_surface(self) -> np.ndarray:
        """Performs weighted linear combination of criteria rasters.

        Validates that all rasters have the same shape, weights length matches number
        of criteria, and raster values are in [0, 1].

        Returns:
            np.ndarray: The raw weighted suitability surface before constraint masking.
            
        Raises:
            ValueError: If raster shapes mismatch or weights length is incorrect.
        """
        if len(self.criteria_rasters) != len(self.weights):
            raise ValueError(
                f"Number of criteria ({len(self.criteria_rasters)}) "
                f"must match weights length ({len(self.weights)})."
            )

        raster_shape = self.elevation.shape
        if self.constraint_mask.shape != raster_shape:
            raise ValueError("Constraint mask shape must match elevation shape.")

        surface = np.zeros(raster_shape)

        for i, (name, raster) in enumerate(self.criteria_rasters.items()):
            if raster.shape != raster_shape:
                raise ValueError(f"Raster '{name}' shape {raster.shape} does not match {raster_shape}.")
            
            if np.any((raster < 0) | (raster > 1)):
                warnings.warn(f"Raster '{name}' values outside [0, 1]. Clipping.")
                raster = np.clip(raster, 0, 1)

            surface += self.weights[i] * raster

        return surface

    def apply_constraints(self, surface: np.ndarray) -> np.ndarray:
        """Multiplies the surface by the constraint mask element-wise.

        Args:
            surface (np.ndarray): The raw suitability surface.

        Returns:
            np.ndarray: The constrained suitability surface.
        """
        return surface * self.constraint_mask

    def get_suitability_matrix(self) -> np.ndarray:
        """Builds the surface, applies constraints, and stores the result.

        Returns:
            np.ndarray: The final suitability matrix.
        """
        raw_surface = self.build_surface()
        self.suitability_matrix = self.apply_constraints(raw_surface)

        valid_cells = self.suitability_matrix[self.suitability_matrix > 0]
        if valid_cells.size > 0:
            print("Suitability Summary:")
            print(f"  Min:  {valid_cells.min():.4f}")
            print(f"  Max:  {valid_cells.max():.4f}")
            print(f"  Mean: {valid_cells.mean():.4f}")
            print(f"  Std:  {valid_cells.std():.4f}")
        else:
            print("Suitability Summary: No valid non-zero cells.")

        return self.suitability_matrix

    def get_suitability_3d(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets the 3D representation of the suitability surface.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the suitability matrix
                and the elevation grid.
        """
        if self.suitability_matrix is None:
            self.get_suitability_matrix()
        
        return self.suitability_matrix, self.elevation # type: ignore

    def lookup(self, x: int, y: int) -> Dict[str, Any]:
        """Looks up suitability and elevation at given integer grid coordinates.

        Args:
            x (int): Row index.
            y (int): Column index.

        Returns:
            Dict[str, Any]: Dictionary containing x, y, suitability, elevation,
                and is_constrained boolean.
        """
        if self.suitability_matrix is None:
            self.get_suitability_matrix()

        suitability = float(self.suitability_matrix[x, y]) # type: ignore
        elev = float(self.elevation[x, y])
        is_constrained = bool(self.constraint_mask[x, y] == 0)

        return {
            'x': x,
            'y': y,
            'suitability': suitability,
            'elevation': elev,
            'is_constrained': is_constrained,
        }

    def find_top_k(self, k: int = 10) -> List[Dict[str, Any]]:
        """Finds the top-k highest suitability cells.

        Args:
            k (int, optional): Number of top cells to find. Defaults to 10.

        Returns:
            List[Dict[str, Any]]: List of dictionaries for the top-k cells, sorted
                by suitability descending.
        """
        if self.suitability_matrix is None:
            self.get_suitability_matrix()
            
        flat_indices = np.argsort(self.suitability_matrix.flatten())[::-1] # type: ignore
        top_k_indices = flat_indices[:k]

        results = []
        rows, cols = self.suitability_matrix.shape # type: ignore
        for idx in top_k_indices:
            x, y = divmod(idx, cols)
            results.append({
                'x': int(x),
                'y': int(y),
                'suitability': float(self.suitability_matrix[x, y]), # type: ignore
                'elevation': float(self.elevation[x, y])
            })
            
        return results


if __name__ == '__main__':
    import os
    from data_generator import SyntheticRasterGenerator
    from ahp_engine import AHPEngine

    # Generate synthetic data
    generator = SyntheticRasterGenerator(grid_size=200, random_seed=42)
    layers = generator.generate_all()

    # Run AHP
    criteria_names = [
        "Road Proximity",
        "Slope Safety",
        "Future Growth Velocity",
        "Existing Amenity Buffer",
    ]
    ahp = AHPEngine(criteria_names=criteria_names)
    weights, _ = ahp.run()

    # Map criteria to rasters (same order as criteria_names)
    criteria_rasters = {
        "Road Proximity": layers["road_proximity"],
        "Slope Safety": layers["slope"],
        "Future Growth Velocity": layers["population_growth"],
        "Existing Amenity Buffer": layers["population_growth"] * 0.8,  # proxy
    }

    surface = SuitabilitySurface(
        criteria_rasters=criteria_rasters,
        weights=weights,
        constraint_mask=layers["constraint_mask"],
        elevation=layers["elevation"],
    )
    matrix = surface.get_suitability_matrix()

    print("\nTop 10 Sites:")
    for site in surface.find_top_k(10):
        print(site)

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="viridis", origin="lower")
    plt.colorbar(label="Suitability Score")
    plt.title("Composite Suitability Surface — Matale District")
    plt.savefig("outputs/suitability_preview.png", dpi=200, bbox_inches="tight")
    print("\nSaved suitability preview to outputs/suitability_preview.png")
