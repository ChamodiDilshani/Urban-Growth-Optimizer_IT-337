"""
Main Benchmark Runner for Multi-Criteria Spatial Optimization.

Executes both Genetic Algorithm (GA) and Particle Swarm Optimization (PSO)
over an identical AHP-weighted suitability surface, measures performance,
and generates comparison visualizations.

Usage:
    python main_benchmark.py
"""

import json
import os
import sys
import time
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Local module imports
from data_generator import SyntheticRasterGenerator
from real_data_loader import RealDataLoader
from ahp_engine import AHPEngine
from suitability_surface import SuitabilitySurface
from genetic_algorithm import SpatialGeneticAlgorithm
from pso_optimizer import SpatialPSO


# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
GRID_SIZE: int = 200
RANDOM_SEED: int = 42
OUTPUT_DIR: str = "outputs"
DATA_DIR: str = "data"

# Toggle between real and synthetic data
# Set to True to use real OSM + WorldPop data from data/ folder
USE_REAL_DATA: bool = True

# AHP criteria (8 criteria — expanded from original 4)
CRITERIA_NAMES: list[str] = [
    "Road Proximity",
    "Slope Safety",
    "Population Density",
    "School Distance",
    "Healthcare Proximity",
    "Land Use Suitability",
    "Transport Access",
    "Amenity Buffer",
]

# 8x8 AHP comparison matrix for school site selection
AHP_MATRIX = np.array([
    #  Road  Slope  Pop   SchDist Health LandUse Trans  Amen
    [1,     3,     5,     7,      3,      5,      3,     7],
    [1/3,   1,     3,     5,      1,      3,      1,     5],
    [1/5,   1/3,   1,     3,      1/3,    1,      1/3,   3],
    [1/7,   1/5,   1/3,   1,      1/5,    1/3,    1/5,   1],
    [1/3,   1,     3,     5,      1,      3,      1,     5],
    [1/5,   1/3,   1,     3,      1/3,    1,      1/3,   3],
    [1/3,   1,     3,     5,      1,      3,      1,     5],
    [1/7,   1/5,   1/3,   1,      1/5,    1/3,    1/5,   1],
])


def setup_output_dir() -> None:
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_spatial_data() -> dict[str, np.ndarray]:
    """Load geospatial raster layers (real or synthetic).

    Returns:
        Dictionary of named raster layers including raw elevation and slope.
    """
    print("=" * 70)
    if USE_REAL_DATA:
        print("STEP 1: Loading REAL Geospatial Data (Matale District)")
    else:
        print("STEP 1: Generating Synthetic Geospatial Data")
    print("=" * 70)

    if USE_REAL_DATA:
        loader = RealDataLoader(
            data_dir=DATA_DIR,
            grid_size=GRID_SIZE,
            random_seed=RANDOM_SEED,
        )
        layers = loader.generate_all()
        layers["slope_raw_degrees"] = loader.calculate_slope(layers["elevation"])
    else:
        generator = SyntheticRasterGenerator(grid_size=GRID_SIZE, random_seed=RANDOM_SEED)
        layers = generator.generate_all()
        layers["slope_raw_degrees"] = generator.calculate_slope(layers["elevation"])

    print(f"\n  Grid dimensions: {GRID_SIZE} x {GRID_SIZE}")
    print(f"  Elevation range: {layers['elevation'].min():.1f} - {layers['elevation'].max():.1f} m")
    print(f"  Raw slope range: {layers['slope_raw_degrees'].min():.1f} - {layers['slope_raw_degrees'].max():.1f} deg")
    return layers


def run_ahp() -> tuple[np.ndarray, dict[str, float]]:
    """Execute Analytic Hierarchy Process for criteria weighting.

    Returns:
        Tuple of (weights array, consistency metrics dict).
    """
    print("\n" + "=" * 70)
    print("STEP 2: Analytic Hierarchy Process (AHP)")
    print("=" * 70)

    ahp = AHPEngine(criteria_names=CRITERIA_NAMES, comparison_matrix=AHP_MATRIX)
    ahp.display_matrix()
    weights, metrics = ahp.run()
    return weights, metrics


def build_suitability_surface(
    layers: dict[str, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Build the weighted composite suitability surface.

    Args:
        layers: Raster data layers from the data generator.
        weights: AHP-derived criteria weights.

    Returns:
        2D suitability matrix with constraint masking applied.
    """
    print("\n" + "=" * 70)
    print("STEP 3: Building Composite Suitability Surface")
    print("=" * 70)

    # Map criteria to raster layers (8 criteria)
    criteria_rasters = {
        "Road Proximity": layers["road_proximity"],
        "Slope Safety": layers["slope"],
        "Population Density": layers["population_growth"],
        "School Distance": layers.get("school_distance", layers["population_growth"]),
        "Healthcare Proximity": layers.get("healthcare_proximity", layers["road_proximity"]),
        "Land Use Suitability": layers.get("landuse_suitability", np.ones_like(layers["slope"]) * 0.5),
        "Transport Access": layers.get("transport_access", layers["road_proximity"]),
        "Amenity Buffer": layers["population_growth"] * 0.8,
    }

    surface = SuitabilitySurface(
        criteria_rasters=criteria_rasters,
        weights=weights,
        constraint_mask=layers["constraint_mask"],
        elevation=layers["elevation"],
    )
    suitability = surface.get_suitability_matrix()

    # Report top sites
    print("\n  Top 5 candidate sites:")
    for i, site in enumerate(surface.find_top_k(5), 1):
        print(
            f"    #{i}  x={site['x']:3d}, y={site['y']:3d}  "
            f"suitability={site['suitability']:.4f}  "
            f"elevation={site['elevation']:.1f} m"
        )

    return suitability


def run_optimizer(
    name: str,
    optimizer_class: type,
    suitability: np.ndarray,
    layers: dict[str, np.ndarray],
    **kwargs: Any,
) -> tuple[dict[str, Any], float]:
    """Run a single optimizer and return results with timing.

    Args:
        name: Display name of the optimizer.
        optimizer_class: The optimizer class to instantiate.
        suitability: 2D suitability matrix.
        layers: Raster data layers.
        **kwargs: Additional keyword arguments for the optimizer.

    Returns:
        Tuple of (results dict, execution time in seconds).
    """
    print(f"\n  Running {name}...")
    opt = optimizer_class(
        suitability_matrix=suitability,
        elevation=layers["elevation"],
        slope=layers["slope_raw_degrees"],
        constraint_mask=layers["constraint_mask"],
        **kwargs,
    )

    start = time.perf_counter()
    results = opt.optimize()
    elapsed = time.perf_counter() - start

    print(f"  {name} completed in {elapsed:.3f} seconds")
    print(f"    Best fitness: {results['best_fitness']:.4f}")
    print(f"    Optimal site: ({results['best_x']:.2f}, {results['best_y']:.2f})")
    print(f"    Elevation:    {results['best_z']:.1f} m")

    return results, elapsed


def run_benchmarks(
    suitability: np.ndarray,
    layers: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Execute both GA and PSO on the same suitability surface.

    Args:
        suitability: 2D suitability matrix.
        layers: Raster data layers.

    Returns:
        Combined benchmark results dictionary.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Running Metaheuristic Optimization Benchmark")
    print("=" * 70)

    # Genetic Algorithm
    ga_results, ga_time = run_optimizer(
        name="Genetic Algorithm (GA)",
        optimizer_class=SpatialGeneticAlgorithm,
        suitability=suitability,
        layers=layers,
        population_size=50,
        generations=100,
        random_seed=RANDOM_SEED,
    )

    # Particle Swarm Optimization
    pso_results, pso_time = run_optimizer(
        name="Particle Swarm Optimization (PSO)",
        optimizer_class=SpatialPSO,
        suitability=suitability,
        layers=layers,
        swarm_size=50,
        iterations=100,
        random_seed=RANDOM_SEED,
    )

    return {
        "ga": {"results": ga_results, "time": ga_time},
        "pso": {"results": pso_results, "time": pso_time},
    }


def export_results_json(benchmark: dict[str, Any]) -> str:
    """Export optimal site results as structured JSON.

    Args:
        benchmark: Combined benchmark results.

    Returns:
        Path to the saved JSON file.
    """
    output = []
    for algo_key, algo_name in [("ga", "Genetic Algorithm"), ("pso", "Particle Swarm Optimization")]:
        r = benchmark[algo_key]["results"]
        output.append({
            "optimal_x": round(r["best_x"], 4),
            "optimal_y": round(r["best_y"], 4),
            "elevation_z": round(r["best_z"], 2),
            "suitability_score": round(r["best_fitness"], 6),
            "algorithm": algo_name,
            "execution_time_seconds": round(benchmark[algo_key]["time"], 4),
        })

    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results exported to {json_path}")
    return json_path


def plot_convergence_curves(benchmark: dict[str, Any]) -> str:
    """Generate and save convergence comparison plot.

    Args:
        benchmark: Combined benchmark results.

    Returns:
        Path to the saved plot.
    """
    ga_hist = [max(0.0, v) for v in benchmark["ga"]["results"]["fitness_history"]]
    pso_hist = [max(0.0, v) for v in benchmark["pso"]["results"]["convergence_history"]]

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(
        range(1, len(ga_hist) + 1), ga_hist,
        color="#e74c3c", linewidth=2, label="Genetic Algorithm (GA)",
        marker="o", markersize=3, alpha=0.8,
    )
    ax.plot(
        range(1, len(pso_hist) + 1), pso_hist,
        color="#3498db", linewidth=2, label="Particle Swarm Optimization (PSO)",
        marker="s", markersize=3, alpha=0.8,
    )

    ax.set_xlabel("Iteration / Generation", fontsize=13)
    ax.set_ylabel("Best Fitness (Suitability Score)", fontsize=13)
    ax.set_title(
        "Convergence Comparison: GA vs PSO\n"
        "Multi-Criteria Infrastructure Site Selection — Matale District, Sri Lanka",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(1, max(len(ga_hist), len(pso_hist)))
    ax.set_ylim(0, 1.05)  # Suitability scores are in [0, 1]

    # Annotate final values
    ax.annotate(
        f"GA best: {ga_hist[-1]:.4f}",
        xy=(len(ga_hist), ga_hist[-1]),
        xytext=(len(ga_hist) - 20, ga_hist[-1] - 0.05),
        fontsize=10, color="#e74c3c",
        arrowprops=dict(arrowstyle="->", color="#e74c3c"),
    )
    ax.annotate(
        f"PSO best: {pso_hist[-1]:.4f}",
        xy=(len(pso_hist), pso_hist[-1]),
        xytext=(len(pso_hist) - 20, pso_hist[-1] + 0.03),
        fontsize=10, color="#3498db",
        arrowprops=dict(arrowstyle="->", color="#3498db"),
    )

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "convergence_curves.png")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Convergence plot saved to {plot_path}")
    return plot_path


def plot_suitability_heatmap(
    suitability: np.ndarray,
    benchmark: dict[str, Any],
) -> str:
    """Generate and save suitability heatmap with optimal sites marked.

    Args:
        suitability: 2D suitability matrix.
        benchmark: Combined benchmark results.

    Returns:
        Path to the saved heatmap.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # Custom colormap: dark → warm
    cmap = LinearSegmentedColormap.from_list(
        "suitability",
        ["#1a1a2e", "#16213e", "#0f3460", "#e94560", "#ffbd69", "#fffde7"],
    )

    im = ax.imshow(suitability, cmap=cmap, origin="lower", interpolation="bilinear")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Suitability Score", fontsize=12)

    # Plot optimal sites
    for algo_key, marker_color, label in [
        ("ga", "#ff0000", "GA Optimal"),
        ("pso", "#00ff00", "PSO Optimal"),
    ]:
        r = benchmark[algo_key]["results"]
        ax.plot(
            r["best_y"], r["best_x"],  # imshow uses (col, row)
            marker="*", markersize=20, color=marker_color,
            markeredgecolor="white", markeredgewidth=1.5,
            label=f"{label} ({r['best_fitness']:.4f})",
        )

    ax.set_xlabel("Y (Column)", fontsize=12)
    ax.set_ylabel("X (Row)", fontsize=12)
    ax.set_title(
        "Spatial Suitability Heatmap with Optimal Infrastructure Sites\n"
        "Matale District, Sri Lanka",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right", facecolor="black", labelcolor="white")

    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "suitability_heatmap.png")
    fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Suitability heatmap saved to {heatmap_path}")
    return heatmap_path


def print_summary(benchmark: dict[str, Any]) -> None:
    """Print a formatted comparison summary table."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'GA':>18} {'PSO':>18}")
    print("-" * 70)

    ga_r = benchmark["ga"]["results"]
    pso_r = benchmark["pso"]["results"]

    print(f"{'Best Fitness Score':<30} {ga_r['best_fitness']:>18.6f} {pso_r['best_fitness']:>18.6f}")
    print(f"{'Optimal X':<30} {ga_r['best_x']:>18.2f} {pso_r['best_x']:>18.2f}")
    print(f"{'Optimal Y':<30} {ga_r['best_y']:>18.2f} {pso_r['best_y']:>18.2f}")
    print(f"{'Elevation Z (m)':<30} {ga_r['best_z']:>18.2f} {pso_r['best_z']:>18.2f}")
    print(f"{'Execution Time (s)':<30} {benchmark['ga']['time']:>18.4f} {benchmark['pso']['time']:>18.4f}")

    # Determine winner
    if ga_r["best_fitness"] > pso_r["best_fitness"]:
        winner = "Genetic Algorithm (GA)"
    elif pso_r["best_fitness"] > ga_r["best_fitness"]:
        winner = "Particle Swarm Optimization (PSO)"
    else:
        winner = "Tie"

    print("-" * 70)
    print(f"{'Winner':<30} {winner:>38}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Execute the full benchmark pipeline."""
    print("\n" + "#" * 70)
    print("#  Multi-Criteria Spatial Optimization for Infrastructure Siting")
    print("#  Component 4: AHP + GA/PSO Benchmark")
    print("#  Region: Matale District, Sri Lanka")
    print("#" * 70)

    setup_output_dir()

    # Step 1: Generate spatial data
    layers = generate_spatial_data()

    # Step 2: AHP weighting
    weights, metrics = run_ahp()

    # Step 3: Build suitability surface
    suitability = build_suitability_surface(layers, weights)

    # Step 4: Run GA & PSO benchmarks
    benchmark = run_benchmarks(suitability, layers)

    # Step 5: Generate outputs
    print("\n" + "=" * 70)
    print("STEP 5: Generating Outputs")
    print("=" * 70)

    export_results_json(benchmark)
    plot_convergence_curves(benchmark)
    plot_suitability_heatmap(suitability, benchmark)

    # Final summary
    print_summary(benchmark)

    print("\n[OK] All outputs saved to ./outputs/")
    print("  - benchmark_results.json")
    print("  - convergence_curves.png")
    print("  - suitability_heatmap.png")


if __name__ == "__main__":
    main()
