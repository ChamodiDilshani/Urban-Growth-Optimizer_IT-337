"""
Sensitivity Analysis for AHP Criteria Weights.

Systematically varies each criterion weight and observes how
the optimal site location and fitness score change.

This answers: "Which criteria has the most influence on site selection?"

Generates:
    - Sensitivity line plots for each criterion
    - Tornado chart showing impact ranking
    - Results table (JSON + console)

Usage:
    python sensitivity_analysis.py
"""

import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_data_loader import RealDataLoader
from ahp_engine import AHPEngine
from suitability_surface import SuitabilitySurface
from pso_optimizer import SpatialPSO


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
GRID_SIZE = 200
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

CRITERIA_NAMES = [
    "Road Proximity",
    "Slope Safety",
    "Population Density",
    "School Distance",
    "Healthcare Proximity",
    "Land Use Suitability",
    "Transport Access",
    "Amenity Buffer",
]

# Map criteria name → data layer key
CRITERIA_LAYER_MAP = {
    "Road Proximity": "road_proximity",
    "Slope Safety": "slope",
    "Population Density": "population_growth",
    "School Distance": "school_distance",
    "Healthcare Proximity": "healthcare_proximity",
    "Land Use Suitability": "landuse_suitability",
    "Transport Access": "transport_access",
    "Amenity Buffer": None,  # derived: population_growth * 0.8
}

# Base AHP weights (from 8x8 eigenvector method)
BASE_WEIGHTS = np.array([0.3436, 0.1553, 0.0653, 0.0300, 0.1553, 0.0653, 0.1553, 0.0300])

# Weight variation range: test each criterion from 0.1 to 0.8
WEIGHT_STEPS = np.arange(0.10, 0.85, 0.05)


def load_data():
    """Load real geospatial data (once)."""
    loader = RealDataLoader(data_dir=DATA_DIR, grid_size=GRID_SIZE, random_seed=42)
    layers = loader.generate_all()
    slope_raw = loader.calculate_slope(layers["elevation"])
    return layers, slope_raw


def run_with_weights(weights, layers, slope_raw):
    """Run suitability + PSO optimization with given weights."""
    # Build suitability surface
    criteria_rasters = {}
    for cname in CRITERIA_NAMES:
        layer_key = CRITERIA_LAYER_MAP.get(cname)
        if layer_key is None:
            criteria_rasters[cname] = layers["population_growth"] * 0.8
        else:
            criteria_rasters[cname] = layers[layer_key]
    surface = SuitabilitySurface(
        criteria_rasters=criteria_rasters,
        weights=weights,
        constraint_mask=layers["constraint_mask"],
        elevation=layers["elevation"],
    )
    suitability = surface.build_surface()

    # Run PSO (winner from statistical benchmark)
    pso = SpatialPSO(
        suitability_matrix=suitability,
        elevation=layers["elevation"],
        slope=slope_raw,
        constraint_mask=layers["constraint_mask"],
        swarm_size=50,
        iterations=100,
        random_seed=42,
    )
    result = pso.optimize()

    return {
        "fitness": float(result["best_fitness"]),
        "x": float(result["best_x"]),
        "y": float(result["best_y"]),
        "z": float(result["best_z"]),
        "suitability_mean": float(suitability.mean()),
        "suitability_max": float(suitability.max()),
    }


def vary_single_criterion(criterion_idx, layers, slope_raw):
    """Vary one criterion weight from 0.1 to 0.8, redistribute others proportionally."""
    results = []
    criterion_name = CRITERIA_NAMES[criterion_idx]

    for target_weight in WEIGHT_STEPS:
        # Create modified weights: set criterion_idx to target_weight,
        # distribute remaining (1 - target_weight) proportionally among others
        remaining = 1.0 - target_weight
        other_indices = [i for i in range(len(BASE_WEIGHTS)) if i != criterion_idx]
        other_sum = sum(BASE_WEIGHTS[i] for i in other_indices)

        weights = np.zeros(len(BASE_WEIGHTS))
        weights[criterion_idx] = target_weight
        for i in other_indices:
            weights[i] = remaining * (BASE_WEIGHTS[i] / other_sum)

        result = run_with_weights(weights, layers, slope_raw)
        result["target_weight"] = float(target_weight)
        result["weights"] = weights.tolist()
        results.append(result)

    return results


def compute_sensitivity_index(results):
    """Compute sensitivity index: how much fitness changes per unit weight change."""
    weights = [r["target_weight"] for r in results]
    fitnesses = [r["fitness"] for r in results]

    # Filter out penalty values (-999)
    valid = [(w, f) for w, f in zip(weights, fitnesses) if f > 0]
    if len(valid) < 2:
        return 0.0

    valid_w, valid_f = zip(*valid)
    # Linear regression slope = sensitivity
    coeffs = np.polyfit(valid_w, valid_f, 1)
    return float(coeffs[0])


def compute_location_shift(results):
    """Compute total location shift (Euclidean distance) across weight range."""
    valid = [r for r in results if r["fitness"] > 0]
    if len(valid) < 2:
        return 0.0

    first = valid[0]
    last = valid[-1]
    dx = last["x"] - first["x"]
    dy = last["y"] - first["y"]
    return float(np.sqrt(dx**2 + dy**2))


def plot_sensitivity_lines(all_results, output_path):
    """Plot fitness vs weight for each criterion."""
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    for idx, (criterion_name, results) in enumerate(all_results.items()):
        ax = axes[idx // 4][idx % 4]
        weights = [r["target_weight"] for r in results]
        fitnesses = [max(r["fitness"], 0) for r in results]  # Clip -999 to 0

        ax.plot(weights, fitnesses, "o-", color=colors[idx], linewidth=2.5,
                markersize=7, markeredgecolor="white", markeredgewidth=1.5)

        # Mark base weight with vertical line
        base_w = BASE_WEIGHTS[idx]
        ax.axvline(x=base_w, color="gray", linestyle="--", alpha=0.7, label=f"Base: {base_w:.3f}")

        ax.set_xlabel("Weight Value", fontsize=11)
        ax.set_ylabel("Best Fitness Score", fontsize=11)
        ax.set_title(f"{criterion_name}", fontsize=13, fontweight="bold", color=colors[idx])
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(0.05, 0.85)
        ax.legend(fontsize=10)

    plt.suptitle(
        "Sensitivity Analysis: Fitness vs Criteria Weight\n"
        "Matale District, Sri Lanka (30 runs per criterion)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tornado(sensitivity_indices, output_path):
    """Plot tornado chart showing which criterion is most influential."""
    names = list(sensitivity_indices.keys())
    values = list(sensitivity_indices.values())

    # Sort by absolute sensitivity
    sorted_pairs = sorted(zip(names, values), key=lambda x: abs(x[1]))
    names = [p[0] for p in sorted_pairs]
    values = [p[1] for p in sorted_pairs]

    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, values, color=colors, height=0.6, edgecolor="white", linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        x_pos = bar.get_width()
        ax.text(x_pos + 0.005 * np.sign(x_pos), bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Sensitivity Index (Fitness change per unit weight change)", fontsize=11)
    ax.set_title(
        "Tornado Chart: Criteria Sensitivity Ranking\n"
        "Matale District, Sri Lanka",
        fontsize=14, fontweight="bold",
    )
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_location_shifts(all_results, output_path):
    """Plot how optimal location shifts as weights change."""
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
    fig, ax = plt.subplots(figsize=(10, 8))

    for idx, (criterion_name, results) in enumerate(all_results.items()):
        valid = [r for r in results if r["fitness"] > 0]
        if len(valid) < 2:
            continue

        xs = [r["y"] for r in valid]  # y = column = "X on map"
        ys = [r["x"] for r in valid]  # x = row = "Y on map"
        weights = [r["target_weight"] for r in valid]

        # Plot path with arrows
        ax.plot(xs, ys, "o-", color=colors[idx], linewidth=2, markersize=6,
                label=criterion_name, alpha=0.8)

        # Mark start and end
        ax.annotate(f"w={weights[0]:.2f}", (xs[0], ys[0]), fontsize=8,
                    color=colors[idx], fontweight="bold")
        ax.annotate(f"w={weights[-1]:.2f}", (xs[-1], ys[-1]), fontsize=8,
                    color=colors[idx], fontweight="bold")

    ax.set_xlabel("Grid Y (Column)", fontsize=12)
    ax.set_ylabel("Grid X (Row)", fontsize=12)
    ax.set_title(
        "Optimal Site Location Shift Under Weight Variation\n"
        "Matale District, Sri Lanka",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary_table(all_results, sensitivity_indices, location_shifts):
    """Print formatted sensitivity analysis summary."""
    print("\n" + "=" * 80)
    print("  SENSITIVITY ANALYSIS RESULTS")
    print("=" * 80)
    print(f"{'Criterion':<25} {'Base Weight':>12} {'Sensitivity':>14} {'Location Shift':>16} {'Rank':>6}")
    print("-" * 80)

    # Rank by absolute sensitivity
    ranked = sorted(sensitivity_indices.items(), key=lambda x: abs(x[1]), reverse=True)

    for rank, (name, sens) in enumerate(ranked, 1):
        base_w = BASE_WEIGHTS[CRITERIA_NAMES.index(name)]
        shift = location_shifts[name]
        print(f"  {name:<23} {base_w:>12.4f} {sens:>+14.4f} {shift:>14.1f} px {rank:>6}")

    print("=" * 80)
    print(f"\n  Most Influential Criterion: {ranked[0][0]}")
    print(f"  Sensitivity Index: {ranked[0][1]:+.4f}")
    print(f"  Interpretation: Changing this weight by 0.1 shifts fitness by {abs(ranked[0][1]) * 0.1:.4f}")
    print()


def main():
    """Run the full sensitivity analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data
    layers, slope_raw = load_data()

    # Step 2: Run sensitivity analysis for each criterion
    all_results = {}
    sensitivity_indices = {}
    location_shifts = {}

    total = len(CRITERIA_NAMES) * len(WEIGHT_STEPS)
    done = 0

    for idx, name in enumerate(CRITERIA_NAMES):
        print(f"\n  Analyzing: {name} (criterion {idx+1}/{len(CRITERIA_NAMES)})")
        results = []

        for target_weight in WEIGHT_STEPS:
            done += 1
            sys.stdout.write(f"\r    Weight={target_weight:.2f} [{done}/{total}]")
            sys.stdout.flush()

            remaining = 1.0 - target_weight
            other_indices = [i for i in range(len(BASE_WEIGHTS)) if i != idx]
            other_sum = sum(BASE_WEIGHTS[i] for i in other_indices)

            weights = np.zeros(len(BASE_WEIGHTS))
            weights[idx] = target_weight
            for i in other_indices:
                weights[i] = remaining * (BASE_WEIGHTS[i] / other_sum)

            result = run_with_weights(weights, layers, slope_raw)
            result["target_weight"] = float(target_weight)
            result["weights"] = weights.tolist()
            results.append(result)

        all_results[name] = results
        sensitivity_indices[name] = compute_sensitivity_index(results)
        location_shifts[name] = compute_location_shift(results)

    print(f"\r    All {total} configurations tested!                    ")

    # Step 3: Print summary
    print_summary_table(all_results, sensitivity_indices, location_shifts)

    # Step 4: Generate plots
    line_path = os.path.join(OUTPUT_DIR, "sensitivity_lines.png")
    plot_sensitivity_lines(all_results, line_path)
    print(f"  Sensitivity line plots saved: {line_path}")

    tornado_path = os.path.join(OUTPUT_DIR, "sensitivity_tornado.png")
    plot_tornado(sensitivity_indices, tornado_path)
    print(f"  Tornado chart saved: {tornado_path}")

    shift_path = os.path.join(OUTPUT_DIR, "sensitivity_location_shift.png")
    plot_location_shifts(all_results, shift_path)
    print(f"  Location shift plot saved: {shift_path}")

    # Step 5: Save JSON
    json_result = {
        "criteria": CRITERIA_NAMES,
        "base_weights": BASE_WEIGHTS.tolist(),
        "weight_steps": WEIGHT_STEPS.tolist(),
        "sensitivity_indices": sensitivity_indices,
        "location_shifts": location_shifts,
        "detailed_results": {
            name: [
                {k: v for k, v in r.items()}
                for r in results
            ]
            for name, results in all_results.items()
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "sensitivity_analysis.json")
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2)
    print(f"  JSON results saved: {json_path}")

    print(f"\n[OK] Sensitivity analysis complete!")


if __name__ == "__main__":
    main()
