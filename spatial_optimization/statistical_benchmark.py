"""
Statistical Benchmark: 30-Run Comparison of GA vs PSO.

Runs both optimizers 30 times with different random seeds,
collects performance metrics, and performs statistical significance
testing (Wilcoxon rank-sum test).

Generates:
    - Statistical summary table
    - Box plots
    - Convergence comparison (average across runs)
    - JSON results file

Usage:
    python statistical_benchmark.py
"""

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from real_data_loader import RealDataLoader
from ahp_engine import AHPEngine
from suitability_surface import SuitabilitySurface
from genetic_algorithm import SpatialGeneticAlgorithm
from pso_optimizer import SpatialPSO


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
GRID_SIZE = 200
NUM_RUNS = 30
DATA_DIR = "data"
OUTPUT_DIR = "outputs"

CRITERIA_NAMES = [
    "Road Proximity",
    "Slope Safety",
    "Future Growth Velocity",
    "Existing Amenity Buffer",
]

AHP_MATRIX = np.array([
    [1,     3,     5,     7],
    [1/3,   1,     3,     5],
    [1/5,   1/3,   1,     3],
    [1/7,   1/5,   1/3,   1],
])


def load_data_and_surface():
    """Load data, compute AHP weights, and build suitability surface."""
    print("Loading data and building suitability surface...")

    # Load real data
    loader = RealDataLoader(data_dir=DATA_DIR, grid_size=GRID_SIZE, random_seed=42)
    layers = loader.generate_all()
    slope_raw = loader.calculate_slope(layers["elevation"])

    # AHP
    engine = AHPEngine(criteria_names=CRITERIA_NAMES, comparison_matrix=AHP_MATRIX)
    weights, metrics = engine.run()

    # Suitability surface
    criteria_rasters = {
        CRITERIA_NAMES[0]: layers["road_proximity"],
        CRITERIA_NAMES[1]: layers["slope"],
        CRITERIA_NAMES[2]: layers["population_growth"],
        CRITERIA_NAMES[3]: layers["population_growth"] * 0.8,
    }
    surface = SuitabilitySurface(
        criteria_rasters=criteria_rasters,
        weights=weights,
        constraint_mask=layers["constraint_mask"],
        elevation=layers["elevation"],
    )
    suitability = surface.build_surface()

    return suitability, layers, slope_raw, weights, metrics


def run_single_ga(suitability, layers, slope_raw, seed):
    """Run a single GA optimization with given seed."""
    t0 = time.time()
    ga = SpatialGeneticAlgorithm(
        suitability_matrix=suitability,
        elevation=layers["elevation"],
        slope=slope_raw,
        constraint_mask=layers["constraint_mask"],
        population_size=50,
        generations=100,
        random_seed=seed,
    )
    result = ga.optimize()
    elapsed = time.time() - t0
    return {
        "fitness": result["best_fitness"],
        "x": result["best_x"],
        "y": result["best_y"],
        "z": result["best_z"],
        "time": elapsed,
        "history": result.get("fitness_history", []),
    }


def run_single_pso(suitability, layers, slope_raw, seed):
    """Run a single PSO optimization with given seed."""
    t0 = time.time()
    pso = SpatialPSO(
        suitability_matrix=suitability,
        elevation=layers["elevation"],
        slope=slope_raw,
        constraint_mask=layers["constraint_mask"],
        swarm_size=50,
        iterations=100,
        random_seed=seed,
    )
    result = pso.optimize()
    elapsed = time.time() - t0
    return {
        "fitness": result["best_fitness"],
        "x": result["best_x"],
        "y": result["best_y"],
        "z": result["best_z"],
        "time": elapsed,
        "history": result.get("convergence_history", []),
    }


def compute_statistics(results: List[dict]) -> dict:
    """Compute descriptive statistics from a list of run results."""
    fitnesses = [r["fitness"] for r in results]
    times = [r["time"] for r in results]

    return {
        "mean_fitness": float(np.mean(fitnesses)),
        "std_fitness": float(np.std(fitnesses)),
        "best_fitness": float(np.max(fitnesses)),
        "worst_fitness": float(np.min(fitnesses)),
        "median_fitness": float(np.median(fitnesses)),
        "mean_time": float(np.mean(times)),
        "std_time": float(np.std(times)),
        "all_fitnesses": [float(f) for f in fitnesses],
        "all_times": [float(t) for t in times],
    }


def wilcoxon_test(ga_fitnesses, pso_fitnesses):
    """Perform Wilcoxon rank-sum test for statistical significance."""
    stat, p_value = stats.ranksums(ga_fitnesses, pso_fitnesses)
    return {
        "test": "Wilcoxon Rank-Sum",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def plot_box_comparison(ga_stats, pso_stats, output_path):
    """Generate box plot comparing GA and PSO fitness distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fitness box plot
    ax1 = axes[0]
    bp = ax1.boxplot(
        [ga_stats["all_fitnesses"], pso_stats["all_fitnesses"]],
        tick_labels=["GA", "PSO"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#e74c3c")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("#3498db")
    bp["boxes"][1].set_alpha(0.7)
    ax1.set_ylabel("Best Fitness Score", fontsize=12)
    ax1.set_title("Fitness Distribution (30 Runs)", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Add mean markers
    ax1.plot(1, ga_stats["mean_fitness"], "D", color="darkred", markersize=8, label=f'GA Mean: {ga_stats["mean_fitness"]:.4f}')
    ax1.plot(2, pso_stats["mean_fitness"], "D", color="darkblue", markersize=8, label=f'PSO Mean: {pso_stats["mean_fitness"]:.4f}')
    ax1.legend(fontsize=10)

    # Time box plot
    ax2 = axes[1]
    bp2 = ax2.boxplot(
        [ga_stats["all_times"], pso_stats["all_times"]],
        tick_labels=["GA", "PSO"],
        patch_artist=True,
        widths=0.5,
    )
    bp2["boxes"][0].set_facecolor("#e74c3c")
    bp2["boxes"][0].set_alpha(0.7)
    bp2["boxes"][1].set_facecolor("#3498db")
    bp2["boxes"][1].set_alpha(0.7)
    ax2.set_ylabel("Execution Time (seconds)", fontsize=12)
    ax2.set_title("Execution Time Distribution (30 Runs)", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax2.plot(1, ga_stats["mean_time"], "D", color="darkred", markersize=8, label=f'GA Mean: {ga_stats["mean_time"]:.3f}s')
    ax2.plot(2, pso_stats["mean_time"], "D", color="darkblue", markersize=8, label=f'PSO Mean: {pso_stats["mean_time"]:.3f}s')
    ax2.legend(fontsize=10)

    plt.suptitle(
        "Statistical Comparison: GA vs PSO (30 Independent Runs)\n"
        "Matale District, Sri Lanka",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_average_convergence(ga_results, pso_results, output_path):
    """Plot average convergence curves across all runs."""
    # Pad histories to same length and compute mean
    def pad_and_average(results, key="history"):
        histories = [r[key] for r in results if len(r[key]) > 0]
        if not histories:
            return []
        max_len = max(len(h) for h in histories)
        padded = []
        for h in histories:
            arr = np.array(h)
            arr = np.clip(arr, 0, 1.05)
            if len(arr) < max_len:
                arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
            padded.append(arr)
        return np.mean(padded, axis=0), np.std(padded, axis=0)

    ga_mean, ga_std = pad_and_average(ga_results)
    pso_mean, pso_std = pad_and_average(pso_results)

    fig, ax = plt.subplots(figsize=(10, 6))
    x_ga = range(1, len(ga_mean) + 1)
    x_pso = range(1, len(pso_mean) + 1)

    # Mean line + std shading
    ax.plot(x_ga, ga_mean, color="#e74c3c", linewidth=2, label="GA (Mean)")
    ax.fill_between(x_ga, ga_mean - ga_std, ga_mean + ga_std, color="#e74c3c", alpha=0.15, label="GA (Std)")

    ax.plot(x_pso, pso_mean, color="#3498db", linewidth=2, label="PSO (Mean)")
    ax.fill_between(x_pso, pso_mean - pso_std, pso_mean + pso_std, color="#3498db", alpha=0.15, label="PSO (Std)")

    ax.set_xlabel("Iteration / Generation", fontsize=12)
    ax.set_ylabel("Best Fitness Score", fontsize=12)
    ax.set_title(
        "Average Convergence Curves (30 Runs)\n"
        "Matale District, Sri Lanka",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_comparison_table(ga_stats, pso_stats, test_result):
    """Print formatted comparison table."""
    print("\n" + "=" * 72)
    print("  STATISTICAL BENCHMARK RESULTS (30 Independent Runs)")
    print("=" * 72)
    print(f"{'Metric':<30} {'GA':>18} {'PSO':>18}")
    print("-" * 72)
    print(f"{'Mean Fitness':<30} {ga_stats['mean_fitness']:>18.6f} {pso_stats['mean_fitness']:>18.6f}")
    print(f"{'Std Fitness':<30} {ga_stats['std_fitness']:>18.6f} {pso_stats['std_fitness']:>18.6f}")
    print(f"{'Best Fitness':<30} {ga_stats['best_fitness']:>18.6f} {pso_stats['best_fitness']:>18.6f}")
    print(f"{'Worst Fitness':<30} {ga_stats['worst_fitness']:>18.6f} {pso_stats['worst_fitness']:>18.6f}")
    print(f"{'Median Fitness':<30} {ga_stats['median_fitness']:>18.6f} {pso_stats['median_fitness']:>18.6f}")
    print(f"{'Mean Time (s)':<30} {ga_stats['mean_time']:>18.4f} {pso_stats['mean_time']:>18.4f}")
    print(f"{'Std Time (s)':<30} {ga_stats['std_time']:>18.4f} {pso_stats['std_time']:>18.4f}")
    print("-" * 72)
    print(f"{'Wilcoxon Statistic':<30} {test_result['statistic']:>38.4f}")
    print(f"{'p-value':<30} {test_result['p_value']:>38.6f}")
    print(f"{'Significant (p<0.05)?':<30} {'YES' if test_result['significant_005'] else 'NO':>38}")
    print(f"{'Significant (p<0.01)?':<30} {'YES' if test_result['significant_001'] else 'NO':>38}")
    print("=" * 72)

    if pso_stats["mean_fitness"] > ga_stats["mean_fitness"]:
        winner = "PSO"
    elif ga_stats["mean_fitness"] > pso_stats["mean_fitness"]:
        winner = "GA"
    else:
        winner = "Tie"

    sig = "statistically significant" if test_result["significant_005"] else "NOT statistically significant"
    print(f"\n  CONCLUSION: {winner} outperforms on average.")
    print(f"  The difference is {sig} (p={test_result['p_value']:.6f}).\n")


def main():
    """Run the full 30-run statistical benchmark."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load data (once — shared across all runs)
    suitability, layers, slope_raw, weights, ahp_metrics = load_data_and_surface()

    print(f"\nRunning {NUM_RUNS} independent optimization trials...")
    print("-" * 60)

    ga_results = []
    pso_results = []

    for i in range(NUM_RUNS):
        seed = i + 1  # Seeds: 1, 2, 3, ..., 30
        sys.stdout.write(f"\r  Run {i+1:2d}/{NUM_RUNS} (seed={seed})...")
        sys.stdout.flush()

        ga_r = run_single_ga(suitability, layers, slope_raw, seed)
        pso_r = run_single_pso(suitability, layers, slope_raw, seed)

        ga_results.append(ga_r)
        pso_results.append(pso_r)

    print(f"\r  All {NUM_RUNS} runs completed!                    ")

    # Step 2: Compute statistics
    ga_stats = compute_statistics(ga_results)
    pso_stats = compute_statistics(pso_results)

    # Step 3: Statistical test
    test_result = wilcoxon_test(ga_stats["all_fitnesses"], pso_stats["all_fitnesses"])

    # Step 4: Print results
    print_comparison_table(ga_stats, pso_stats, test_result)

    # Step 5: Generate plots
    box_path = os.path.join(OUTPUT_DIR, "statistical_boxplot.png")
    plot_box_comparison(ga_stats, pso_stats, box_path)
    print(f"  Box plot saved: {box_path}")

    conv_path = os.path.join(OUTPUT_DIR, "average_convergence.png")
    plot_average_convergence(ga_results, pso_results, conv_path)
    print(f"  Convergence plot saved: {conv_path}")

    # Step 6: Save JSON results
    json_result = {
        "num_runs": NUM_RUNS,
        "ga_statistics": {k: v for k, v in ga_stats.items() if k != "all_fitnesses" and k != "all_times"},
        "pso_statistics": {k: v for k, v in pso_stats.items() if k != "all_fitnesses" and k != "all_times"},
        "ga_all_fitnesses": ga_stats["all_fitnesses"],
        "pso_all_fitnesses": pso_stats["all_fitnesses"],
        "statistical_test": test_result,
        "ahp_consistency_ratio": float(ahp_metrics["cr"]),
    }
    json_path = os.path.join(OUTPUT_DIR, "statistical_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2)
    print(f"  JSON results saved: {json_path}")

    print(f"\n[OK] Statistical benchmark complete!")


if __name__ == "__main__":
    main()
