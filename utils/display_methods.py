from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_benchmark_comparison(df_results):
    """Plot the long-run versus short-run simulated annealing benchmark."""

    labels = [f"Inst {index}" for index in df_results["Instance"]]
    long_costs = df_results["Long_Run_Cost"]
    short_costs = df_results["Best_of_Short_Runs"]
    errors = df_results["Short_Runs_Std"]

    positions = np.arange(len(labels))
    width = 0.35

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.bar(
        positions - width / 2,
        long_costs,
        width,
        label="Strategy A: One Long Run",
        color="#2c3e50",
    )
    axis.bar(
        positions + width / 2,
        short_costs,
        width,
        yerr=errors,
        label="Strategy B: Best of 10 Short Runs",
        capsize=5,
        color="#e74c3c",
    )

    axis.set_ylabel("Energy (lower is better)")
    axis.set_title("Resource Efficiency Benchmark: Long Run vs Short Runs")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.legend()
    axis.grid(axis="y", linestyle="--", alpha=0.6)
    figure.tight_layout()
    return axis
