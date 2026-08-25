"""Render the captured Colab run as a compact portfolio figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    results_dir = Path("results")
    run = pd.read_csv(results_dir / "run_summary.csv").iloc[0]
    classes = pd.read_csv(results_dir / "class_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    bars = axes[0].bar(
        ["3-fold CV", "Held-out test"],
        [run["mean_cv_accuracy"], run["test_accuracy"]],
        color=["#7c3aed", "#2563eb"],
    )
    axes[0].bar_label(bars, fmt="%.4f", padding=4, weight="bold")
    axes[0].set_ylim(0.75, 0.92)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Selected configuration generalizes to test", weight="bold")

    ordered = classes.sort_values("f1_score")
    bars = axes[1].barh(ordered["class"], ordered["f1_score"], color="#0f766e")
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].set_xlim(0.6, 1.02)
    axes[1].set_xlabel("F1 score")
    axes[1].set_title("Per-class test performance", weight="bold")

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Fashion-MNIST dense network — 12 candidates × 3 folds",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(results_dir / "experiment_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

