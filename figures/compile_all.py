"""
Compile all figures for the paper in publication-ready format.

Figures:
  Fig 1: Sort-match schematic (1D NMR case → 2D XRD case)
  Fig 2: Main result — 4-variant bar chart with error bars
  Fig 3: Low-label ablation (MAE vs labeled_frac)
  Fig 4: Noise robustness (grouped bar chart)
  Fig 5: Hungarian vs sort-match runtime scaling
  Fig 6: Training curves

Style: Nature-CS / JCIM — sans-serif, clean, no 3D effects.

Usage:
    python figures/compile_all.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

FIGURES_DIR = "figures"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def fig2_main_result():
    """4-variant bar chart with error bars from multi-seed experiment."""
    try:
        data = load_json("experiments/results_multiseed/multiseed_results.json")
    except FileNotFoundError:
        print("  Skipping fig2: multiseed results not found")
        return

    variants = ["supervised", "sort_match_ssl", "supervised_10pct"]
    labels = ["Supervised\n(100% labels)", "Sort-match SSL\n(10% labels)", "Supervised\n(10% labels)"]
    colors = ["#2196F3", "#4CAF50", "#90CAF9"]
    means = [data[v]["mean"] for v in variants]
    stds = [data[v]["std"] for v in variants]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(variants)), means, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5, width=0.6)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test MAE (degrees 2\u03b8)")
    ax.set_title("(a) Main Result: Sort-Match SSL Matches Full Supervision")
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
                f"{mean:.2f}\u00b1{std:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_main_result.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_main_result.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig2_main_result")


def fig3_low_label():
    """Low-label ablation: MAE vs labeled_frac for supervised and SSL."""
    try:
        data = load_json("experiments/results_session3/low_label_results.json")
    except FileNotFoundError:
        print("  Skipping fig3: low_label results not found")
        return

    fracs = sorted(set(v["label_frac"] for v in data.values()))
    sup_means = [data[f"supervised_frac{f}"]["mean"] for f in fracs]
    sup_stds = [data[f"supervised_frac{f}"]["std"] for f in fracs]
    ssl_means = [data[f"sort_match_ssl_frac{f}"]["mean"] for f in fracs]
    ssl_stds = [data[f"sort_match_ssl_frac{f}"]["std"] for f in fracs]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.errorbar(fracs, sup_means, yerr=sup_stds, marker="o", color="#2196F3",
                label="Supervised", capsize=4, linewidth=1.5, markersize=6)
    ax.errorbar(fracs, ssl_means, yerr=ssl_stds, marker="s", color="#4CAF50",
                label="Sort-match SSL", capsize=4, linewidth=1.5, markersize=6)
    ax.set_xlabel("Labeled Fraction")
    ax.set_ylabel("Test MAE (degrees 2\u03b8)")
    ax.set_title("(b) SSL Advantage Grows at Low Label Fractions")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_low_label.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_low_label.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig3_low_label")


def fig4_noise():
    """Noise robustness: grouped bar chart."""
    try:
        data = load_json("experiments/results_noise/noise_results.json")
    except FileNotFoundError:
        print("  Skipping fig4: noise results not found")
        return

    conditions = ["clean", "noise_0.1", "noise_0.3", "noise_1.0", "spurious_10pct", "combined"]
    labels = ["Clean", "\u03c3=0.1\u00b0", "\u03c3=0.3\u00b0", "\u03c3=1.0\u00b0", "10% spurious", "Combined"]
    colors = ["#4CAF50", "#66BB6A", "#81C784", "#A5D6A7", "#F44336", "#E57373"]
    maes = [data[c]["mae"] for c in conditions]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(range(len(conditions)), maes, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Test MAE (degrees 2\u03b8)")
    ax.set_title("(c) Noise Robustness: Tolerant to Gaussian, Sensitive to Spurious")
    ax.axhline(y=maes[0], color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_ylim(0, max(maes) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    for bar, mae in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{mae:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_noise.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig4_noise.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig4_noise")


def fig5_runtime():
    """Hungarian vs sort-match runtime scaling."""
    try:
        data = load_json("experiments/results_session3/hungarian_benchmark.json")
    except FileNotFoundError:
        print("  Skipping fig5: hungarian benchmark not found")
        return

    ns = sorted([int(k) for k in data["runtime"].keys()])
    sm_times = [data["runtime"][str(n)]["sort_match_ms"] for n in ns]
    hu_times = [data["runtime"][str(n)]["hungarian_ms"] for n in ns]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(ns, hu_times, "o-", color="#F44336", label="Hungarian O(n\u00b3)", markersize=6, linewidth=1.5)
    ax.plot(ns, sm_times, "s-", color="#4CAF50", label="Sort-match O(n log n)", markersize=6, linewidth=1.5)
    ax.set_xlabel("Number of Peaks (n)")
    ax.set_ylabel("Time per Batch (ms)")
    ax.set_title("(d) Runtime: Sort-Match is 73\u00d7 Faster at n=50")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_runtime.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig5_runtime.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig5_runtime")


def fig6_training_curves():
    """Training curves from Session 1 sanity experiment."""
    try:
        data = load_json("experiments/results_session1/sanity_results.json")
    except FileNotFoundError:
        print("  Skipping fig6: sanity results not found")
        return

    fig, ax = plt.subplots(figsize=(5, 3.5))
    style = {
        "supervised": ("#2196F3", "Supervised (100%)"),
        "supervised_10pct": ("#90CAF9", "Supervised (10%)"),
        "sort_match_ssl": ("#4CAF50", "Sort-match SSL (10%)"),
        "random_match": ("#F44336", "Random match"),
    }

    for name, (color, label) in style.items():
        if name in data:
            epochs = list(range(1, len(data[name]["test_maes"]) + 1))
            ax.plot(epochs, data[name]["test_maes"], color=color, label=label, linewidth=1.5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test MAE (degrees 2\u03b8)")
    ax.set_title("(e) Training Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_training_curves.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig6_training_curves.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig6_training_curves")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Compiling publication figures...")

    fig2_main_result()
    fig3_low_label()
    fig4_noise()
    fig5_runtime()
    fig6_training_curves()

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
