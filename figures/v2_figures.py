"""
Publication figures for v2 paper.

fig1: UMAP of learned representations colored by mineral class
fig2: Bar chart: Set+SSL vs baselines at all label fractions
fig3: Low-label scaling curve (accuracy vs label fraction)
"""

from __future__ import annotations
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "figure.dpi": 300,
})

RESULTS_DIR = "experiments/v2/results"
FIGURES_DIR = "figures/v2"


def fig1_umap():
    """UMAP of SSL-learned representations."""
    try:
        from sklearn.manifold import TSNE  # use t-SNE as fallback (no umap dep)
    except ImportError:
        print("  Skipping UMAP: sklearn not available")
        return

    data = np.load(os.path.join(RESULTS_DIR, "embeddings.npz"), allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    names = data["names"]

    # t-SNE reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    coords = tsne.fit_transform(embeddings)

    # Get top 10 classes by frequency for coloring
    unique, counts = np.unique(labels, return_counts=True)
    top10 = unique[np.argsort(-counts)[:10]]
    unique_names = {l: names[np.where(labels == l)[0][0]] for l in top10}

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.tab10

    # Plot "other" in gray first
    other_mask = ~np.isin(labels, top10)
    ax.scatter(coords[other_mask, 0], coords[other_mask, 1],
               c="lightgray", s=15, alpha=0.3, label="Other")

    for i, cls in enumerate(top10):
        mask = labels == cls
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], s=25, alpha=0.7, label=unique_names[cls])

    ax.legend(fontsize=7, loc="upper right", ncol=2, markerscale=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("SSL-Learned Representations of Experimental XRD Patterns")
    ax.grid(True, alpha=0.2)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_tsne.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_tsne.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig1_tsne")


def fig2_bar_chart():
    """Main result bar chart at 100% labels."""
    try:
        data = json.load(open(os.path.join(RESULTS_DIR, "final_results.json")))
    except FileNotFoundError:
        data = json.load(open(os.path.join(RESULTS_DIR, "full_comparison.json")))
        # Convert single-run format
        for k, v in data.items():
            if "acc" in v and "acc_mean" not in v:
                data[k] = {"acc_mean": v["acc"], "acc_std": 0, "top5_mean": v["top5"], "top5_std": 0}

    methods = ["cnn_sup", "set_sup", "set_ssl_ft"]
    labels = ["CNN\nSupervised", "Set Transformer\nSupervised", "Set Transformer\n+ SSL"]
    colors = ["#90CAF9", "#2196F3", "#4CAF50"]

    means = [data.get(f"{m}_1.0", {}).get("acc_mean", data.get(f"{m}_1.0", {}).get("acc", 0)) for m in methods]
    stds = [data.get(f"{m}_1.0", {}).get("acc_std", 0) for m in methods]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(range(len(methods)), [m*100 for m in means],
                  yerr=[s*100 for s in stds], capsize=5,
                  color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Mineral Classification on RRUFF Experimental XRD")
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)

    for bar, m, s in zip(bars, means, stds):
        txt = f"{m:.1%}" if s == 0 else f"{m:.1%}\n+/-{s:.1%}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                txt, ha="center", va="bottom", fontsize=8)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_main_result.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_main_result.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig2_main_result")


def fig3_scaling():
    """Accuracy vs label fraction for all methods."""
    try:
        data = json.load(open(os.path.join(RESULTS_DIR, "final_results.json")))
    except FileNotFoundError:
        data = json.load(open(os.path.join(RESULTS_DIR, "full_comparison.json")))
        for k, v in data.items():
            if "acc" in v and "acc_mean" not in v:
                data[k] = {"acc_mean": v["acc"], "acc_std": 0}

    fracs = [0.1, 0.2, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for method, label, color, marker in [
        ("cnn_sup", "CNN Supervised", "#90CAF9", "o"),
        ("set_sup", "Set Supervised", "#2196F3", "s"),
        ("set_ssl_ft", "Set + SSL", "#4CAF50", "D"),
    ]:
        means = [data.get(f"{method}_{f}", {}).get("acc_mean", data.get(f"{method}_{f}", {}).get("acc", 0)) for f in fracs]
        stds = [data.get(f"{method}_{f}", {}).get("acc_std", 0) for f in fracs]
        ax.errorbar(fracs, [m*100 for m in means], yerr=[s*100 for s in stds],
                    marker=marker, color=color, label=label, capsize=4,
                    linewidth=1.5, markersize=7)

    ax.set_xlabel("Labeled Fraction")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("SSL Advantage Grows at Low Label Fractions")
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 85)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_scaling.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_scaling.png"), bbox_inches="tight")
    plt.close()
    print("  Saved fig3_scaling")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("Generating v2 publication figures...")
    fig1_umap()
    fig2_bar_chart()
    fig3_scaling()
    print("Done.")


if __name__ == "__main__":
    main()
