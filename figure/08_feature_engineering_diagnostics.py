"""Tutorial figures for feature-space and data-quality inspection.

The data are deterministic teaching data. The figures show why a distribution
section often includes a two-dimensional embedding, a correlation view, and a
missingness view before the main model result.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

from polarviz.style import paper_style


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figure/output"
OUTPUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2026)
n = 180
labels = np.repeat(["negative", "positive"], n // 2)

# Four correlated engineered features with a class-dependent shift.
base = rng.normal(size=(n, 4))
features = base @ np.array(
    [[1.0, 0.55, 0.10, 0.00],
     [0.00, 0.85, 0.40, 0.00],
     [0.10, 0.00, 0.90, 0.45],
     [0.00, 0.10, 0.00, 0.80]]
)
features[labels == "positive"] += np.array([1.1, 0.8, 0.35, 0.95])
feature_names = ("feature_1", "feature_2", "feature_3", "feature_4")

# PCA from a standardized feature matrix. The same interface can be replaced
# with sklearn.decomposition.PCA or another embedding method.
standardized = (features - features.mean(axis=0)) / features.std(axis=0, ddof=1)
_, _, components = np.linalg.svd(standardized, full_matrices=False)
embedding = standardized @ components[:2].T

def add_covariance_ellipse(axis, points, color):
    """Add a restrained 1-SD ellipse to show each class footprint."""
    center = points.mean(axis=0)
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    ellipse = Ellipse(
        center,
        width,
        height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.2,
        alpha=0.10,
        zorder=1,
    )
    axis.add_patch(ellipse)


# Figure 1: two-dimensional feature-space view.
colors = {"negative": "#2F6690", "positive": "#D1495B"}
with paper_style():
    fig, ax = plt.subplots(figsize=(6.8, 4.8), layout="constrained")
    for label in ("negative", "positive"):
        mask = labels == label
        points = embedding[mask]
        add_covariance_ellipse(ax, points, colors[label])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=25,
            alpha=0.68,
            color=colors[label],
            edgecolors="white",
            linewidths=0.45,
            zorder=2,
        )
        ax.scatter(
            points[:, 0].mean(),
            points[:, 1].mean(),
            s=55,
            marker="D",
            color=colors[label],
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )
    ax.axhline(0, color="#B7C1C8", linewidth=0.75, linestyle=(0, (3, 3)), zorder=0)
    ax.axvline(0, color="#B7C1C8", linewidth=0.75, linestyle=(0, (3, 3)), zorder=0)
    explained = np.linalg.svd(standardized, compute_uv=False) ** 2
    explained = explained / explained.sum()
    ax.set(
        xlabel=f"PC1 ({explained[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({explained[1] * 100:.1f}% variance)",
        title="PCA feature space",
    )
    ax.grid(True, color="#E7ECEF", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["negative"],
               markeredgecolor="white", markersize=6.5, label="Negative"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["positive"],
               markeredgecolor="white", markersize=6.5, label="Positive"),
        Line2D([0], [0], marker="D", color="#56616A", markerfacecolor="#56616A",
               markersize=5, label="Class centre"),
    ]
    ax.legend(handles=legend_handles, frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), handletextpad=0.35, columnspacing=1.1)
    fig.savefig(OUTPUT / "08_feature_engineering_pca.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# The following panels are exported separately. Each image answers one
# diagnostic question, so a paper section can include only the relevant panel.
mask = rng.random((30, len(feature_names))) < np.array([0.06, 0.13, 0.22, 0.09])
missing_rates = mask.mean(axis=0)
corr = np.corrcoef(features, rowvar=False)

with paper_style():
    # Panel a: signed correlation, with the values printed for quick reading.
    fig, ax = plt.subplots(figsize=(3.7, 3.25), layout="constrained")
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set(
        xticks=range(4),
        xticklabels=["F1", "F2", "F3", "F4"],
        yticks=range(4),
        yticklabels=["F1", "F2", "F3", "F4"],
    )
    ax.set_title("a  Feature correlation", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for row in range(4):
        for column in range(4):
            value = corr[row, column]
            ax.text(column, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(value) > 0.55 else "#273238")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Pearson r", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    fig.savefig(OUTPUT / "08_feature_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Panel b: make missing entries visually distinct from the white background.
    fig, ax = plt.subplots(figsize=(4.25, 3.05), layout="constrained")
    missing_cmap = ListedColormap(["#F0F3F5", "#D1495B"])
    ax.imshow(mask, aspect="auto", interpolation="none", cmap=missing_cmap,
              vmin=0, vmax=1)
    ax.set(
        xticks=range(4),
        xticklabels=["F1", "F2", "F3", "F4"],
        yticks=[0, 9, 19, 29],
        yticklabels=["S01", "S10", "S20", "S30"],
    )
    ax.set_title("b  Missingness pattern", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    ax.set_xlabel("feature")
    ax.set_ylabel("sample")
    ax.legend(
        handles=[Patch(facecolor="#F0F3F5", edgecolor="#D0D8DE", label="observed"),
                 Patch(facecolor="#D1495B", edgecolor="#D1495B", label="missing")],
        frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16),
        ncol=2, handlelength=1.0, columnspacing=0.8,
    )
    fig.savefig(OUTPUT / "08_feature_missingness_pattern.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # Panel c: sort by the observed missingness rate and annotate percentages.
    fig, ax = plt.subplots(figsize=(4.0, 2.75), layout="constrained")
    order = np.argsort(missing_rates)
    sorted_rates = missing_rates[order]
    sorted_names = np.array(["F1", "F2", "F3", "F4"])[order]
    ax.hlines(np.arange(4), 0, sorted_rates, color="#B7C1C8", linewidth=1.3)
    ax.scatter(sorted_rates, np.arange(4), color="#157F8C", s=46, zorder=2)
    for index, value in enumerate(sorted_rates):
        ax.text(value + 0.008, index, f"{value:.0%}", va="center", fontsize=7)
    ax.set(
        yticks=np.arange(4),
        yticklabels=sorted_names,
        xlabel="missing rate",
        xlim=(0, 0.30),
    )
    ax.set_title("c  Missingness by feature", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#E7ECEF", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(OUTPUT / "08_feature_missingness_rate.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

# Remove the old combined image so it cannot be mistaken for the intended
# subfigure exports.
(OUTPUT / "08_feature_diagnostics.png").unlink(missing_ok=True)

for filename in (
    "08_feature_engineering_pca.png",
    "08_feature_correlation.png",
    "08_feature_missingness_pattern.png",
    "08_feature_missingness_rate.png",
):
    print(OUTPUT / filename)
