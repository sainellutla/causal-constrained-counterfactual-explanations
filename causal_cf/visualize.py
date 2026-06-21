"""
Visualization module — three plots:
  1. Boxplots: proximity vs. causal validity residual per method
  2. DAG violation heatmap (one subplot per method)
  3. t-SNE arrow map (original → CF), colored by causal validity score
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
from sklearn.manifold import TSNE
from typing import Dict, List

from causal_graph import ENDOGENOUS_NODES, IMMUTABLE_FEATURES

METHOD_COLORS = {
    "DiCE (genetic)": "#E15759",
    "Gradient (NN)": "#F28E2B",
    "Causal Gradient": "#4E79A7",
    "Causal GA": "#59A14F",
}

METHOD_ORDER = ["DiCE (genetic)", "Gradient (NN)", "Causal Gradient", "Causal GA"]


def _method_palette(results_df: pd.DataFrame):
    methods = results_df["method"].unique()
    return {m: METHOD_COLORS.get(m, "#999999") for m in methods}


# ---------------------------------------------------------------------------
# Plot 1: Proximity vs. Causal Validity Residual (side-by-side boxplots)
# ---------------------------------------------------------------------------

def plot_proximity_vs_causal_validity(results_df: pd.DataFrame, fig_dir: str):
    palette = _method_palette(results_df)
    methods = [m for m in METHOD_ORDER if m in results_df["method"].values]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("CF Method Comparison: Proximity vs. Causal Validity", fontsize=14, y=1.01)

    metrics = [
        ("proximity", "Proximity (L2)", "Lower is better → CFs are closer to originals"),
        ("causal_validity_residual", "Causal Validity Residual", "Lower is better → CFs are more causally consistent"),
    ]

    for ax, (metric, ylabel, subtitle) in zip(axes, metrics):
        data_per_method = [
            results_df[results_df["method"] == m][metric].values
            for m in methods
        ]
        colors = [palette[m] for m in methods]
        bp = ax.boxplot(data_per_method, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([m.replace(" ", "\n") for m in methods], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle, fontsize=9, color="gray")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(fig_dir, "boxplot_proximity_causal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: DAG Violation Heatmap
# ---------------------------------------------------------------------------

def _build_dag_layout(G: nx.DiGraph) -> dict:
    """Fixed layout for the causal DAG (prettier than spring_layout)."""
    pos = {
        "age":            (0, 2),
        "education":      (1, 3),
        "marital_status": (1, 1),
        "occupation":     (2, 3),
        "hours_per_week": (2, 2),
        "capital_gain":   (3, 3),
        "sex":            (0, 0),
        "race":           (-1, 1),
        "income":         (4, 2),
    }
    # Only keep positions for nodes that exist in G
    return {n: pos[n] for n in G.nodes() if n in pos}


def plot_dag_violation_heatmap(results_df: pd.DataFrame, G: nx.DiGraph, fig_dir: str):
    """
    2×2 grid of DAG subplots, one per method.
    Edges are colored by the average causal residual of their child node.
    """
    methods = [m for m in METHOD_ORDER if m in results_df["method"].values]
    n_methods = len(methods)
    ncols = 2
    nrows = (n_methods + 1) // 2

    pos = _build_dag_layout(G)
    nodes_in_layout = set(pos.keys())
    # Subgraph with only nodes that have layout positions
    Gv = G.subgraph(nodes_in_layout).copy()

    # Compute per-method per-node average residual
    # (only endogenous nodes — edges to other nodes get 0)
    node_residuals = {}
    for method in methods:
        method_df = results_df[results_df["method"] == method]
        node_res = {}
        for node in ENDOGENOUS_NODES:
            if node in nodes_in_layout:
                node_res[node] = method_df["causal_validity_residual"].mean()
        node_residuals[method] = node_res

    # Color scale across all methods
    all_vals = [v for method_dict in node_residuals.values()
                for v in method_dict.values()]
    vmin = 0.0
    vmax = max(all_vals) if all_vals else 1.0

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = np.array(axes).flatten()

    cmap = cm.Reds
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        node_res = node_residuals[method]

        edge_colors = []
        edge_widths = []
        for u, v in Gv.edges():
            res = node_res.get(v, 0.0)
            edge_colors.append(cmap(norm(res)))
            edge_widths.append(1.5 + 3.0 * norm(res))

        node_colors = []
        for n in Gv.nodes():
            if n == "income":
                node_colors.append("#FFD700")
            elif n in IMMUTABLE_FEATURES:
                node_colors.append("#AAAAAA")
            else:
                res = node_res.get(n, 0.0)
                node_colors.append(cmap(norm(res)))

        nx.draw_networkx(
            Gv, pos=pos, ax=ax,
            node_color=node_colors, node_size=800,
            edge_color=edge_colors, width=edge_widths,
            arrows=True, arrowsize=15,
            font_size=7, font_weight="bold",
            with_labels=True,
        )
        ax.set_title(method, fontsize=11, fontweight="bold",
                     color=METHOD_COLORS.get(method, "black"))
        ax.axis("off")

    # Hide unused subplots
    for ax_idx in range(n_methods, len(axes)):
        axes[ax_idx].axis("off")

    # Shared colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes[:n_methods], shrink=0.6,
                 label="Avg. Causal Validity Residual")

    fig.suptitle("Causal DAG: Edge Violation Intensity by CF Method",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(fig_dir, "dag_violation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: t-SNE Arrow Map
# ---------------------------------------------------------------------------

def plot_tsne_arrow_map(
    originals_encoded: pd.DataFrame,
    cfs_dict: Dict[str, pd.DataFrame],
    results_df: pd.DataFrame,
    feat_cols: List[str],
    fig_dir: str,
):
    """
    2×2 subplots, one per method.
    Runs t-SNE on combined [originals + all CFs], then draws arrows
    from original position to CF position, colored by causal validity residual.
    """
    methods = [m for m in METHOD_ORDER if m in cfs_dict]
    n_orig = len(originals_encoded)

    orig_X = originals_encoded[feat_cols].values.astype(np.float32)

    # Stack all data for joint t-SNE
    all_X = [orig_X]
    for method in methods:
        cf_X = cfs_dict[method][feat_cols].values.astype(np.float32)
        all_X.append(cf_X)
    combined = np.vstack(all_X)

    print("  Running t-SNE (this may take ~60s)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30,
                max_iter=1000, init="pca", learning_rate="auto")
    embedding = tsne.fit_transform(combined)

    orig_emb = embedding[:n_orig]
    cf_embs = {}
    offset = n_orig
    for method in methods:
        cf_embs[method] = embedding[offset: offset + n_orig]
        offset += n_orig

    n_methods = len(methods)
    ncols = 2
    nrows = (n_methods + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 6 * nrows))
    axes = np.array(axes).flatten()

    for ax_idx, method in enumerate(methods):
        ax = axes[ax_idx]
        method_res = results_df[results_df["method"] == method]
        residuals = method_res.sort_values("instance_idx")["causal_validity_residual"].values

        vmin = residuals.min()
        vmax = residuals.max() if residuals.max() > vmin else vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.RdYlGn_r  # red = high residual (bad), green = low (good)

        cf_emb = cf_embs[method]

        # Plot original points
        ax.scatter(orig_emb[:, 0], orig_emb[:, 1],
                   c="steelblue", s=15, alpha=0.5, label="Original", zorder=2)

        # Draw arrows and CF points
        for j in range(n_orig):
            if j >= len(residuals):
                continue
            color = cmap(norm(residuals[j]))
            ax.annotate(
                "", xy=(cf_emb[j, 0], cf_emb[j, 1]),
                xytext=(orig_emb[j, 0], orig_emb[j, 1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.6),
            )

        ax.scatter(cf_emb[:, 0], cf_emb[:, 1],
                   c=residuals, cmap=cmap, norm=norm,
                   s=20, alpha=0.7, zorder=3)

        ax.set_title(method, fontsize=11, fontweight="bold",
                     color=METHOD_COLORS.get(method, "black"))
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Causal Residual")

    for ax_idx in range(n_methods, len(axes)):
        axes[ax_idx].axis("off")

    fig.suptitle("t-SNE: Original → CF Transitions (colored by Causal Validity Residual)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(fig_dir, "tsne_arrow_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
