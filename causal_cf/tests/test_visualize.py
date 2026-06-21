"""
Tests for visualize.py — DAG layout, method palette, and plot file output.
All plot tests write to pytest's tmp_path so no real figures/ dir is needed.
t-SNE tests use n_iter=250 and n=10 instances to stay fast (< 5s).
"""

import os
import pytest
import numpy as np
import pandas as pd

from visualize import (
    _build_dag_layout,
    _method_palette,
    plot_proximity_vs_causal_validity,
    plot_dag_violation_heatmap,
    plot_tsne_arrow_map,
)
from conftest import METHOD_NAMES


# ---------------------------------------------------------------------------
# _build_dag_layout
# ---------------------------------------------------------------------------

def test_build_dag_layout_returns_dict(dag):
    pos = _build_dag_layout(dag)
    assert isinstance(pos, dict)


def test_build_dag_layout_has_age(dag):
    pos = _build_dag_layout(dag)
    assert "age" in pos


def test_build_dag_layout_has_income(dag):
    pos = _build_dag_layout(dag)
    assert "income" in pos


def test_build_dag_layout_all_coords_are_2d(dag):
    pos = _build_dag_layout(dag)
    for node, coords in pos.items():
        assert len(coords) == 2, f"Node {node} has {len(coords)}-D coords, expected 2"


def test_build_dag_layout_covers_endogenous_nodes(dag):
    from causal_graph import ENDOGENOUS_NODES
    pos = _build_dag_layout(dag)
    for node in ENDOGENOUS_NODES:
        assert node in pos, f"Endogenous node {node} missing from DAG layout"


# ---------------------------------------------------------------------------
# _method_palette
# ---------------------------------------------------------------------------

def test_method_palette_covers_all_methods(results_df):
    palette = _method_palette(results_df)
    for method in results_df["method"].unique():
        assert method in palette, f"Method {method} not in palette"


def test_method_palette_returns_strings(results_df):
    palette = _method_palette(results_df)
    for method, color in palette.items():
        assert isinstance(color, str), f"Color for {method} is not a string: {color}"


# ---------------------------------------------------------------------------
# Plot file output tests
# ---------------------------------------------------------------------------

def test_plot_proximity_saves_png(results_df, tmp_path):
    plot_proximity_vs_causal_validity(results_df, str(tmp_path))
    assert (tmp_path / "boxplot_proximity_causal.png").exists()


def test_plot_dag_saves_png(results_df, dag, tmp_path):
    plot_dag_violation_heatmap(results_df, dag, str(tmp_path))
    assert (tmp_path / "dag_violation_heatmap.png").exists()


def test_plot_tsne_saves_png(originals_df, cfs_dict, results_df, feat_cols, tmp_path, monkeypatch):
    """
    Patch TSNE to use minimal settings so this test runs in < 5 seconds.
    """
    import sklearn.manifold

    original_tsne_init = sklearn.manifold.TSNE.__init__

    def fast_tsne_init(self, **kwargs):
        kwargs["max_iter"] = 250
        kwargs["perplexity"] = min(kwargs.get("perplexity", 5), 5)
        original_tsne_init(self, **kwargs)

    monkeypatch.setattr(sklearn.manifold.TSNE, "__init__", fast_tsne_init)

    plot_tsne_arrow_map(originals_df, cfs_dict, results_df, feat_cols, str(tmp_path))
    assert (tmp_path / "tsne_arrow_map.png").exists()
