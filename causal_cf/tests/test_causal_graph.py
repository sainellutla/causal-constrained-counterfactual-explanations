"""
Tests for causal_graph.py — DAG structure, topological order,
parent/descendant queries, SCM propagation, and causal penalty.
"""

import numpy as np
import pytest
from causal_graph import (
    build_dag, topological_order, get_parents, get_descendants,
    propagate_scm, compute_causal_penalty,
    IMMUTABLE_FEATURES, ENDOGENOUS_NODES, CATEGORICAL_ENDOGENOUS,
    CAUSAL_EDGES,
)


# ---------------------------------------------------------------------------
# DAG structure
# ---------------------------------------------------------------------------

def test_build_dag_has_income_node(dag):
    assert "income" in dag.nodes


def test_build_dag_key_edges(dag):
    assert dag.has_edge("age", "education")
    assert dag.has_edge("education", "occupation")
    assert dag.has_edge("occupation", "capital_gain")
    assert dag.has_edge("education", "hours_per_week")
    assert dag.has_edge("age", "marital_status")


def test_build_dag_all_edges_present(dag):
    for u, v in CAUSAL_EDGES:
        assert dag.has_edge(u, v), f"Missing edge {u} -> {v}"


def test_immutable_features_are_correct():
    assert IMMUTABLE_FEATURES == ["age", "sex", "race"]


# ---------------------------------------------------------------------------
# Topological order
# ---------------------------------------------------------------------------

def test_topological_order_excludes_income(dag):
    order = topological_order(dag)
    assert "income" not in order


def test_topological_order_age_before_education(dag):
    order = topological_order(dag)
    assert order.index("age") < order.index("education")


def test_topological_order_education_before_occupation(dag):
    order = topological_order(dag)
    assert order.index("education") < order.index("occupation")


def test_topological_order_occupation_before_capital_gain(dag):
    order = topological_order(dag)
    assert order.index("occupation") < order.index("capital_gain")


# ---------------------------------------------------------------------------
# get_parents
# ---------------------------------------------------------------------------

def test_get_parents_education_has_age(dag):
    parents = get_parents(dag, "education")
    assert "age" in parents


def test_get_parents_occupation_has_education(dag):
    parents = get_parents(dag, "occupation")
    assert "education" in parents


def test_get_parents_capital_gain_has_occupation(dag):
    parents = get_parents(dag, "capital_gain")
    assert "occupation" in parents


def test_get_parents_excludes_income(dag):
    # income should never appear as a parent (it's excluded)
    for node in dag.nodes:
        parents = get_parents(dag, node)
        assert "income" not in parents


# ---------------------------------------------------------------------------
# get_descendants
# ---------------------------------------------------------------------------

def test_get_descendants_age_includes_downstream(dag):
    desc = get_descendants(dag, ["age"])
    assert "education" in desc
    assert "occupation" in desc
    assert "capital_gain" in desc


def test_get_descendants_excludes_income(dag):
    desc = get_descendants(dag, ["age"])
    assert "income" not in desc


def test_get_descendants_capital_gain_empty(dag):
    # capital_gain → income (excluded) only; no other descendants
    desc = get_descendants(dag, ["capital_gain"])
    assert desc == []


def test_get_descendants_empty_input(dag):
    desc = get_descendants(dag, [])
    assert desc == []


# ---------------------------------------------------------------------------
# propagate_scm
# ---------------------------------------------------------------------------

def test_propagate_scm_no_changed_returns_unchanged(dag, feat_cols, scm_models, encoders, encoded_instance):
    result = propagate_scm(encoded_instance, feat_cols, scm_models, dag, encoders, [])
    np.testing.assert_array_equal(result, encoded_instance)


def test_propagate_scm_returns_ndarray(dag, feat_cols, scm_models, encoders, encoded_instance):
    result = propagate_scm(encoded_instance, feat_cols, scm_models, dag, encoders, ["age"])
    assert isinstance(result, np.ndarray)


def test_propagate_scm_preserves_length(dag, feat_cols, scm_models, encoders, encoded_instance):
    result = propagate_scm(encoded_instance, feat_cols, scm_models, dag, encoders, ["age"])
    assert len(result) == len(feat_cols)


def test_propagate_scm_categorical_clipped_to_valid_range(dag, feat_cols, scm_models, encoders, encoded_instance):
    result = propagate_scm(encoded_instance, feat_cols, scm_models, dag, encoders, ["age"])
    for node in CATEGORICAL_ENDOGENOUS:
        if node in feat_cols and node in encoders:
            idx = feat_cols.index(node)
            n_classes = len(encoders[node].classes_)
            val = result[idx]
            assert 0 <= int(round(float(val))) < n_classes, (
                f"{node} predicted value {val} out of [0, {n_classes-1}]"
            )


def test_propagate_scm_does_not_modify_input(dag, feat_cols, scm_models, encoders, encoded_instance):
    original = encoded_instance.copy()
    propagate_scm(encoded_instance, feat_cols, scm_models, dag, encoders, ["age"])
    np.testing.assert_array_equal(encoded_instance, original)


def test_propagate_scm_immutables_not_recomputed(dag, feat_cols, scm_models, encoders, encoded_instance):
    result = propagate_scm(encoded_instance, feat_cols, scm_models, dag, encoders, ["age"])
    # sex and race are not endogenous nodes → they must not be touched by propagation
    for feat in ["sex", "race"]:
        if feat in feat_cols:
            idx = feat_cols.index(feat)
            assert result[idx] == encoded_instance[idx]


# ---------------------------------------------------------------------------
# compute_causal_penalty
# ---------------------------------------------------------------------------

def test_compute_causal_penalty_nonnegative(dag, feat_cols, scm_models, encoders):
    rng = np.random.RandomState(99)
    x = rng.randn(len(feat_cols)).astype(np.float32)
    penalty = compute_causal_penalty(x, feat_cols, scm_models, dag, encoders)
    assert penalty >= 0.0


def test_compute_causal_penalty_zero_for_perfect_cf(dag, feat_cols, scm_models, encoders):
    """A CF where downstream values exactly match SCM predictions → penalty ≈ 0."""
    x = np.zeros(len(feat_cols), dtype=np.float32)
    feat_to_idx = {col: i for i, col in enumerate(feat_cols)}

    for node in ENDOGENOUS_NODES:
        parents = get_parents(dag, node)
        if not parents or node not in scm_models:
            continue
        parent_vals = np.array([x[feat_to_idx[p]] for p in parents]).reshape(1, -1)
        predicted = scm_models[node].predict(parent_vals)[0]
        if node in CATEGORICAL_ENDOGENOUS and node in encoders:
            n_classes = len(encoders[node].classes_)
            predicted = max(0, min(n_classes - 1, int(round(float(predicted)))))
        x[feat_to_idx[node]] = predicted

    penalty = compute_causal_penalty(x, feat_cols, scm_models, dag, encoders)
    assert penalty < 1e-5
