"""
Tests for evaluate.py — all five metrics.
Uses synthetic fixtures only (no real data, no trained models).
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from evaluate import (
    compute_validity,
    compute_proximity,
    compute_plausibility,
    compute_immutability_violation,
    compute_causal_validity_residual,
)
from causal_graph import ENDOGENOUS_NODES, get_parents, CATEGORICAL_ENDOGENOUS


# ---------------------------------------------------------------------------
# compute_validity
# ---------------------------------------------------------------------------

def test_validity_flip(mock_xgb, feat_cols):
    # mock_xgb always predicts class 1; original label is 0 → should flip
    cf = np.zeros(len(feat_cols), dtype=np.float32)
    assert compute_validity(cf, 0, mock_xgb) == 1


def test_validity_no_flip(mock_xgb, feat_cols):
    # mock_xgb always predicts class 1; original label is also 1 → no flip
    cf = np.zeros(len(feat_cols), dtype=np.float32)
    assert compute_validity(cf, 1, mock_xgb) == 0


def test_validity_returns_int(mock_xgb, feat_cols):
    cf = np.zeros(len(feat_cols), dtype=np.float32)
    result = compute_validity(cf, 0, mock_xgb)
    assert isinstance(result, (int, np.integer))


# ---------------------------------------------------------------------------
# compute_proximity
# ---------------------------------------------------------------------------

def test_proximity_identical_arrays():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert compute_proximity(x, x) == 0.0


def test_proximity_known_distance():
    a = np.array([0.0, 0.0], dtype=np.float32)
    b = np.array([3.0, 4.0], dtype=np.float32)
    assert np.isclose(compute_proximity(a, b), 5.0), "Expected L2 = 5.0"


def test_proximity_nonnegative():
    rng = np.random.RandomState(7)
    a = rng.randn(10).astype(np.float32)
    b = rng.randn(10).astype(np.float32)
    assert compute_proximity(a, b) >= 0.0


def test_proximity_symmetric():
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    b = np.array([0.0, 5.0, -1.0], dtype=np.float32)
    assert np.isclose(compute_proximity(a, b), compute_proximity(b, a))


# ---------------------------------------------------------------------------
# compute_plausibility
# ---------------------------------------------------------------------------

def test_plausibility_returns_float(mock_vae, encoded_instance):
    result = compute_plausibility(encoded_instance, mock_vae)
    assert isinstance(result, float)


def test_plausibility_nonnegative(mock_vae, encoded_instance):
    result = compute_plausibility(encoded_instance, mock_vae)
    assert result >= 0.0


def test_plausibility_finite(mock_vae, encoded_instance):
    result = compute_plausibility(encoded_instance, mock_vae)
    assert np.isfinite(result)


# ---------------------------------------------------------------------------
# compute_immutability_violation
# ---------------------------------------------------------------------------

def test_no_violation_when_identical(feat_cols, encoded_instance):
    result = compute_immutability_violation(encoded_instance, encoded_instance, feat_cols)
    assert result == 0


def test_violation_when_age_changes(feat_cols, encoded_instance):
    cf = encoded_instance.copy()
    age_idx = feat_cols.index("age")
    cf[age_idx] += 10.0
    result = compute_immutability_violation(cf, encoded_instance, feat_cols)
    assert result == 1


def test_violation_when_sex_changes(feat_cols, encoded_instance):
    cf = encoded_instance.copy()
    sex_idx = feat_cols.index("sex")
    cf[sex_idx] = (cf[sex_idx] + 1) % 5  # change sex (immutable)
    result = compute_immutability_violation(cf, encoded_instance, feat_cols)
    assert result == 1


def test_no_violation_when_education_changes(feat_cols, encoded_instance):
    cf = encoded_instance.copy()
    edu_idx = feat_cols.index("education")
    cf[edu_idx] = (cf[edu_idx] + 1) % 5  # change education (mutable)
    result = compute_immutability_violation(cf, encoded_instance, feat_cols)
    assert result == 0


def test_violation_returns_int(feat_cols, encoded_instance):
    result = compute_immutability_violation(encoded_instance, encoded_instance, feat_cols)
    assert isinstance(result, (int, np.integer))


# ---------------------------------------------------------------------------
# compute_causal_validity_residual
# ---------------------------------------------------------------------------

def test_residual_nonneg_for_random_cf(dag, feat_cols, scm_models, encoded_instance, feature_stds):
    result = compute_causal_validity_residual(
        encoded_instance, feat_cols, scm_models, dag, feature_stds
    )
    assert result >= 0.0


def test_residual_returns_float(dag, feat_cols, scm_models, encoded_instance, feature_stds):
    result = compute_causal_validity_residual(
        encoded_instance, feat_cols, scm_models, dag, feature_stds
    )
    assert isinstance(result, float)


def test_residual_finite(dag, feat_cols, scm_models, encoded_instance, feature_stds):
    result = compute_causal_validity_residual(
        encoded_instance, feat_cols, scm_models, dag, feature_stds
    )
    assert np.isfinite(result)


def test_residual_near_zero_for_scm_consistent_cf(dag, feat_cols, scm_models, encoders, feature_stds):
    """
    Build a CF by running the SCM forward from zero inputs.
    The resulting downstream values exactly match SCM predictions → residual ≈ 0.
    """
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

    result = compute_causal_validity_residual(x, feat_cols, scm_models, dag, feature_stds)
    assert result < 1e-5, f"Expected near-zero residual for SCM-consistent CF, got {result}"
