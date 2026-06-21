"""
Tests for private helper functions in cf_methods.py.
Only tests the three helpers that are testable without torch optimization or PyGAD.
"""

import numpy as np
import pytest

from cf_methods import _immutable_indices, _valid_ranges, _snap_to_valid
from causal_graph import IMMUTABLE_FEATURES


# ---------------------------------------------------------------------------
# _immutable_indices
# ---------------------------------------------------------------------------

def test_immutable_indices_contains_age(feat_cols):
    indices = _immutable_indices(feat_cols)
    assert feat_cols.index("age") in indices


def test_immutable_indices_contains_sex(feat_cols):
    indices = _immutable_indices(feat_cols)
    assert feat_cols.index("sex") in indices


def test_immutable_indices_contains_race(feat_cols):
    indices = _immutable_indices(feat_cols)
    assert feat_cols.index("race") in indices


def test_immutable_indices_does_not_contain_education(feat_cols):
    indices = _immutable_indices(feat_cols)
    assert feat_cols.index("education") not in indices


def test_immutable_indices_count(feat_cols):
    indices = _immutable_indices(feat_cols)
    # Exactly 3 immutable features (age, sex, race) are in feat_cols
    assert len(indices) == 3


# ---------------------------------------------------------------------------
# _valid_ranges
# ---------------------------------------------------------------------------

def test_valid_ranges_shape(feat_cols, df_encoded):
    ranges = _valid_ranges(df_encoded, feat_cols)
    assert ranges.shape == (len(feat_cols), 2)


def test_valid_ranges_max_geq_min(feat_cols, df_encoded):
    ranges = _valid_ranges(df_encoded, feat_cols)
    assert (ranges[:, 1] >= ranges[:, 0]).all()


def test_valid_ranges_matches_dataframe(feat_cols, df_encoded):
    ranges = _valid_ranges(df_encoded, feat_cols)
    for i, col in enumerate(feat_cols):
        if col in df_encoded.columns:
            assert np.isclose(ranges[i, 0], df_encoded[col].min())
            assert np.isclose(ranges[i, 1], df_encoded[col].max())


# ---------------------------------------------------------------------------
# _snap_to_valid
# ---------------------------------------------------------------------------

def test_snap_to_valid_rounds_categoricals(feat_cols, encoders):
    rng = np.random.RandomState(42)
    x = rng.randn(len(feat_cols)).astype(np.float32) * 2 + 2  # values around 2.0
    ranges = np.column_stack([
        np.zeros(len(feat_cols)),
        np.full(len(feat_cols), 10.0),
    ])
    snapped = _snap_to_valid(x, feat_cols, encoders, ranges)
    for i, col in enumerate(feat_cols):
        if col in encoders:
            assert float(snapped[i]) == float(int(round(float(snapped[i])))), (
                f"Categorical {col} not rounded: got {snapped[i]}"
            )


def test_snap_to_valid_clips_above_max(feat_cols, encoders):
    x = np.full(len(feat_cols), 100.0, dtype=np.float32)
    ranges = np.column_stack([
        np.zeros(len(feat_cols)),
        np.full(len(feat_cols), 5.0),
    ])
    snapped = _snap_to_valid(x, feat_cols, encoders, ranges)
    assert (snapped <= 5.0).all(), "Values should be clipped to max=5.0"


def test_snap_to_valid_clips_below_min(feat_cols, encoders):
    x = np.full(len(feat_cols), -100.0, dtype=np.float32)
    ranges = np.column_stack([
        np.zeros(len(feat_cols)),
        np.full(len(feat_cols), 5.0),
    ])
    snapped = _snap_to_valid(x, feat_cols, encoders, ranges)
    assert (snapped >= 0.0).all(), "Values should be clipped to min=0.0"


def test_snap_to_valid_preserves_inrange_values(feat_cols, encoders):
    """Non-categorical features already in range should not change (except clip)."""
    x = np.ones(len(feat_cols), dtype=np.float32) * 2.5
    ranges = np.column_stack([
        np.zeros(len(feat_cols)),
        np.full(len(feat_cols), 10.0),
    ])
    snapped = _snap_to_valid(x, feat_cols, encoders, ranges)
    for i, col in enumerate(feat_cols):
        if col not in encoders:
            assert np.isclose(snapped[i], 2.5), f"Numeric {col} changed unexpectedly"


def test_snap_to_valid_returns_copy(feat_cols, encoders):
    x = np.ones(len(feat_cols), dtype=np.float32)
    ranges = np.column_stack([
        np.zeros(len(feat_cols)),
        np.full(len(feat_cols), 5.0),
    ])
    snapped = _snap_to_valid(x, feat_cols, encoders, ranges)
    snapped[0] = 999.0
    assert x[0] != 999.0, "_snap_to_valid should return a copy, not a view"
