"""
Evaluation metrics for counterfactual explanations:
  - Validity (prediction flip)
  - Proximity (L2 in scaled feature space)
  - Plausibility (VAE reconstruction error)
  - Causal validity residual (SCM consistency, normalized)
  - Immutability violation rate
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import networkx as nx
from typing import Dict, List, Any

from causal_graph import (
    IMMUTABLE_FEATURES, ENDOGENOUS_NODES, CATEGORICAL_ENDOGENOUS, get_parents,
)


def compute_validity(cf_encoded: np.ndarray, original_label: int, xgb_model) -> int:
    """Returns 1 if CF flips the XGBoost prediction, 0 otherwise."""
    pred = xgb_model.predict(cf_encoded.reshape(1, -1))[0]
    return int(pred != original_label)


def compute_proximity(cf_encoded: np.ndarray, original_encoded: np.ndarray) -> float:
    """L2 distance in encoded (scaled) feature space."""
    return float(np.linalg.norm(cf_encoded - original_encoded))


def compute_plausibility(cf_encoded: np.ndarray, vae_model) -> float:
    """
    VAE reconstruction MSE for the CF point.
    Lower = CF looks more like a real data point.
    """
    vae_model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(cf_encoded).unsqueeze(0)
        x_recon, mu, logvar = vae_model(x)
        return F.mse_loss(x_recon, x, reduction="mean").item()


def compute_immutability_violation(
    cf_encoded: np.ndarray,
    original_encoded: np.ndarray,
    feat_cols: List[str],
) -> int:
    """Returns 1 if any immutable feature changed between original and CF."""
    for i, col in enumerate(feat_cols):
        if col in IMMUTABLE_FEATURES:
            if not np.isclose(cf_encoded[i], original_encoded[i], atol=1e-4):
                return 1
    return 0


def compute_causal_validity_residual(
    cf_encoded: np.ndarray,
    feat_cols: List[str],
    scm_models: Dict[str, Any],
    G: nx.DiGraph,
    feature_stds: Dict[str, float],
) -> float:
    """
    Causal validity residual: measures how much the CF's downstream features
    deviate from what the SCM would predict given the CF's upstream features.

    For each endogenous node v:
        parents_encoded from cf -> scm_predict -> v_predicted
        residual_v = |cf_encoded[v] - v_predicted| / std(v)

    Returns mean normalized residual across all endogenous nodes.
    This is the novel metric. Lower = CF is more causally consistent.
    """
    cf_dict = dict(zip(feat_cols, cf_encoded))
    residuals = []

    for node in ENDOGENOUS_NODES:
        if node not in scm_models or node not in cf_dict:
            continue
        parents = get_parents(G, node, exclude=["income"])
        if not parents:
            continue

        # Check all parents are present
        if not all(p in cf_dict for p in parents):
            continue

        parent_vals = np.array([cf_dict[p] for p in parents]).reshape(1, -1)
        predicted = scm_models[node].predict(parent_vals)[0]

        # Categorical predictions are stored as rounded integers in CFs (matching
        # propagate_scm behaviour), so round here too for a fair comparison.
        if node in CATEGORICAL_ENDOGENOUS:
            predicted = round(float(predicted))

        actual = cf_dict[node]
        residual = abs(actual - predicted)

        # Normalize by feature std to make residuals comparable across features
        std = feature_stds.get(node, 1.0)
        if std < 1e-9:
            std = 1.0
        residuals.append(residual / std)

    if not residuals:
        return 0.0
    return float(np.mean(residuals))


def evaluate_all(
    methods: Dict[str, pd.DataFrame],
    originals_encoded: pd.DataFrame,
    originals_raw: pd.DataFrame,
    xgb_model,
    nn_model,
    vae_model,
    scm_models: Dict[str, Any],
    G: nx.DiGraph,
    feat_cols: List[str],
    encoders: dict,
    scaler,
    feature_stds: Dict[str, float],
) -> pd.DataFrame:
    """
    Evaluate all CF methods across all metrics.

    Returns a long-form DataFrame with columns:
        [method, instance_idx, validity, proximity, plausibility,
         causal_validity_residual, immutability_violation]
    """
    records = []
    orig_arr = originals_encoded[feat_cols].values.astype(np.float32)
    orig_labels = originals_encoded["income"].values.astype(int)

    for method_name, cfs_df in methods.items():
        cf_arr = cfs_df[feat_cols].values.astype(np.float32)

        for i in range(len(originals_encoded)):
            cf = cf_arr[i]
            orig = orig_arr[i]
            orig_label = orig_labels[i]

            validity = compute_validity(cf, orig_label, xgb_model)
            proximity = compute_proximity(cf, orig)
            plausibility = compute_plausibility(cf, vae_model)
            imm_violation = compute_immutability_violation(cf, orig, feat_cols)
            causal_residual = compute_causal_validity_residual(
                cf, feat_cols, scm_models, G, feature_stds
            )

            records.append({
                "method": method_name,
                "instance_idx": i,
                "validity": validity,
                "proximity": proximity,
                "plausibility": plausibility,
                "causal_validity_residual": causal_residual,
                "immutability_violation": imm_violation,
            })

        print(f"  Evaluated: {method_name}")

    return pd.DataFrame(records)
