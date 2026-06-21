"""
Causal DAG definition and SCM propagation logic for Adult Income dataset.

All SCM operations work in encoded space:
  - Numeric features: StandardScaler z-scored
  - Categorical features: LabelEncoder integer codes
This matches how SCM models are trained (on df_encoded).
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any

IMMUTABLE_FEATURES = ["age", "sex", "race"]

CAUSAL_EDGES = [
    ("age", "education"),
    ("age", "marital_status"),
    ("education", "occupation"),
    ("education", "hours_per_week"),
    ("occupation", "capital_gain"),
    ("age", "income"),
    ("education", "income"),
    ("occupation", "income"),
    ("hours_per_week", "income"),
    ("capital_gain", "income"),
]

# Nodes predicted by SCM regressors (non-root, non-target)
ENDOGENOUS_NODES = ["education", "marital_status", "occupation", "hours_per_week", "capital_gain"]

# Categorical endogenous nodes (SCM output is float → round to int LabelCode)
CATEGORICAL_ENDOGENOUS = ["education", "marital_status", "occupation"]

# Numeric endogenous nodes (SCM output is z-scored float)
NUMERIC_ENDOGENOUS = ["hours_per_week", "capital_gain"]


def build_dag() -> nx.DiGraph:
    """Build and return the causal DAG (income is a node but not an SCM target)."""
    G = nx.DiGraph()
    for u, v in CAUSAL_EDGES:
        G.add_edge(u, v)
    return G


def topological_order(G: nx.DiGraph, exclude: List[str] = None) -> List[str]:
    """Return nodes in topological sort order, optionally excluding specified nodes."""
    if exclude is None:
        exclude = ["income"]
    order = list(nx.topological_sort(G))
    return [n for n in order if n not in exclude]


def get_parents(G: nx.DiGraph, node: str, exclude: List[str] = None) -> List[str]:
    """Return list of parent nodes for a given node."""
    if exclude is None:
        exclude = ["income"]
    return [p for p in G.predecessors(node) if p not in exclude]


def get_descendants(G: nx.DiGraph, nodes: List[str], exclude: List[str] = None) -> List[str]:
    """Return all descendants of a set of nodes (union), in topological order."""
    if exclude is None:
        exclude = ["income"]
    desc_set = set()
    for n in nodes:
        desc_set.update(nx.descendants(G, n))
    desc_set -= set(exclude)
    topo = topological_order(G, exclude=exclude)
    return [n for n in topo if n in desc_set]


def propagate_scm(
    cf_encoded: np.ndarray,
    feat_cols: List[str],
    scm_models: Dict[str, Any],
    G: nx.DiGraph,
    encoders: Dict[str, Any],
    changed_upstream_nodes: List[str],
) -> np.ndarray:
    """
    Propagate downstream features through SCM in encoded space.

    Both inputs and outputs are in encoded space:
      - Numeric features: z-scored (StandardScaler)
      - Categorical features: integer LabelEncoder codes

    Parameters
    ----------
    cf_encoded             : Encoded feature vector (1D numpy array)
    feat_cols              : Feature column names, aligned with cf_encoded
    scm_models             : Dict mapping node_name -> fitted sklearn regressor
    G                      : The causal DAG
    encoders               : Dict of LabelEncoders (for clipping categorical outputs)
    changed_upstream_nodes : Nodes explicitly changed by the CF method

    Returns
    -------
    Updated encoded feature vector (same length as cf_encoded)
    """
    cf = cf_encoded.copy()
    feat_to_idx = {col: i for i, col in enumerate(feat_cols)}

    to_recompute = get_descendants(G, changed_upstream_nodes, exclude=["income"])
    to_recompute = [n for n in to_recompute if n in scm_models and n in feat_to_idx]

    for node in to_recompute:
        parents = get_parents(G, node, exclude=["income"])
        if not parents or not all(p in feat_to_idx for p in parents):
            continue

        # Parents are already in encoded space — pass directly to SCM
        parent_vals = np.array([cf[feat_to_idx[p]] for p in parents]).reshape(1, -1)
        predicted = scm_models[node].predict(parent_vals)[0]

        if node in CATEGORICAL_ENDOGENOUS:
            # Clip predicted float to valid LabelEncoder integer range
            n_classes = len(encoders[node].classes_)
            predicted = max(0, min(n_classes - 1, int(round(float(predicted)))))
        # Numeric endogenous nodes: keep predicted z-scored float as-is

        cf[feat_to_idx[node]] = predicted

    return cf


def compute_causal_penalty(
    cf_encoded: np.ndarray,
    feature_names: List[str],
    scm_models: Dict[str, Any],
    G: nx.DiGraph,
    encoders: Dict[str, Any],
) -> float:
    """
    Causal consistency penalty for GA fitness function.
    Operates entirely in encoded (z-scored numeric, LabelEncoded categorical) space.

    Returns sum of |actual_v - scm_predicted_v| for all endogenous nodes.
    """
    feat_to_idx = {col: i for i, col in enumerate(feature_names)}
    penalty = 0.0

    for node in ENDOGENOUS_NODES:
        if node not in scm_models or node not in feat_to_idx:
            continue
        parents = get_parents(G, node, exclude=["income"])
        if not parents or not all(p in feat_to_idx for p in parents):
            continue

        parent_vals = np.array([cf_encoded[feat_to_idx[p]] for p in parents]).reshape(1, -1)
        predicted = scm_models[node].predict(parent_vals)[0]

        if node in CATEGORICAL_ENDOGENOUS:
            n_classes = len(encoders[node].classes_)
            predicted = max(0, min(n_classes - 1, int(round(float(predicted)))))

        actual = cf_encoded[feat_to_idx[node]]
        penalty += abs(actual - predicted)

    return penalty
