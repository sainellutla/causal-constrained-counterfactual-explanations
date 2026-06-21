"""
Four counterfactual generation strategies:
  1. DiCE genetic (baseline, unconstrained)
  2. Gradient-based NN CF (baseline, unconstrained)
  3. Causal-aware gradient CF (SCM propagation after each step)
  4. Causal-constrained GA via PyGAD (causal penalty in fitness)
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import networkx as nx
from typing import Dict, List, Any

from causal_graph import (
    IMMUTABLE_FEATURES, ENDOGENOUS_NODES,
    propagate_scm, compute_causal_penalty,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _immutable_indices(feat_cols: List[str]) -> List[int]:
    return [i for i, c in enumerate(feat_cols) if c in IMMUTABLE_FEATURES]



def _valid_ranges(df_encoded: pd.DataFrame, feat_cols: List[str]) -> np.ndarray:
    """Returns (n_features, 2) array of [min, max] per feature from training data."""
    ranges = np.zeros((len(feat_cols), 2))
    for i, col in enumerate(feat_cols):
        if col in df_encoded.columns:
            ranges[i, 0] = df_encoded[col].min()
            ranges[i, 1] = df_encoded[col].max()
    return ranges


def _snap_to_valid(x: np.ndarray, feat_cols: List[str],
                   encoders: dict, ranges: np.ndarray) -> np.ndarray:
    """Round categorical features to nearest int and clip all to valid range."""
    x = x.copy()
    for i, col in enumerate(feat_cols):
        if col in encoders:
            x[i] = int(round(float(x[i])))
        x[i] = np.clip(x[i], ranges[i, 0], ranges[i, 1])
    return x



# ---------------------------------------------------------------------------
# Method 1: DiCE Genetic
# ---------------------------------------------------------------------------

def generate_dice_cfs(
    test_encoded: pd.DataFrame,
    xgb_model,
    df_encoded: pd.DataFrame,
    feat_cols: List[str],
    encoders: dict,
) -> pd.DataFrame:
    """
    DiCE genetic algorithm CF generation (baseline, unconstrained).
    Uses XGBoost with sklearn backend. Continuous features: age, hours_per_week, capital_gain.
    """
    import dice_ml

    continuous_features = [c for c in feat_cols
                           if c not in encoders and c != "income"]

    train_df = df_encoded[feat_cols + ["income"]].copy().reset_index(drop=True)
    # DiCE needs the target column named "income"
    # Use encoded (int/float) DataFrame throughout

    d = dice_ml.Data(
        dataframe=train_df,
        continuous_features=continuous_features,
        outcome_name="income",
    )

    # Wrap XGBoost for DiCE sklearn backend:
    # predict() must return class labels, predict_proba() must return probability array
    class XGBWrapper:
        def __init__(self, model):
            self.model = model

        def predict(self, X):
            proba = self.model.predict_proba(X)
            return (proba[:, 1] >= 0.5).astype(int)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

    m = dice_ml.Model(model=XGBWrapper(xgb_model), backend="sklearn")
    exp = dice_ml.Dice(d, m, method="genetic")

    cfs_list = []
    for i in range(len(test_encoded)):
        query = test_encoded[feat_cols].iloc[[i]].reset_index(drop=True)
        try:
            result = exp.generate_counterfactuals(
                query,
                total_CFs=1,
                desired_class="opposite",
                verbose=False,
            )
            cf_df = result.cf_examples_list[0].final_cfs_df
            if cf_df is not None and len(cf_df) > 0:
                cf_row = cf_df.iloc[0][feat_cols].values.astype(np.float32)
            else:
                cf_row = None
        except Exception:
            cf_row = None

        if cf_row is None:
            # Fallback: return original (marks as invalid CF in evaluation)
            cf_row = test_encoded[feat_cols].iloc[i].values.astype(np.float32)

        cfs_list.append(cf_row)

        if (i + 1) % 20 == 0:
            print(f"    DiCE: {i+1}/{len(test_encoded)}")

    return pd.DataFrame(cfs_list, columns=feat_cols)


# ---------------------------------------------------------------------------
# Method 2: Gradient-based NN CF (unconstrained)
# ---------------------------------------------------------------------------

def generate_gradient_cfs(
    test_encoded: pd.DataFrame,
    nn_model,
    feat_cols: List[str],
    encoders: dict,
    n_steps: int = 500,
    lr: float = 0.01,
    lambda_proximity: float = 0.5,
) -> pd.DataFrame:
    """
    Unconstrained gradient CF for the NN.
    Immutable features are masked back after each gradient step.
    Categorical features are snapped to valid integers in post-processing.
    """
    ranges = _valid_ranges(test_encoded, feat_cols)
    imm_idx = _immutable_indices(feat_cols)

    X = test_encoded[feat_cols].values.astype(np.float32)
    y = test_encoded["income"].values.astype(np.float32)
    criterion = nn.BCELoss()
    nn_model.eval()

    cfs_list = []
    for i in range(len(test_encoded)):
        x0 = torch.FloatTensor(X[i])
        original_label = float(y[i])
        target_label = 1.0 - original_label

        x_cf = x0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            pred = nn_model(x_cf.unsqueeze(0)).squeeze()
            loss_flip = criterion(pred, torch.tensor(target_label))
            loss_prox = torch.mean((x_cf - x0) ** 2)
            loss = loss_flip + lambda_proximity * loss_prox
            loss.backward()
            optimizer.step()

            # Enforce immutability
            with torch.no_grad():
                for idx in imm_idx:
                    x_cf.data[idx] = x0[idx]

        cf_arr = x_cf.detach().numpy()
        cf_arr = _snap_to_valid(cf_arr, feat_cols, encoders, ranges)
        # Re-enforce immutable features after snapping
        for idx in imm_idx:
            cf_arr[idx] = X[i][idx]

        cfs_list.append(cf_arr)

        if (i + 1) % 20 == 0:
            print(f"    Gradient: {i+1}/{len(test_encoded)}")

    return pd.DataFrame(cfs_list, columns=feat_cols)


# ---------------------------------------------------------------------------
# Method 3: Causal-aware gradient CF
# ---------------------------------------------------------------------------

def generate_causal_gradient_cfs(
    test_encoded: pd.DataFrame,
    nn_model,
    scm_models: dict,
    G: nx.DiGraph,
    feat_cols: List[str],
    encoders: dict,
    n_steps: int = 500,
    lr: float = 0.01,
    lambda_proximity: float = 0.5,
) -> pd.DataFrame:
    """
    Causal-aware gradient CF: same gradient loop as Method 2, but after each
    step we propagate downstream features through the SCM.

    Operates entirely in encoded space (z-scored numeric, LabelEncoded categorical).
    Gradient does NOT flow through the SCM — it is a numpy constraint layer applied
    inside torch.no_grad().
    """
    ranges = _valid_ranges(test_encoded, feat_cols)
    imm_idx = _immutable_indices(feat_cols)

    # Root (non-immutable, non-endogenous, non-income) features in the causal DAG
    # Extra model features not in G (workclass, fnlwgt, etc.) are excluded
    upstream_roots = [c for c in feat_cols
                      if c not in IMMUTABLE_FEATURES
                      and c not in ENDOGENOUS_NODES
                      and c != "income"
                      and c in G.nodes()]

    X = test_encoded[feat_cols].values.astype(np.float32)
    y = test_encoded["income"].values.astype(np.float32)
    criterion = nn.BCELoss()
    nn_model.eval()

    cfs_list = []
    for i in range(len(test_encoded)):
        x0 = torch.FloatTensor(X[i])
        original_label = float(y[i])
        target_label = 1.0 - original_label

        x_cf = x0.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([x_cf], lr=lr)

        for step in range(n_steps):
            optimizer.zero_grad()
            pred = nn_model(x_cf.unsqueeze(0)).squeeze()
            loss_flip = criterion(pred, torch.tensor(target_label))
            loss_prox = torch.mean((x_cf - x0) ** 2)
            loss = loss_flip + lambda_proximity * loss_prox
            loss.backward()
            optimizer.step()

            # Enforce immutability
            with torch.no_grad():
                for idx in imm_idx:
                    x_cf.data[idx] = x0[idx]

            # SCM propagation in encoded space — no gradient flow
            with torch.no_grad():
                cf_arr = x_cf.detach().numpy().copy()
                # Find which root upstream features changed (compared to original)
                changed = [c for c in upstream_roots
                           if not np.isclose(
                               cf_arr[feat_cols.index(c)],
                               X[i][feat_cols.index(c)],
                               atol=1e-3,
                           )]
                if changed:
                    cf_propagated = propagate_scm(
                        cf_arr, feat_cols, scm_models, G, encoders, changed
                    )
                    for idx, col in enumerate(feat_cols):
                        if col not in IMMUTABLE_FEATURES:
                            x_cf.data[idx] = float(cf_propagated[idx])

        cf_arr = x_cf.detach().numpy()
        cf_arr = _snap_to_valid(cf_arr, feat_cols, encoders, ranges)
        for idx in imm_idx:
            cf_arr[idx] = X[i][idx]

        cfs_list.append(cf_arr)

        if (i + 1) % 20 == 0:
            print(f"    Causal Gradient: {i+1}/{len(test_encoded)}")

    return pd.DataFrame(cfs_list, columns=feat_cols)


# ---------------------------------------------------------------------------
# Method 4: Causal-constrained GA (PyGAD)
# ---------------------------------------------------------------------------

def generate_ga_cfs(
    test_encoded: pd.DataFrame,
    xgb_model,
    scm_models: dict,
    G: nx.DiGraph,
    feat_cols: List[str],
    encoders: dict,
    n_generations: int = 100,
    sol_per_pop: int = 20,
    lambda_causal: float = 1.0,
    lambda_proximity: float = 0.5,
) -> pd.DataFrame:
    """
    Causal-constrained GA CF using PyGAD.
    Fitness = xgb_proba(opposite_class) - λ_prox*L2 - λ_causal*causal_penalty
    """
    import pygad

    ranges = _valid_ranges(test_encoded, feat_cols)
    imm_idx = _immutable_indices(feat_cols)

    # Build gene_space: list of dicts {low, high} per feature
    gene_space = []
    for i, col in enumerate(feat_cols):
        if col in encoders:
            n_classes = len(encoders[col].classes_)
            gene_space.append(list(range(n_classes)))
        else:
            gene_space.append({
                "low": float(ranges[i, 0]),
                "high": float(ranges[i, 1]),
            })

    X = test_encoded[feat_cols].values.astype(np.float32)
    y = test_encoded["income"].values.astype(np.float32)

    # Feature range for L2 normalization
    feat_range = ranges[:, 1] - ranges[:, 0]
    feat_range[feat_range == 0] = 1.0  # avoid division by zero

    cfs_list = []
    for i in range(len(test_encoded)):
        x0 = X[i].copy()
        original_label = int(y[i])
        target_label = 1 - original_label

        def fitness_func(ga_instance, solution, solution_idx):
            sol = solution.copy()
            # Enforce immutable features
            for idx in imm_idx:
                sol[idx] = x0[idx]
            # Snap categoricals
            for j, col in enumerate(feat_cols):
                if col in encoders:
                    n_classes = len(encoders[col].classes_)
                    sol[j] = max(0, min(n_classes - 1, int(round(float(sol[j])))))

            sol_2d = sol.reshape(1, -1)
            proba = xgb_model.predict_proba(sol_2d)[0]
            validity_score = proba[target_label]

            # Proximity penalty (normalized L2)
            prox = np.linalg.norm((sol - x0) / feat_range)

            # Causal consistency penalty (encoded space)
            causal_pen = compute_causal_penalty(sol, feat_cols, scm_models, G, encoders)

            return float(validity_score
                         - lambda_proximity * prox
                         - lambda_causal * causal_pen)

        ga = pygad.GA(
            num_generations=n_generations,
            num_parents_mating=max(2, sol_per_pop // 2),
            sol_per_pop=sol_per_pop,
            num_genes=len(feat_cols),
            fitness_func=fitness_func,
            gene_space=gene_space,
            parent_selection_type="sss",
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=10,
            keep_elitism=2,
            suppress_warnings=True,
            save_solutions=False,
        )
        ga.run()

        best_sol, _, _ = ga.best_solution()
        best_sol = best_sol.copy()

        # Post-processing
        for idx in imm_idx:
            best_sol[idx] = x0[idx]
        best_sol = _snap_to_valid(best_sol, feat_cols, encoders, ranges)
        for idx in imm_idx:
            best_sol[idx] = x0[idx]

        cfs_list.append(best_sol.astype(np.float32))

        if (i + 1) % 5 == 0:
            print(f"    GA: {i+1}/{len(test_encoded)}")

    return pd.DataFrame(cfs_list, columns=feat_cols)
