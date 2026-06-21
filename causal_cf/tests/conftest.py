"""
Shared pytest fixtures for the causal-CF test suite.
No real data download or model training required — all synthetic.
"""

import sys
import os

# Bypass the numpy 2.x guard in main.py — tests don't use dice-ml/xgboost
os.environ["CAUSAL_CF_TESTING"] = "1"
# Force non-interactive matplotlib backend (Tk may not be available in CI)
os.environ["MPLBACKEND"] = "Agg"

# Make causal_cf/ importable without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

# ---------------------------------------------------------------------------
# Constants (mirror MODEL_FEATURES from main.py)
# ---------------------------------------------------------------------------
MODEL_FEATURES = [
    "age", "education", "marital_status", "occupation",
    "hours_per_week", "capital_gain", "sex", "race",
    "workclass", "relationship", "education_num",
    "fnlwgt", "capital_loss", "native_country",
]
CATEGORICAL_COLS = [
    "education", "marital_status", "occupation", "sex", "race",
    "workclass", "relationship", "native_country",
]
NUMERIC_COLS = [c for c in MODEL_FEATURES if c not in CATEGORICAL_COLS]
N_CLASSES = 5  # number of dummy classes per categorical feature
N_SAMPLES = 100
RNG = np.random.RandomState(42)

METHOD_NAMES = ["DiCE (genetic)", "Gradient (NN)", "Causal Gradient", "Causal GA"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def feat_cols():
    return list(MODEL_FEATURES)


@pytest.fixture(scope="session")
def encoders():
    enc = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit([f"cls_{i}" for i in range(N_CLASSES)])
        enc[col] = le
    return enc


@pytest.fixture(scope="session")
def dag():
    from causal_graph import build_dag
    return build_dag()


@pytest.fixture(scope="session")
def scm_models(dag, feat_cols, encoders):
    from causal_graph import ENDOGENOUS_NODES, get_parents
    models = {}
    for node in ENDOGENOUS_NODES:
        parents = get_parents(dag, node)
        if not parents:
            continue
        n_parents = len(parents)
        X = RNG.randn(N_SAMPLES, n_parents).astype(np.float32)
        y = RNG.randn(N_SAMPLES).astype(np.float32)
        reg = LinearRegression().fit(X, y)
        models[node] = reg
    return models


@pytest.fixture(scope="session")
def encoded_instance(feat_cols, encoders):
    """A valid encoded feature vector: z-scored numerics, int categorical codes."""
    x = RNG.randn(len(feat_cols)).astype(np.float32)
    for i, col in enumerate(feat_cols):
        if col in encoders:
            x[i] = float(RNG.randint(0, N_CLASSES))
    return x


@pytest.fixture(scope="session")
def feature_stds(feat_cols):
    return {col: 1.0 for col in feat_cols}


@pytest.fixture(scope="session")
def df_encoded(feat_cols, encoders):
    """100-row synthetic encoded DataFrame with all MODEL_FEATURES + income."""
    data = {}
    for col in feat_cols:
        if col in encoders:
            data[col] = RNG.randint(0, N_CLASSES, size=N_SAMPLES).astype(np.float32)
        else:
            data[col] = RNG.randn(N_SAMPLES).astype(np.float32)
    data["income"] = RNG.randint(0, 2, size=N_SAMPLES).astype(np.float32)
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def mock_xgb():
    """DummyClassifier that always predicts class 1."""
    clf = DummyClassifier(strategy="constant", constant=1)
    X_dummy = np.zeros((4, len(MODEL_FEATURES)), dtype=np.float32)
    y_dummy = np.array([0, 1, 0, 1])
    clf.fit(X_dummy, y_dummy)
    return clf


@pytest.fixture(scope="session")
def mock_vae(feat_cols):
    """Real VAE with random weights in eval mode."""
    from main import VAE
    vae = VAE(input_dim=len(feat_cols), latent_dim=4)
    vae.eval()
    return vae


@pytest.fixture(scope="session")
def results_df(feat_cols):
    """Long-form results DataFrame with all 4 method names, 10 instances each."""
    records = []
    for method in METHOD_NAMES:
        for i in range(10):
            records.append({
                "method": method,
                "instance_idx": i,
                "validity": RNG.randint(0, 2),
                "proximity": float(RNG.uniform(0.1, 3.0)),
                "plausibility": float(RNG.uniform(0.01, 1.0)),
                "causal_validity_residual": float(RNG.uniform(0.0, 2.0)),
                "immutability_violation": RNG.randint(0, 2),
            })
    return pd.DataFrame(records)


@pytest.fixture(scope="session")
def cfs_dict(feat_cols, encoders):
    """Dict of method_name → DataFrame of 10 synthetic CFs."""
    d = {}
    for method in METHOD_NAMES:
        data = {}
        for col in feat_cols:
            if col in encoders:
                data[col] = RNG.randint(0, N_CLASSES, size=10).astype(np.float32)
            else:
                data[col] = RNG.randn(10).astype(np.float32)
        d[method] = pd.DataFrame(data)
    return d


@pytest.fixture(scope="session")
def originals_df(feat_cols, encoders):
    """10-row synthetic encoded DataFrame for t-SNE tests."""
    data = {}
    for col in feat_cols:
        if col in encoders:
            data[col] = RNG.randint(0, N_CLASSES, size=10).astype(np.float32)
        else:
            data[col] = RNG.randn(10).astype(np.float32)
    data["income"] = np.zeros(10, dtype=np.float32)
    return pd.DataFrame(data)
