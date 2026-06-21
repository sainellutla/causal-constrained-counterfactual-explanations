"""
Causal-Constrained Counterfactual Explanations — End-to-End Pipeline

Usage:
    python main.py            # full run (200 test instances)
    python main.py --fast     # quick run (20 instances, reduced GA iterations)
"""

import os
import argparse
import warnings
import importlib.metadata

# Numpy 2.x breaks dice-ml and xgboost internals.
# Check version via importlib so we can import torch first (torch must be
# imported before numpy on Windows to avoid an OpenBLAS/DLL conflict).
# CAUSAL_CF_TESTING=1 bypasses this check for the test suite.
_np_ver = importlib.metadata.version("numpy")
if int(_np_ver.split(".")[0]) >= 2 and not os.environ.get("CAUSAL_CF_TESTING"):
    raise RuntimeError(
        f"numpy {_np_ver} is not supported. Run: pip install numpy==1.26.4"
    )

warnings.filterwarnings("ignore")

# torch must be imported before numpy/sklearn on Windows (OpenBLAS DLL order)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
try:
    import xgboost as xgb
except ImportError:
    xgb = None  # will fail at runtime when XGBoost functions are called
import networkx as nx

from causal_graph import (
    build_dag, topological_order, get_parents,
    ENDOGENOUS_NODES, CATEGORICAL_ENDOGENOUS, IMMUTABLE_FEATURES,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------
class IncomeNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h = self.encoder_fc(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    recon = F.mse_loss(x_recon, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["age", "fnlwgt", "education_num", "hours_per_week", "capital_gain", "capital_loss"]
CATEGORICAL_FEATURES = ["workclass", "education", "marital_status", "occupation",
                         "relationship", "race", "sex", "native_country"]
TARGET = "income"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # before ordering

# Features used in CF / causal analysis (subset — matches causal DAG)
DAG_FEATURES = ["age", "education", "marital_status", "occupation",
                "hours_per_week", "capital_gain", "sex", "race"]

# All model input features (DAG features + extras for better accuracy)
MODEL_FEATURES = ["age", "education", "marital_status", "occupation",
                  "hours_per_week", "capital_gain", "sex", "race",
                  "workclass", "relationship", "education_num",
                  "fnlwgt", "capital_loss", "native_country"]


def load_and_preprocess_data():
    """
    Load Adult Income dataset, clean, encode, and scale.

    Returns
    -------
    df_raw        : pd.DataFrame  (string categoricals, raw numerics)
    df_encoded    : pd.DataFrame  (all integer/float, income as 0/1)
    scaler        : fitted StandardScaler (numeric cols only)
    encoders      : dict of LabelEncoder keyed by categorical column name
    feature_stds  : dict of training std per feature (for causal residual normalization)
    """
    csv_path = os.path.join(DATA_DIR, "adult.csv")

    if not os.path.exists(csv_path):
        _download_adult(csv_path)

    df = pd.read_csv(csv_path, na_values="?")
    df.dropna(inplace=True)

    # Normalize column names: ucimlrepo uses hyphens, urllib fallback uses underscores
    df.columns = [c.replace("-", "_") for c in df.columns]

    # Strip whitespace from all string columns (Adult CSV has leading spaces)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Normalize income label (some versions have trailing period)
    df[TARGET] = df[TARGET].str.replace(".", "", regex=False)
    df[TARGET] = (df[TARGET] == ">50K").astype(int)

    df_raw = df.copy()

    # Encode categoricals
    encoders = {}
    df_encoded = df.copy()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Scale numeric features
    numeric_cols = [c for c in MODEL_FEATURES if c in df_encoded.columns
                    and c not in encoders]
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    feature_stds = {col: df_encoded[col].std() for col in MODEL_FEATURES
                    if col in df_encoded.columns}

    return df_raw, df_encoded, scaler, encoders, feature_stds


def _download_adult(csv_path: str):
    """Download Adult Income dataset from UCI repository."""
    print("Downloading Adult Income dataset...")
    try:
        from ucimlrepo import fetch_ucirepo
        adult = fetch_ucirepo(id=2)
        df = adult.data.features.copy()
        df[TARGET] = adult.data.targets.values.ravel()
        df.to_csv(csv_path, index=False)
        print(f"  Saved to {csv_path}")
        return
    except Exception as e:
        print(f"  ucimlrepo failed ({e}), trying urllib fallback...")

    import urllib.request
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    col_names = ["age", "workclass", "fnlwgt", "education", "education_num",
                 "marital_status", "occupation", "relationship", "race", "sex",
                 "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    urllib.request.urlretrieve(url, csv_path + ".raw")
    df = pd.read_csv(csv_path + ".raw", header=None, names=col_names, na_values=" ?")
    df.to_csv(csv_path, index=False)
    os.remove(csv_path + ".raw")
    print(f"  Saved to {csv_path}")


# ---------------------------------------------------------------------------
# Model training helpers
# ---------------------------------------------------------------------------
def train_or_load_xgb(df_encoded: pd.DataFrame, path: str):
    if os.path.exists(path):
        print("Loading XGBoost from cache...")
        return joblib.load(path)

    print("Training XGBoost classifier...")
    feat_cols = [c for c in MODEL_FEATURES if c in df_encoded.columns]
    X = df_encoded[feat_cols].values
    y = df_encoded[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  XGBoost test accuracy: {acc:.4f}")
    joblib.dump(model, path)
    return model


def train_or_load_nn(df_encoded: pd.DataFrame, path: str, n_epochs: int = 50):
    feat_cols = [c for c in MODEL_FEATURES if c in df_encoded.columns]
    input_dim = len(feat_cols)
    model = IncomeNN(input_dim)

    if os.path.exists(path):
        print("Loading NN from cache...")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        return model, feat_cols

    print("Training neural network...")
    X = df_encoded[feat_cols].values.astype(np.float32)
    y = df_encoded[TARGET].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5,
    )
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_test))
            val_loss = criterion(val_pred, torch.FloatTensor(y_test)).item()
            val_acc = ((val_pred > 0.5).float().numpy() == y_test).mean()

        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), path)
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model, feat_cols


def train_or_load_scm(df_raw: pd.DataFrame, df_encoded: pd.DataFrame,
                      G: nx.DiGraph, encoders: dict, path: str):
    if os.path.exists(path):
        print("Loading SCM from cache...")
        return joblib.load(path)

    print("Training SCM regressors...")
    from sklearn.linear_model import LinearRegression
    scm_models = {}

    for node in ENDOGENOUS_NODES:
        parents = get_parents(G, node, exclude=["income"])
        if not parents:
            continue

        X = df_encoded[parents].values.astype(np.float32)
        # Train on df_encoded (z-scored numeric, LabelEncoded categorical)
        # so SCM operates in the same space as all CF methods
        y = df_encoded[node].values.astype(np.float32)

        reg = LinearRegression()
        reg.fit(X, y)
        r2 = reg.score(X, y)
        print(f"  SCM[{node}] parents={parents} R²={r2:.3f}")
        scm_models[node] = reg

    joblib.dump(scm_models, path)
    return scm_models


def train_or_load_vae(df_encoded: pd.DataFrame, path: str, n_epochs: int = 80):
    feat_cols = [c for c in MODEL_FEATURES if c in df_encoded.columns]
    input_dim = len(feat_cols)
    vae = VAE(input_dim)

    if os.path.exists(path):
        print("Loading VAE from cache...")
        vae.load_state_dict(torch.load(path, map_location="cpu"))
        vae.eval()
        return vae, feat_cols

    print("Training VAE...")
    X = df_encoded[feat_cols].values.astype(np.float32)
    ds = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        vae.train()
        total_loss = 0.0
        for (xb,) in loader:
            optimizer.zero_grad()
            x_recon, mu, logvar = vae(xb)
            loss = vae_loss(x_recon, xb, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            print(f"  VAE Epoch {epoch:3d} | loss={total_loss/len(X):.4f}")

    torch.save(vae.state_dict(), path)
    vae.eval()
    return vae, feat_cols


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick run with 20 test instances and reduced iterations")
    args = parser.parse_args()

    n_test = 20 if args.fast else 200
    ga_gens = 50 if args.fast else 100
    ga_pop = 10 if args.fast else 20
    grad_steps = 200 if args.fast else 500

    for d in [DATA_DIR, MODEL_DIR, FIG_DIR]:
        os.makedirs(d, exist_ok=True)

    # ---- Stage 1: Data ----
    print("\n=== Stage 1: Data Loading ===")
    df_raw, df_encoded, scaler, encoders, feature_stds = load_and_preprocess_data()
    print(f"  Dataset shape: {df_encoded.shape}")
    print(f"  Income distribution: {df_encoded[TARGET].value_counts().to_dict()}")

    G = build_dag()
    feat_cols = [c for c in MODEL_FEATURES if c in df_encoded.columns]

    # ---- Stage 2: Models ----
    print("\n=== Stage 2: Model Training ===")
    xgb_model = train_or_load_xgb(
        df_encoded, os.path.join(MODEL_DIR, "xgb_final.pkl")
    )
    nn_model, nn_feat_cols = train_or_load_nn(
        df_encoded, os.path.join(MODEL_DIR, "nn_final.pt")
    )
    scm_models = train_or_load_scm(
        df_raw, df_encoded, G, encoders,
        os.path.join(MODEL_DIR, "scm_models.pkl")
    )
    vae_model, vae_feat_cols = train_or_load_vae(
        df_encoded, os.path.join(MODEL_DIR, "vae_final.pt")
    )

    # ---- Stage 3: Select test instances ----
    print("\n=== Stage 3: Selecting Test Instances ===")
    _, test_idx = train_test_split(
        df_encoded.index, test_size=0.2, stratify=df_encoded[TARGET], random_state=42
    )
    test_encoded = df_encoded.loc[test_idx].reset_index(drop=True)
    test_raw = df_raw.loc[test_idx].reset_index(drop=True)

    X_test_feat = test_encoded[feat_cols].values
    preds = xgb_model.predict(X_test_feat)
    neg_mask = (preds == 0)
    test_encoded_neg = test_encoded[neg_mask].reset_index(drop=True)
    test_raw_neg = test_raw[neg_mask].reset_index(drop=True)

    if len(test_encoded_neg) >= n_test:
        test_encoded_neg = test_encoded_neg.sample(n_test, random_state=42).reset_index(drop=True)
        test_raw_neg = test_raw_neg.iloc[test_encoded_neg.index].reset_index(drop=True)
    print(f"  Using {len(test_encoded_neg)} test instances (all predicted class 0)")

    # ---- Stage 4: CF Generation ----
    print("\n=== Stage 4: Generating Counterfactuals ===")

    from cf_methods import (
        generate_dice_cfs,
        generate_gradient_cfs,
        generate_causal_gradient_cfs,
        generate_ga_cfs,
    )

    print("  [1/4] DiCE genetic...")
    try:
        cfs_dice = generate_dice_cfs(
            test_encoded_neg, xgb_model, df_encoded, feat_cols, encoders,
        )
    except Exception as e:
        print(f"  DiCE failed ({e}); using original instances as fallback.")
        cfs_dice = test_encoded_neg[feat_cols].copy().reset_index(drop=True)

    print("  [2/4] Gradient NN...")
    cfs_grad = generate_gradient_cfs(
        test_encoded_neg, nn_model, feat_cols, encoders, n_steps=grad_steps,
    )

    print("  [3/4] Causal-aware gradient...")
    cfs_causal_grad = generate_causal_gradient_cfs(
        test_encoded_neg, nn_model, scm_models, G,
        feat_cols, encoders, n_steps=grad_steps,
    )

    print("  [4/4] Causal-constrained GA...")
    cfs_ga = generate_ga_cfs(
        test_encoded_neg, xgb_model, scm_models, G, feat_cols, encoders,
        n_generations=ga_gens, sol_per_pop=ga_pop,
    )

    all_cfs = {
        "DiCE (genetic)": cfs_dice,
        "Gradient (NN)": cfs_grad,
        "Causal Gradient": cfs_causal_grad,
        "Causal GA": cfs_ga,
    }

    # ---- Stage 5: Evaluation ----
    print("\n=== Stage 5: Evaluation ===")
    from evaluate import evaluate_all
    results_df = evaluate_all(
        methods=all_cfs,
        originals_encoded=test_encoded_neg,
        originals_raw=test_raw_neg,
        xgb_model=xgb_model,
        nn_model=nn_model,
        vae_model=vae_model,
        scm_models=scm_models,
        G=G,
        feat_cols=feat_cols,
        encoders=encoders,
        scaler=scaler,
        feature_stds=feature_stds,
    )

    metric_cols = ["validity", "proximity", "plausibility",
                   "causal_validity_residual", "immutability_violation"]
    summary = results_df.groupby("method")[metric_cols].mean().round(4)
    print("\n--- Results Summary ---")
    print(summary.to_string())
    results_df.to_csv(os.path.join(FIG_DIR, "results.csv"), index=False)

    # ---- Stage 6: Visualizations ----
    print("\n=== Stage 6: Visualizations ===")
    from visualize import (
        plot_proximity_vs_causal_validity,
        plot_dag_violation_heatmap,
        plot_tsne_arrow_map,
    )

    plot_proximity_vs_causal_validity(results_df, FIG_DIR)
    plot_dag_violation_heatmap(results_df, G, FIG_DIR)
    plot_tsne_arrow_map(test_encoded_neg, all_cfs, results_df, feat_cols, FIG_DIR)

    print(f"\nDone! Figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
