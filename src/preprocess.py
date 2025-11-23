"""
src/preprocess.py
Deterministic preprocessing used for both training and inference.
Edit NUMERIC_COLS and CATEGORICAL_COLS to match your notebook.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# === EDIT THESE to match the exact columns from your notebook/dataset ===
NUMERIC_COLS = ["age", "autism_screening_score", "other_feature_1", "other_feature_2"]
CATEGORICAL_COLS = ["gender", "ethnicity", "jaundice"]
TARGET_COL = "ASD"  # replace with the exact target column name in your dataset

def fit_and_save_transformers(df: pd.DataFrame, model_dir="models"):
    """Fit scaler and encoder on training data and save them to model_dir."""
    X_num = df[NUMERIC_COLS].astype(float)
    scaler = StandardScaler().fit(X_num)
    joblib.dump(scaler, f"{model_dir}/scaler.pkl")

    X_cat = df[CATEGORICAL_COLS].astype(str)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X_cat)
    joblib.dump(ohe, f"{model_dir}/ohe.pkl")

    print(f"Saved scaler and encoder to {model_dir}")

def load_transformers(model_dir="models"):
    """Load scaler and encoder (if present)."""
    scaler = None
    ohe = None
    try:
        scaler = joblib.load(f"{model_dir}/scaler.pkl")
    except Exception:
        print("Warning: scaler.pkl not found in", model_dir)
    try:
        ohe = joblib.load(f"{model_dir}/ohe.pkl")
    except Exception:
        print("Warning: ohe.pkl not found in", model_dir)
    return scaler, ohe

def preprocess(df: pd.DataFrame, model_dir="models"):
    """
    Preprocess input DataFrame and return numpy array ready for model.predict.
    Ensure that NUMERIC_COLS and CATEGORICAL_COLS match notebook columns.
    """
    df = df.copy()

    # --- Basic cleaning ---
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = 0.0  # fallback if column missing

    for c in CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("missing").astype(str)
        else:
            df[c] = "missing"

    # --- Load transformers if available ---
    scaler, ohe = load_transformers(model_dir=model_dir)

    # numeric transform
    X_num = df[NUMERIC_COLS].astype(float).values
    if scaler is not None:
        X_num = scaler.transform(X_num)

    # categorical transform
    X_cat = df[CATEGORICAL_COLS].astype(str)
    if ohe is not None:
        X_cat = ohe.transform(X_cat)
    else:
        # fallback: simple one-hot creation (may differ from training)
        X_cat = pd.get_dummies(X_cat).values

    # concatenate and return
    X = np.hstack([X_num, X_cat])
    return X
