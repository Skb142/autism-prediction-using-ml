"""
src/inference.py

Simple CLI for running inference:
    python src/inference.py --input examples/sample_input.csv --output examples/predictions.csv

It expects a trained model in models/model.pkl and optional
transformers in models/scaler.pkl and models/ohe.pkl.
"""

import argparse
import os
import pandas as pd
import joblib
from preprocess import preprocess

def load_model(path="models/model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Place trained model as model.pkl in 'models/'.")
    return joblib.load(path)

def main(args):
    df = pd.read_csv(args.input)
    X = preprocess(df, model_dir=args.model_dir)
    model = load_model(args.model)

    # Get probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[:, 1]
    else:
        # fallback: use predict (may be binary labels)
        try:
            probs = model.predict(X)
        except Exception:
            probs = [None] * len(df)

    preds = model.predict(X)
    out = df.copy()
    out["ASD_pred"] = preds
    out["ASD_prob"] = probs
    out.to_csv(args.output, index=False)
    print(f"[inference] Wrote predictions to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on input CSV")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path for predictions")
    parser.add_argument("--model", default="models/model.pkl", help="Path to model file")
    parser.add_argument("--model_dir", default="models", help="Directory containing transformers")
    args = parser.parse_args()
    main(args)
