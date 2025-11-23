"""
src/train_model.py

Optional script to (re)train a baseline model and save model + transformers into models/.
This uses a local CSV at data/full_dataset.csv by default — replace path if needed.

Note: This script is a clean example (RandomForest baseline).
You can replace the classifier with XGBoost/LightGBM/CatBoost as in your notebook.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from preprocess import fit_and_save_transformers, preprocess, TARGET_COL

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    # Replace this path with your dataset path if you add data/ to the repo
    data_path = "data/full_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Place a CSV at this path to run training.")
    df = pd.read_csv(data_path)

    # Fit and save transformers (creates scaler.pkl and ohe.pkl)
    fit_and_save_transformers(df, model_dir=MODEL_DIR)

    # Build X and y (preprocess uses saved transformers)
    X = preprocess(df, model_dir=MODEL_DIR)
    y = df[TARGET_COL].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    joblib.dump(clf, f"{MODEL_DIR}/model.pkl")
    print(f"[train] Model saved to {MODEL_DIR}/model.pkl")

if __name__ == "__main__":
    main()
