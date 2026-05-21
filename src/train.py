"""Train Random Forest and XGBoost attribution classifiers.

Usage:
    python -m src.train --data data/sample/attribution_sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from .preprocess import load_dataset, split


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def cross_validate(model, X: pd.DataFrame, y: pd.Series, label: str) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for metric in ("accuracy", "f1", "roc_auc"):
        s = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
        scores[metric] = float(np.mean(s))
        print(f"  {label} {metric:10s}: {scores[metric]:.4f}  (+/- {np.std(s):.4f})")
    return scores


def train(data_path: str | Path) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {data_path} ...")
    X, y = load_dataset(data_path)
    print(f"  rows={len(X)}  real={int(y.sum())}  spoof/decoy={int((1 - y).sum())}")
    print(f"  features: {list(X.columns)}")

    processed = X.copy()
    processed["label"] = y.values
    processed.to_csv(PROCESSED_DIR / "features.csv", index=False)

    X_train, X_test, y_train, y_test = split(X, y)

    rf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )

    print("\n[Random Forest] 5-fold cross-validation:")
    cross_validate(rf, X_train, y_train, "RF")
    rf.fit(X_train, y_train)
    joblib.dump(rf, MODELS_DIR / "rf.pkl")

    print("\n[XGBoost] 5-fold cross-validation:")
    cross_validate(xgb, X_train, y_train, "XGB")
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, MODELS_DIR / "xgb.pkl")

    test_split = X_test.copy()
    test_split["label"] = y_test.values
    test_split.to_csv(PROCESSED_DIR / "test_split.csv", index=False)

    print(f"\nSaved models to {MODELS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train attribution detectors.")
    parser.add_argument(
        "--data",
        default="data/sample/attribution_sample.csv",
        help="Path to a CSV file or directory of CSVs.",
    )
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
