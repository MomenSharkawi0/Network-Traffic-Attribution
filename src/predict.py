"""Score a CSV with a pretrained attribution classifier.

Usage:
    python -m src.predict --model models/rf.pkl
    python -m src.predict --model models/xgb.pkl --data my.csv --out preds.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

from .preprocess import load_dataset


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "sample" / "attribution_sample.csv"


def _align_to_model(X: pd.DataFrame, model) -> pd.DataFrame:
    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        return X
    missing = [c for c in expected if c not in X.columns]
    if missing:
        raise ValueError(
            f"Input is missing features the model expects: {missing}"
        )
    return X[list(expected)]


def predict(model_path: str | Path, data_path: str | Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    X, y_true = load_dataset(data_path)
    X = _align_to_model(X, model)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    return pd.DataFrame(
        {
            "label_true": y_true.values,
            "label_pred": y_pred.astype(int),
            "prob_real": y_prob,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score data with a pretrained model.")
    parser.add_argument("--model", default=str(ROOT / "models" / "rf.pkl"))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = predict(args.model, args.data)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Wrote {len(df):,} predictions to {args.out}")

    print(f"\nUsing model: {args.model}")
    print(f"Data:        {args.data}")
    print(f"Rows scored: {len(df):,}\n")

    y_true = df["label_true"].values
    y_pred = df["label_pred"].values
    y_prob = df["prob_real"].values
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1       : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    if len(np.unique(y_true)) > 1:
        print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["spoof/decoy", "real"]))


if __name__ == "__main__":
    main()
