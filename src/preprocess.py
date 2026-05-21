"""Dataset loading and cleaning for the Network Traffic Attribution project.

Handles two layouts:

  1. CSV files with raw flow records exported from packet captures (any CSV
     with the numeric attribution features listed in features.py plus a
     binary label column).
  2. Pre-extracted feature CSVs from CIC-DDoS2019, where each row is one
     flow with numeric statistics and a label that we collapse to:
        1 = real attacker source
        0 = spoofed / decoy traffic

The single public entry point is `load_dataset`, which returns a tuple of
(features_df, labels_series).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split


_LABEL_CANDIDATES = ("label", "Label", "class", "Class", "is_real", "Source", "Type")

# Strings that map to the positive class (real attacker = 1).
_POSITIVE_TOKENS = {
    "1",
    "real",
    "real_attacker",
    "attacker",
    "true_source",
    "true",
    "genuine",
}

# Strings that map to the negative class (spoof / decoy = 0).
_NEGATIVE_TOKENS = {
    "0",
    "spoof",
    "spoofed",
    "decoy",
    "fake",
    "false_source",
    "false",
}


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _normalize_labels(series: pd.Series) -> pd.Series:
    """Map label encodings to {0, 1} (real = 1, spoof/decoy = 0)."""
    if pd.api.types.is_numeric_dtype(series):
        return (series > 0).astype(int)
    lower = series.astype(str).str.strip().str.lower()
    out = lower.where(~lower.isin(_NEGATIVE_TOKENS), "0")
    out = out.where(~lower.isin(_POSITIVE_TOKENS), "1")
    return out.where(out.isin(("0", "1")), "0").astype(int)


def load_csv_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        raise FileNotFoundError("No CSV files were provided.")
    return pd.concat(frames, ignore_index=True)


def load_dataset(
    path: str | Path,
    label_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a network-traffic dataset from a CSV file or a directory of CSVs.

    Returns (features, labels) with labels in {0, 1}.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.csv"))
        df = load_csv_files(files)
    else:
        df = pd.read_csv(path)

    df = df.dropna(how="all").reset_index(drop=True)

    label_col = label_col or _find_column(df, _LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(
            f"Could not find a label column. Looked for {_LABEL_CANDIDATES}. "
            f"Columns present: {list(df.columns)}"
        )
    labels = _normalize_labels(df[label_col])

    feature_df = df.drop(columns=[label_col])
    data = feature_df.select_dtypes(include="number").copy()
    if data.empty:
        raise ValueError(
            "No numeric features found. "
            f"Columns: {list(df.columns)}"
        )

    keep = ~data.duplicated()
    data = data[keep].reset_index(drop=True)
    labels = labels[keep].reset_index(drop=True)

    # Median-impute any NaNs in feature columns.
    numeric_cols = data.select_dtypes(include="number").columns
    if len(numeric_cols):
        data[numeric_cols] = data[numeric_cols].fillna(
            data[numeric_cols].median(numeric_only=True)
        )

    return data, labels


def split(
    data: pd.DataFrame,
    labels: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified 80/20 train/test split."""
    return train_test_split(
        data,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample/attribution_sample.csv")
    X, y = load_dataset(target)
    print(f"Loaded {len(X)} rows | real={int(y.sum())} | spoof/decoy={int((1 - y).sum())}")
    print(X.head())
