"""
src/preprocessing.py
====================
Builds and fits a sklearn preprocessing pipeline.

Steps
-----
1. Drop G1 and G2 (target-leakage prevention).
2. Remove duplicate rows.
3. Separate features (X) and target (y = G3).
4. Build ColumnTransformer:
     - Numerical  : StandardScaler
     - Categorical : OneHotEncoder (handle_unknown='ignore')
5. Fit on training data; transform train and test.
6. Persist the fitted preprocessor to disk.

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    DROP_COLUMNS, TARGET_COL, TEST_SIZE, RANDOM_STATE, PREPROCESSOR_PATH
)


# ── Column groups (derived from dataset schema) ──────────────────────────────
CATEGORICAL_COLS = [
    "school", "sex", "address", "famsize", "Pstatus",
    "Mjob", "Fjob", "reason", "guardian",
    "schoolsup", "famsup", "paid", "activities",
    "nursery", "higher", "internet", "romantic",
]

NUMERICAL_COLS = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "goout",
    "Dalc", "Walc", "health", "absences",
]


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer."""
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, NUMERICAL_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
    ], remainder="drop")

    return preprocessor


def prepare_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df : Raw DataFrame (validated, still contains G1, G2, G3)

    Returns
    -------
    X_train_t, X_test_t : np.ndarray  (transformed feature matrices)
    y_train, y_test     : pd.Series
    preprocessor        : fitted ColumnTransformer
    feature_names       : list[str]
    report              : dict  with preprocessing summary
    """
    # ── 1. Remove duplicates ─────────────────────────────────────────────────
    initial_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_dups = initial_rows - len(df)
    if removed_dups:
        print(f"[PREPROCESS] Removed {removed_dups} duplicate row(s).")

    # ── 2. Drop leakage columns ───────────────────────────────────────────────
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    print(f"[PREPROCESS] Dropped leakage columns: {DROP_COLUMNS}")
    print(f"[PREPROCESS] Remaining columns: {list(df.columns)}")

    # ── 3. Separate target ────────────────────────────────────────────────────
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not found after dropping {DROP_COLUMNS}.")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    print(f"[PREPROCESS] Features: {X.shape[1]}  |  Target: '{TARGET_COL}'  |  Samples: {len(y)}")

    # ── 4. Train / test split ─────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[PREPROCESS] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # ── 5. Fit preprocessor on training data only ─────────────────────────────
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # ── 6. Recover feature names ─────────────────────────────────────────────
    ohe_names = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["ohe"]
        .get_feature_names_out(CATEGORICAL_COLS)
        .tolist()
    )
    feature_names = NUMERICAL_COLS + ohe_names

    # ── 7. Save preprocessor ──────────────────────────────────────────────────
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"[PREPROCESS] Preprocessor saved → {PREPROCESSOR_PATH}")

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(df),
        "duplicate_rows_removed": removed_dups,
        "n_features": X_train_t.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "leakage_columns_dropped": DROP_COLUMNS,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
    }

    return X_train_t, X_test_t, y_train, y_test, preprocessor, feature_names, report


if __name__ == "__main__":
    from src.data_validation import load_and_validate
    df = load_and_validate()
    X_train_t, X_test_t, y_train, y_test, pre, fnames, rpt = prepare_data(df)
    print(f"[SUCCESS] Preprocessing done. Train shape: {X_train_t.shape}")
