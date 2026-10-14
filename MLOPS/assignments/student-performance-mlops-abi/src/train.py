"""
src/train.py
============
Trains three regression models, evaluates them, selects the best one,
saves it to disk, and logs everything to MLflow.

Models
------
  1. Linear Regression
  2. Random Forest Regressor
  3. Gradient Boosting Regressor

Metrics
-------
  MAE, MSE, RMSE, R²

Best-model criterion
--------------------
  Lowest RMSE (configurable in configs/params.yaml  →  selection.metric)

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import json
import math
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    MODEL_PARAMS, SELECTION_METRIC, SELECTION_DIRECTION,
    BEST_MODEL_PATH, MODEL_DIR,
    METRICS_JSON, METRICS_CSV,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT,
    RANDOM_STATE,
)


# ── Model registry ────────────────────────────────────────────────────────────
def _build_models() -> dict:
    lr_p  = MODEL_PARAMS["linear_regression"]
    rf_p  = MODEL_PARAMS["random_forest"]
    gb_p  = MODEL_PARAMS["gradient_boosting"]

    return {
        "Linear Regression": LinearRegression(
            fit_intercept=lr_p["fit_intercept"]
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators  = rf_p["n_estimators"],
            max_depth      = rf_p["max_depth"],
            min_samples_split = rf_p["min_samples_split"],
            random_state   = rf_p["random_state"],
            n_jobs         = -1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators  = gb_p["n_estimators"],
            learning_rate = gb_p["learning_rate"],
            max_depth     = gb_p["max_depth"],
            random_state  = gb_p["random_state"],
        ),
    }


# ── Metrics helper ────────────────────────────────────────────────────────────
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(math.sqrt(mse))
    r2   = float(r2_score(y_true, y_pred))
    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# ── Core training function ────────────────────────────────────────────────────
def train_and_evaluate(
    X_train, X_test, y_train, y_test, feature_names=None
) -> dict:
    """
    Train all models, evaluate on test set, log to MLflow.

    Returns
    -------
    dict  with keys:
        "results"    : {model_name: {metrics…, "model": fitted_model}}
        "best_name"  : str
        "best_model" : fitted model object
        "best_metrics": dict
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    models   = _build_models()
    results  = {}
    all_runs = []

    print("\n" + "=" * 60)
    print("  MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n[TRAIN] Training: {name} ...")

        with mlflow.start_run(run_name=name):
            # ── Log hyperparameters ──────────────────────────────────────────
            params = model.get_params()
            mlflow.log_params({k: str(v) for k, v in params.items()})

            # ── Fit ──────────────────────────────────────────────────────────
            model.fit(X_train, y_train)

            # ── Predict ──────────────────────────────────────────────────────
            y_pred  = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)

            # ── Log metrics ──────────────────────────────────────────────────
            mlflow.log_metrics(metrics)

            # ── Save model artifact to MLflow ─────────────────────────────────
            mlflow.sklearn.log_model(model, artifact_path="model")

            run_id = mlflow.active_run().info.run_id

        results[name] = {**metrics, "model": model, "y_pred": y_pred, "run_id": run_id}

        print(f"  MAE  : {metrics['mae']:.4f}")
        print(f"  MSE  : {metrics['mse']:.4f}")
        print(f"  RMSE : {metrics['rmse']:.4f}")
        print(f"  R²   : {metrics['r2']:.4f}")

        all_runs.append({
            "Model": name,
            "MAE":   round(metrics["mae"],  4),
            "MSE":   round(metrics["mse"],  4),
            "RMSE":  round(metrics["rmse"], 4),
            "R2":    round(metrics["r2"],   4),
            "MLflow_Run_ID": run_id,
        })

    # ── Select best model ─────────────────────────────────────────────────────
    reverse = (SELECTION_DIRECTION == "maximize")
    best_name = sorted(
        results.keys(),
        key=lambda n: results[n][SELECTION_METRIC],
        reverse=reverse
    )[0]
    best_model   = results[best_name]["model"]
    best_metrics = {k: results[best_name][k] for k in ("mae","mse","rmse","r2")}

    print("\n" + "=" * 60)
    print(f"  ✅ BEST MODEL: {best_name}")
    print(f"     Selection metric: {SELECTION_METRIC.upper()} = {best_metrics[SELECTION_METRIC]:.4f}")
    print("=" * 60)

    # ── Save best model ───────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    print(f"[SAVE] Best model saved → {BEST_MODEL_PATH}")

    # ── Save metrics ──────────────────────────────────────────────────────────
    _save_metrics(results, best_name, all_runs)

    return {
        "results":      results,
        "best_name":    best_name,
        "best_model":   best_model,
        "best_metrics": best_metrics,
        "y_test":       y_test,
        "all_runs":     all_runs,
    }


def _save_metrics(results: dict, best_name: str, all_runs: list) -> None:
    """Persist metrics to JSON and CSV."""
    # JSON
    json_payload = {
        "best_model": best_name,
        "selection_metric": SELECTION_METRIC,
        "models": {
            name: {
                k: v for k, v in data.items()
                if k not in ("model", "y_pred", "run_id")
            }
            for name, data in results.items()
        }
    }
    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, "w") as fh:
        json.dump(json_payload, fh, indent=2)
    print(f"[SAVE] Metrics JSON → {METRICS_JSON}")

    # CSV
    df_csv = pd.DataFrame(all_runs)
    df_csv.to_csv(METRICS_CSV, index=False)
    print(f"[SAVE] Metrics CSV  → {METRICS_CSV}")


if __name__ == "__main__":
    from src.data_ingestion  import download_dataset
    from src.data_validation import load_and_validate
    from src.preprocessing   import prepare_data

    download_dataset()
    df = load_and_validate()
    X_tr, X_te, y_tr, y_te, pre, fnames, _ = prepare_data(df)
    outcome = train_and_evaluate(X_tr, X_te, y_tr, y_te, fnames)
    print("\n[DONE] Training complete.")
