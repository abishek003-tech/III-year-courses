"""
monitoring/monitor.py
=====================
Basic model monitoring:
  1. Loads the best model and preprocessor.
  2. Runs predictions on a batch of sample inputs.
  3. Compares predicted distribution against training-time metrics.
  4. Flags potential data-drift or model-degradation issues.

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    BEST_MODEL_PATH, PREPROCESSOR_PATH, METRICS_JSON,
    CSV_FILE, CSV_SEPARATOR, TARGET_COL
)
from src.preprocessing import CATEGORICAL_COLS, NUMERICAL_COLS, prepare_data
from src.train import compute_metrics
from src.config import DROP_COLUMNS


def run_monitoring() -> dict:
    """
    Execute monitoring checks and return a report.

    Checks
    ------
    1. Model file integrity — can we load the model?
    2. Preprocessor file integrity — can we load the preprocessor?
    3. Prediction sanity — are predictions within [0, 20]?
    4. Performance drift — compare current metrics to saved baseline.
    5. Latency — is inference reasonably fast?

    Returns
    -------
    dict  with monitoring report and pass/fail status.
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "checks": {},
        "overall_status": "PASS",
    }

    # ── 1. Model loading ──────────────────────────────────────────────────────
    try:
        model = joblib.load(BEST_MODEL_PATH)
        report["checks"]["model_load"] = {"status": "PASS", "path": str(BEST_MODEL_PATH)}
    except Exception as exc:
        report["checks"]["model_load"] = {"status": "FAIL", "error": str(exc)}
        report["overall_status"] = "FAIL"
        print(f"[MONITOR FAIL] Cannot load model: {exc}")
        return report

    # ── 2. Preprocessor loading ───────────────────────────────────────────────
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        report["checks"]["preprocessor_load"] = {"status": "PASS", "path": str(PREPROCESSOR_PATH)}
    except Exception as exc:
        report["checks"]["preprocessor_load"] = {"status": "FAIL", "error": str(exc)}
        report["overall_status"] = "FAIL"
        print(f"[MONITOR FAIL] Cannot load preprocessor: {exc}")
        return report

    # ── 3. Prediction sanity on full dataset ──────────────────────────────────
    try:
        df = pd.read_csv(CSV_FILE, sep=CSV_SEPARATOR)
        df = df.drop(columns=DROP_COLUMNS, errors="ignore")
        X = df.drop(columns=[TARGET_COL])
        y_true = df[TARGET_COL].values

        feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
        X = X[feature_cols]

        start = time.time()
        X_transformed = preprocessor.transform(X)
        y_pred = model.predict(X_transformed)
        latency_ms = (time.time() - start) * 1000

        # Check range
        out_of_range = int(((y_pred < -2) | (y_pred > 22)).sum())
        report["checks"]["prediction_range"] = {
            "status": "PASS" if out_of_range == 0 else "WARNING",
            "out_of_range_count": out_of_range,
            "pred_min": round(float(y_pred.min()), 4),
            "pred_max": round(float(y_pred.max()), 4),
            "pred_mean": round(float(y_pred.mean()), 4),
        }
        if out_of_range > 0:
            print(f"[MONITOR WARNING] {out_of_range} prediction(s) outside expected range.")

        # Latency
        report["checks"]["latency"] = {
            "status": "PASS" if latency_ms < 5000 else "WARNING",
            "total_ms": round(latency_ms, 2),
            "per_sample_ms": round(latency_ms / len(X), 4),
            "n_samples": len(X),
        }

    except Exception as exc:
        report["checks"]["prediction_sanity"] = {"status": "FAIL", "error": str(exc)}
        report["overall_status"] = "FAIL"
        print(f"[MONITOR FAIL] Prediction sanity check failed: {exc}")
        return report

    # ── 4. Performance drift check ────────────────────────────────────────────
    try:
        current_metrics = compute_metrics(y_true, y_pred)

        if METRICS_JSON.exists():
            with open(METRICS_JSON) as fh:
                saved = json.load(fh)
            best_name = saved.get("best_model", "")
            baseline  = saved.get("models", {}).get(best_name, {})

            if baseline:
                rmse_drift = abs(current_metrics["rmse"] - baseline.get("rmse", 0))
                r2_drift   = abs(current_metrics["r2"]   - baseline.get("r2", 0))
                drift_ok   = rmse_drift < 1.0 and r2_drift < 0.1

                report["checks"]["performance_drift"] = {
                    "status": "PASS" if drift_ok else "WARNING",
                    "baseline_rmse": baseline.get("rmse"),
                    "current_rmse": round(current_metrics["rmse"], 4),
                    "rmse_drift": round(rmse_drift, 4),
                    "baseline_r2": baseline.get("r2"),
                    "current_r2": round(current_metrics["r2"], 4),
                    "r2_drift": round(r2_drift, 4),
                }
                if not drift_ok:
                    report["overall_status"] = "WARNING"
                    print(f"[MONITOR WARNING] Performance drift detected!")

        report["checks"]["current_metrics"] = {
            k: round(v, 4) for k, v in current_metrics.items()
        }

    except Exception as exc:
        report["checks"]["performance_drift"] = {"status": "FAIL", "error": str(exc)}

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("     MODEL MONITORING REPORT")
    print("=" * 55)
    for check_name, check_data in report["checks"].items():
        status = check_data.get("status", "INFO")
        icon   = "✅" if status == "PASS" else ("⚠️" if status == "WARNING" else "❌")
        print(f"  {icon} {check_name}: {status}")
    print(f"\n  Overall: {report['overall_status']}")
    print("=" * 55)

    # Save report
    report_path = Path(__file__).resolve().parent.parent / "reports" / "monitoring_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[MONITOR] Report saved → {report_path}")

    return report


if __name__ == "__main__":
    run_monitoring()
