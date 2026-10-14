"""
pipeline/flow.py
================
Prefect-orchestrated ML pipeline that automates the full workflow:

  1. Download / verify dataset
  2. Validate data quality
  3. Preprocess features
  4. Train & evaluate models
  5. Generate evaluation charts
  6. Run prediction demo
  7. Run monitoring checks

Each stage is a Prefect @task, orchestrated by a @flow.
If any critical task fails, the pipeline halts with a clear error.

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import traceback
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prefect import flow, task, get_run_logger


# ── Tasks ─────────────────────────────────────────────────────────────────────

@task(name="download_dataset", retries=2, retry_delay_seconds=10)
def task_download():
    """Download the UCI Student Performance dataset."""
    from src.data_ingestion import download_dataset
    csv_path = download_dataset()
    return str(csv_path)


@task(name="validate_data")
def task_validate(csv_path: str):
    """Run quality-validation checks on the raw dataset."""
    from src.data_validation import load_and_validate
    df = load_and_validate(csv_path)
    return df


@task(name="preprocess_data")
def task_preprocess(df):
    """Build preprocessing pipeline, fit on train, transform."""
    from src.preprocessing import prepare_data
    X_train, X_test, y_train, y_test, preprocessor, feature_names, report = prepare_data(df)
    return {
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "report": report,
    }


@task(name="train_and_evaluate")
def task_train(prep_data: dict):
    """Train all models, evaluate, select best, log to MLflow."""
    from src.train import train_and_evaluate
    outcome = train_and_evaluate(
        prep_data["X_train"],
        prep_data["X_test"],
        prep_data["y_train"],
        prep_data["y_test"],
        prep_data["feature_names"],
    )
    return outcome


@task(name="generate_charts")
def task_charts(outcome: dict):
    """Create performance visualisation charts."""
    from src.evaluate import generate_all_charts
    charts = generate_all_charts(
        outcome["results"],
        outcome["best_name"],
        outcome["y_test"],
    )
    return [str(c) for c in charts]


@task(name="prediction_demo")
def task_predict_demo():
    """Run a quick prediction to verify the saved model works."""
    from src.predict import predict_single, SAMPLE_STUDENT
    result = predict_single(SAMPLE_STUDENT)
    if result.get("error"):
        raise RuntimeError(f"Prediction demo failed: {result['messages']}")
    print(f"\n[DEMO] Predicted G3 = {result['prediction']} ({result['grade_label']})")
    return result


@task(name="monitoring")
def task_monitor():
    """Run model monitoring checks."""
    from monitoring.monitor import run_monitoring
    report = run_monitoring()
    if report["overall_status"] == "FAIL":
        raise RuntimeError("Monitoring detected a FAIL condition. See report.")
    return report


# ── Flow ──────────────────────────────────────────────────────────────────────

@flow(
    name="student_performance_pipeline",
    log_prints=True,
)
def student_performance_pipeline():
    """
    End-to-end ML pipeline for Student Performance Prediction.

    Workflow:
      Download → Validate → Preprocess → Train → Charts → Predict → Monitor
    """
    print("\n" + "=" * 60)
    print("  🚀 STUDENT PERFORMANCE MLOps PIPELINE")
    print("  Author: Abi (Abhishek S.)")
    print("  College: KGiSL Institute of Technology")
    print("=" * 60 + "\n")

    # Stage 1 ── Data
    csv_path  = task_download()
    df        = task_validate(csv_path)

    # Stage 2 ── Features
    prep_data = task_preprocess(df)

    # Stage 3 ── Models
    outcome   = task_train(prep_data)

    # Stage 4 ── Reporting
    charts    = task_charts(outcome)

    # Stage 5 ── Verification
    pred      = task_predict_demo()
    mon       = task_monitor()

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Best model : {outcome['best_name']}")
    print(f"  RMSE       : {outcome['best_metrics']['rmse']:.4f}")
    print(f"  R²         : {outcome['best_metrics']['r2']:.4f}")
    print(f"  Charts     : {len(charts)} saved")
    print(f"  Monitoring : {mon['overall_status']}")
    print("=" * 60 + "\n")

    return outcome


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        result = student_performance_pipeline()
        sys.exit(0)
    except Exception as exc:
        print(f"\n[PIPELINE FAILED] {exc}")
        traceback.print_exc()
        sys.exit(1)
