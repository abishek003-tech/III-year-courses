"""
tests/test_pipeline.py
======================
Automated tests for the Student Performance MLOps system.

Covers: config, data validation, preprocessing, training, prediction,
        and monitoring — all with real execution against the actual dataset.

Run with:  pytest tests/ -v

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
"""

import sys
import json
import math
import pytest
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
class TestConfig:
    def test_params_yaml_loads(self):
        from src.config import load_params
        cfg = load_params()
        assert "data" in cfg
        assert "features" in cfg
        assert "models" in cfg

    def test_project_paths_exist(self):
        from src.config import CHARTS_DIR, MODEL_DIR
        assert MODEL_DIR.parent.exists()


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
class TestDataIngestion:
    def test_download_returns_path(self):
        from src.data_ingestion import download_dataset
        path = download_dataset()
        assert Path(path).exists()
        assert Path(path).suffix == ".csv"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
class TestDataValidation:
    @pytest.fixture
    def raw_df(self):
        from src.config import CSV_FILE, CSV_SEPARATOR
        return pd.read_csv(CSV_FILE, sep=CSV_SEPARATOR)

    def test_validate_passes(self, raw_df):
        from src.data_validation import validate
        report = validate(raw_df, verbose=False)
        assert report["status"] == "PASSED"

    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_row_count(self, raw_df):
        assert len(raw_df) >= 100

    def test_g3_range(self, raw_df):
        assert raw_df["G3"].min() >= 0
        assert raw_df["G3"].max() <= 20

    def test_required_columns(self, raw_df):
        for col in ["G1", "G2", "G3", "school", "sex", "age"]:
            assert col in raw_df.columns, f"Missing: {col}"

    def test_invalid_df_raises(self):
        from src.data_validation import validate
        tiny = pd.DataFrame({"G3": [5]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate(tiny, verbose=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
class TestPreprocessing:
    @pytest.fixture
    def prepared(self):
        from src.config import CSV_FILE, CSV_SEPARATOR
        from src.preprocessing import prepare_data
        df = pd.read_csv(CSV_FILE, sep=CSV_SEPARATOR)
        return prepare_data(df)

    def test_returns_seven_items(self, prepared):
        assert len(prepared) == 7

    def test_g1_g2_not_in_features(self, prepared):
        _, _, _, _, _, feature_names, _ = prepared
        for leak in ("G1", "G2"):
            assert leak not in feature_names

    def test_shapes_consistent(self, prepared):
        X_tr, X_te, y_tr, y_te, _, _, _ = prepared
        assert X_tr.shape[0] == len(y_tr)
        assert X_te.shape[0] == len(y_te)
        assert X_tr.shape[1] == X_te.shape[1]

    def test_preprocessor_saved(self):
        from src.config import PREPROCESSOR_PATH
        assert PREPROCESSOR_PATH.exists()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
class TestTraining:
    def test_compute_metrics(self):
        from src.train import compute_metrics
        y_true = np.array([10, 12, 14, 16, 18])
        y_pred = np.array([10.5, 11.5, 14.5, 15.5, 17.5])
        m = compute_metrics(y_true, y_pred)
        assert "mae" in m and "mse" in m and "rmse" in m and "r2" in m
        assert m["rmse"] == pytest.approx(math.sqrt(m["mse"]), abs=1e-6)
        assert 0 < m["r2"] <= 1

    def test_metrics_json_exists(self):
        from src.config import METRICS_JSON
        assert METRICS_JSON.exists()
        with open(METRICS_JSON) as fh:
            data = json.load(fh)
        assert "best_model" in data
        assert "models" in data

    def test_three_models_evaluated(self):
        from src.config import METRICS_JSON
        with open(METRICS_JSON) as fh:
            data = json.load(fh)
        assert len(data["models"]) == 3

    def test_best_model_saved(self):
        from src.config import BEST_MODEL_PATH
        assert BEST_MODEL_PATH.exists()
        model = joblib.load(BEST_MODEL_PATH)
        assert hasattr(model, "predict")


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
class TestPrediction:
    def test_valid_prediction(self):
        from src.predict import predict_single, SAMPLE_STUDENT
        result = predict_single(SAMPLE_STUDENT)
        assert result["error"] is False
        assert 0 <= result["prediction"] <= 20
        assert "grade_label" in result

    def test_missing_field_returns_error(self):
        from src.predict import predict_single
        result = predict_single({"school": "GP"})  # missing fields
        assert result["error"] is True
        assert len(result["messages"]) > 0

    def test_invalid_value_returns_error(self):
        from src.predict import predict_single, SAMPLE_STUDENT
        bad = SAMPLE_STUDENT.copy()
        bad["school"] = "INVALID"
        result = predict_single(bad)
        assert result["error"] is True

    def test_prediction_is_clipped(self):
        from src.predict import predict_single, SAMPLE_STUDENT
        result = predict_single(SAMPLE_STUDENT)
        assert 0 <= result["prediction"] <= 20


# ═══════════════════════════════════════════════════════════════════════════════
#  MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
class TestMonitoring:
    def test_monitor_runs(self):
        from monitoring.monitor import run_monitoring
        report = run_monitoring()
        assert report["overall_status"] in ("PASS", "WARNING")
        assert "checks" in report

    def test_model_load_check(self):
        from monitoring.monitor import run_monitoring
        report = run_monitoring()
        assert report["checks"]["model_load"]["status"] == "PASS"
