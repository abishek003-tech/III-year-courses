"""
src/predict.py
==============
Loads the saved best model + preprocessor and makes predictions
on new student data.  Includes input validation and error messages.

Can be used standalone (CLI) or imported by app.py / tests.

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH
from src.preprocessing import CATEGORICAL_COLS, NUMERICAL_COLS

# ── Valid value ranges (used for input validation) ────────────────────────────
VALID_RANGES = {
    "school": ["GP", "MS"],
    "sex": ["F", "M"],
    "age": (15, 22),
    "address": ["U", "R"],
    "famsize": ["GT3", "LE3"],
    "Pstatus": ["T", "A"],
    "Medu": (0, 4),
    "Fedu": (0, 4),
    "Mjob": ["teacher", "health", "services", "at_home", "other"],
    "Fjob": ["teacher", "health", "services", "at_home", "other"],
    "reason": ["home", "reputation", "course", "other"],
    "guardian": ["mother", "father", "other"],
    "traveltime": (1, 4),
    "studytime": (1, 4),
    "failures": (0, 4),
    "schoolsup": ["yes", "no"],
    "famsup": ["yes", "no"],
    "paid": ["yes", "no"],
    "activities": ["yes", "no"],
    "nursery": ["yes", "no"],
    "higher": ["yes", "no"],
    "internet": ["yes", "no"],
    "romantic": ["yes", "no"],
    "famrel": (1, 5),
    "freetime": (1, 5),
    "goout": (1, 5),
    "Dalc": (1, 5),
    "Walc": (1, 5),
    "health": (1, 5),
    "absences": (0, 93),
}

ALL_FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS


def validate_input(data: dict) -> list[str]:
    """
    Validate a single student record.

    Returns a list of error messages (empty if valid).
    """
    errors: list[str] = []

    # Check required fields
    for col in ALL_FEATURE_COLS:
        if col not in data:
            errors.append(f"Missing required field: '{col}'")

    # Check value ranges
    for col, valid in VALID_RANGES.items():
        if col not in data:
            continue
        val = data[col]
        if isinstance(valid, list):
            if val not in valid:
                errors.append(f"'{col}' must be one of {valid}, got '{val}'")
        elif isinstance(valid, tuple):
            lo, hi = valid
            try:
                num = float(val)
                if not (lo <= num <= hi):
                    errors.append(f"'{col}' must be between {lo} and {hi}, got {num}")
            except (ValueError, TypeError):
                errors.append(f"'{col}' must be a number between {lo} and {hi}, got '{val}'")

    return errors


def load_model_and_preprocessor():
    """Load saved artifacts. Raises FileNotFoundError with clear messages."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model not found at {BEST_MODEL_PATH}.\n"
            "Run the training pipeline first:  python pipeline/flow.py"
        )
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {PREPROCESSOR_PATH}.\n"
            "Run the training pipeline first:  python pipeline/flow.py"
        )

    model         = joblib.load(BEST_MODEL_PATH)
    preprocessor  = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def predict_single(data: dict) -> dict:
    """
    Predict the G3 grade for a single student record.

    Parameters
    ----------
    data : dict  with feature names as keys

    Returns
    -------
    dict  {"prediction": float, "grade_label": str}
    """
    # Validate
    errors = validate_input(data)
    if errors:
        return {"error": True, "messages": errors}

    # Load
    model, preprocessor = load_model_and_preprocessor()

    # Build DataFrame (column order must match training)
    df = pd.DataFrame([data])[ALL_FEATURE_COLS]

    # Transform & predict
    X_transformed = preprocessor.transform(df)
    pred = float(model.predict(X_transformed)[0])

    # Clip to valid range [0, 20]
    pred = float(np.clip(pred, 0, 20))

    return {
        "prediction": round(pred, 2),
        "grade_label": _grade_label(pred),
        "error": False,
    }


def _grade_label(score: float) -> str:
    """Convert numeric G3 to a human-readable label."""
    if score >= 16:
        return "Excellent (16-20)"
    elif score >= 14:
        return "Good (14-15)"
    elif score >= 12:
        return "Satisfactory (12-13)"
    elif score >= 10:
        return "Sufficient (10-11)"
    else:
        return "Fail (0-9)"


# ── CLI demo ──────────────────────────────────────────────────────────────────
SAMPLE_STUDENT = {
    "school": "GP", "sex": "F", "age": 17, "address": "U",
    "famsize": "GT3", "Pstatus": "A", "Medu": 4, "Fedu": 4,
    "Mjob": "teacher", "Fjob": "health", "reason": "course",
    "guardian": "mother", "traveltime": 2, "studytime": 2,
    "failures": 0, "schoolsup": "yes", "famsup": "yes",
    "paid": "no", "activities": "no", "nursery": "yes",
    "higher": "yes", "internet": "yes", "romantic": "no",
    "famrel": 4, "freetime": 3, "goout": 4,
    "Dalc": 1, "Walc": 1, "health": 3, "absences": 4,
}


if __name__ == "__main__":
    print("=" * 50)
    print("  STUDENT PERFORMANCE PREDICTION — Demo")
    print("=" * 50)

    result = predict_single(SAMPLE_STUDENT)

    if result.get("error"):
        print("[ERROR] Validation failed:")
        for msg in result["messages"]:
            print(f"  - {msg}")
    else:
        print(f"\n  Predicted G3 Score : {result['prediction']}")
        print(f"  Grade Label       : {result['grade_label']}")
    print("=" * 50)
