"""
src/data_validation.py
======================
Validates the raw dataset before any transformation is applied.
Raises ValueError when critical checks fail so the pipeline halts early.

Checks performed
----------------
1. File existence
2. Row count (minimum 100 rows expected)
3. Required column presence
4. Missing-value report
5. Duplicate-row count
6. Target variable (G3) range  [0, 20]
7. Numeric column range sanity
8. Categorical column allowed-value check

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CSV_FILE, CSV_SEPARATOR, TARGET_COL, DROP_COLUMNS

# ── Expected schema ──────────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
    "traveltime", "studytime", "failures", "schoolsup", "famsup",
    "paid", "activities", "nursery", "higher", "internet", "romantic",
    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",
    "G1", "G2", "G3"
]

CATEGORICAL_ALLOWED: dict = {
    "school":    {"GP", "MS"},
    "sex":       {"F", "M"},
    "address":   {"U", "R"},
    "famsize":   {"GT3", "LE3"},
    "Pstatus":   {"T", "A"},
    "schoolsup": {"yes", "no"},
    "famsup":    {"yes", "no"},
    "paid":      {"yes", "no"},
    "activities":{"yes", "no"},
    "nursery":   {"yes", "no"},
    "higher":    {"yes", "no"},
    "internet":  {"yes", "no"},
    "romantic":  {"yes", "no"},
}

NUMERIC_RANGES: dict = {
    "age":       (15, 22),
    "Medu":      (0, 4),
    "Fedu":      (0, 4),
    "traveltime":(1, 4),
    "studytime": (1, 4),
    "failures":  (0, 4),
    "famrel":    (1, 5),
    "freetime":  (1, 5),
    "goout":     (1, 5),
    "Dalc":      (1, 5),
    "Walc":      (1, 5),
    "health":    (1, 5),
    "absences":  (0, 93),
    "G1":        (0, 20),
    "G2":        (0, 20),
    "G3":        (0, 20),
}


def validate(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run all validation checks on the raw dataframe.

    Parameters
    ----------
    df      : Raw pandas DataFrame loaded from student-mat.csv
    verbose : Print a summary to stdout

    Returns
    -------
    dict
        Validation report. Raises ValueError on critical failures.
    """
    report: dict = {}

    # ── 1. Row count ─────────────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    report["rows"] = n_rows
    report["columns"] = n_cols
    if n_rows < 100:
        raise ValueError(f"[VALIDATION FAIL] Dataset has only {n_rows} rows. Expected at least 100.")

    # ── 2. Required columns ───────────────────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report["missing_required_columns"] = missing_cols
    if missing_cols:
        raise ValueError(f"[VALIDATION FAIL] Missing required columns: {missing_cols}")

    # ── 3. Missing values ─────────────────────────────────────────────────────
    null_counts = df.isnull().sum()
    report["missing_values"] = null_counts[null_counts > 0].to_dict()
    if null_counts.sum() > 0:
        print(f"[WARNING] Missing values detected:\n{null_counts[null_counts > 0]}")
    else:
        report["missing_values_status"] = "CLEAN — no missing values"

    # ── 4. Duplicate rows ─────────────────────────────────────────────────────
    n_dups = int(df.duplicated().sum())
    report["duplicate_rows"] = n_dups
    if n_dups > 0:
        print(f"[WARNING] {n_dups} duplicate row(s) found. They will be removed during preprocessing.")

    # ── 5. Target variable range ──────────────────────────────────────────────
    g3_min, g3_max = df[TARGET_COL].min(), df[TARGET_COL].max()
    report["G3_range"] = {"min": int(g3_min), "max": int(g3_max)}
    if not (0 <= g3_min and g3_max <= 20):
        raise ValueError(f"[VALIDATION FAIL] G3 values out of [0,20] range: min={g3_min}, max={g3_max}")

    # ── 6. Numeric range checks ───────────────────────────────────────────────
    range_violations: dict = {}
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if not bad.empty:
            range_violations[col] = {"count": len(bad), "allowed": [lo, hi]}
    report["numeric_range_violations"] = range_violations
    if range_violations:
        print(f"[WARNING] Numeric range violations: {range_violations}")

    # ── 7. Categorical domain checks ─────────────────────────────────────────
    cat_violations: dict = {}
    for col, allowed in CATEGORICAL_ALLOWED.items():
        if col not in df.columns:
            continue
        bad_vals = set(df[col].unique()) - allowed
        if bad_vals:
            cat_violations[col] = list(bad_vals)
    report["categorical_violations"] = cat_violations
    if cat_violations:
        print(f"[WARNING] Unexpected categorical values: {cat_violations}")

    # ── 8. G3 = 0 students (informational) ────────────────────────────────────
    zero_grade = int((df[TARGET_COL] == 0).sum())
    report["students_with_G3_zero"] = zero_grade

    # ── Summary ───────────────────────────────────────────────────────────────
    if verbose:
        _print_report(report, df)

    report["status"] = "PASSED"
    return report


def _print_report(report: dict, df: pd.DataFrame) -> None:
    print("\n" + "=" * 55)
    print("     DATA VALIDATION REPORT")
    print("=" * 55)
    print(f"  Rows           : {report['rows']}")
    print(f"  Columns        : {report['columns']}")
    print(f"  Missing values : {report.get('missing_values', 'None')}")
    print(f"  Duplicate rows : {report['duplicate_rows']}")
    print(f"  G3 range       : {report['G3_range']}")
    print(f"  G3 = 0 students: {report['students_with_G3_zero']}")
    print(f"  Range violations: {report['numeric_range_violations'] or 'None'}")
    print(f"  Cat. violations : {report['categorical_violations'] or 'None'}")
    print("=" * 55)
    print(f"\n  Dataset dtypes summary:")
    print(f"    Numeric  : {len(df.select_dtypes(include='number').columns)} columns")
    print(f"    Object   : {len(df.select_dtypes(include='object').columns)} columns")
    print("=" * 55)
    print("  STATUS: ALL CRITICAL CHECKS PASSED ✅")
    print("=" * 55 + "\n")


def load_and_validate(csv_path=None) -> pd.DataFrame:
    """Load the raw CSV and run validation. Returns the raw DataFrame."""
    path = Path(csv_path) if csv_path else CSV_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"[ERROR] Dataset not found at {path}.\n"
            "Run: python src/data_ingestion.py  to download it."
        )
    df = pd.read_csv(path, sep=CSV_SEPARATOR)
    validate(df, verbose=True)
    return df


if __name__ == "__main__":
    df = load_and_validate()
    print(f"[SUCCESS] Validation complete. Shape: {df.shape}")
