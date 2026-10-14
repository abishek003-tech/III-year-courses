"""
src/data_ingestion.py
=====================
Downloads the UCI Student Performance dataset and extracts student-mat.csv.
Idempotent: skips download if the CSV already exists.

Dataset source:
  P. Cortez and A. Silva (2008).
  "Using Data Mining to Predict Secondary School Student Performance."
  UCI Machine Learning Repository.
  URL: https://archive.ics.uci.edu/ml/datasets/Student+Performance

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import zipfile
import requests
import sys
import os
from pathlib import Path

# Allow running this file standalone or as part of the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_URL, RAW_ZIP, CSV_FILE


def download_dataset(force: bool = False) -> Path:
    """
    Download student.zip from UCI if the CSV does not already exist.

    Parameters
    ----------
    force : bool
        If True, re-download even if the file exists.

    Returns
    -------
    Path
        Absolute path to student-mat.csv.
    """
    if CSV_FILE.exists() and not force:
        print(f"[INFO] Dataset already present at {CSV_FILE}. Skipping download.")
        return CSV_FILE

    # ── Create data directory ────────────────────────────────────────────────
    RAW_ZIP.parent.mkdir(parents=True, exist_ok=True)

    # ── Download ZIP ─────────────────────────────────────────────────────────
    print(f"[INFO] Downloading dataset from:\n       {DATA_URL}")
    try:
        response = requests.get(DATA_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ERROR] Download failed: {exc}")
        print("[INFO]  Falling back to local copy if available...")
        _use_local_fallback()
        if not CSV_FILE.exists():
            raise RuntimeError(
                "Dataset download failed and no local copy is available.\n"
                "Please manually place student-mat.csv inside the data/ folder."
            ) from exc
        return CSV_FILE

    with open(RAW_ZIP, "wb") as fh:
        fh.write(response.content)
    print(f"[INFO] Saved ZIP to {RAW_ZIP} ({len(response.content) / 1024:.1f} KB)")

    # ── Extract CSV ──────────────────────────────────────────────────────────
    _extract_csv(RAW_ZIP)
    return CSV_FILE


def _extract_csv(zip_path: Path) -> None:
    """Extract student-mat.csv from the ZIP archive."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        target  = next((m for m in members if "student-mat.csv" in m), None)
        if target is None:
            raise FileNotFoundError(
                f"student-mat.csv not found inside {zip_path}. "
                f"Archive contains: {members}"
            )
        # Extract just the CSV, flatten any sub-folder path
        data = zf.read(target)
        with open(CSV_FILE, "wb") as fh:
            fh.write(data)
    print(f"[INFO] Extracted student-mat.csv to {CSV_FILE}")


def _use_local_fallback() -> None:
    """Check if a zip was manually placed in data/ and extract from it."""
    for candidate in RAW_ZIP.parent.glob("*.zip"):
        try:
            _extract_csv(candidate)
            print(f"[INFO] Used local fallback: {candidate}")
            return
        except Exception:
            continue


if __name__ == "__main__":
    csv = download_dataset()
    print(f"[SUCCESS] Dataset ready at: {csv}")
