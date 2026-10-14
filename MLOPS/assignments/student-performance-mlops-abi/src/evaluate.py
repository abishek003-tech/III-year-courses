"""
src/evaluate.py
===============
Generates performance visualisation charts from trained model results.

Charts produced
---------------
1. Model Comparison bar chart  (MAE, MSE, RMSE, R²)
2. Actual vs Predicted scatter (for each model)
3. Residual / Prediction Error plot (best model)

Author : Abi (Abhishek S.)
College: KGiSL Institute of Technology
Dept   : Artificial Intelligence and Machine Learning
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works headless on Windows)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import CHARTS_DIR


def generate_all_charts(results: dict, best_name: str, y_test) -> list:
    """
    Generate and save all evaluation charts.

    Parameters
    ----------
    results   : dict  {model_name: {mae, mse, rmse, r2, y_pred, model, …}}
    best_name : str   name of the selected best model
    y_test    : array-like  true target values

    Returns
    -------
    list[Path]  paths of all saved chart images
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    saved.append(_chart_model_comparison(results))
    saved.append(_chart_actual_vs_predicted(results, y_test))
    saved.append(_chart_residuals(results, best_name, y_test))
    saved.append(_chart_error_distribution(results, best_name, y_test))

    print(f"\n[CHARTS] Saved {len(saved)} chart(s) to {CHARTS_DIR}/")
    for p in saved:
        print(f"  → {p.name}")
    return saved


# ── 1. Bar chart: model comparison ────────────────────────────────────────────
def _chart_model_comparison(results: dict) -> Path:
    models  = list(results.keys())
    metrics = ["mae", "mse", "rmse", "r2"]
    labels  = ["MAE", "MSE", "RMSE", "R²"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold", y=1.02)

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for ax, metric, label in zip(axes, metrics, labels):
        vals = [results[m][metric] for m in models]
        bars = ax.bar(models, vals, color=colors, edgecolor="white", linewidth=1.2)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylabel(label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = CHARTS_DIR / "model_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 2. Actual vs Predicted scatter ────────────────────────────────────────────
def _chart_actual_vs_predicted(results: dict, y_test) -> Path:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Actual vs Predicted (G3)", fontsize=16, fontweight="bold", y=1.02)

    colors = ["#3498db", "#e74c3c", "#2ecc71"]

    for ax, (name, data), c in zip(axes, results.items(), colors):
        y_pred = data["y_pred"]
        y_true = np.array(y_test)

        ax.scatter(y_true, y_pred, alpha=0.6, color=c, edgecolors="white", s=50)
        lo = min(y_true.min(), y_pred.min()) - 1
        hi = max(y_true.max(), y_pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual G3")
        ax.set_ylabel("Predicted G3")
        ax.set_title(f"{name}\nR² = {data['r2']:.4f}", fontsize=12)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = CHARTS_DIR / "actual_vs_predicted.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 3. Residual plot (best model) ────────────────────────────────────────────
def _chart_residuals(results: dict, best_name: str, y_test) -> Path:
    y_pred    = results[best_name]["y_pred"]
    y_true    = np.array(y_test)
    residuals = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.6, color="#9b59b6", edgecolors="white", s=50)
    ax.axhline(0, color="black", linestyle="--", lw=1.2)
    ax.set_xlabel("Predicted G3", fontsize=12)
    ax.set_ylabel("Residual  (Actual − Predicted)", fontsize=12)
    ax.set_title(f"Residual Plot — {best_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = CHARTS_DIR / "residuals_best_model.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 4. Error distribution histogram (best model) ─────────────────────────────
def _chart_error_distribution(results: dict, best_name: str, y_test) -> Path:
    y_pred    = results[best_name]["y_pred"]
    y_true    = np.array(y_test)
    errors    = y_true - y_pred

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(errors, kde=True, bins=20, color="#1abc9c", ax=ax, edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", lw=1.2, label="Zero error")
    ax.set_xlabel("Prediction Error  (Actual − Predicted)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(f"Error Distribution — {best_name}", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    path = CHARTS_DIR / "error_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
