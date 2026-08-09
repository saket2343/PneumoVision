"""
threshold_optimizer.py
-----------------------
Validation-set-only threshold selection for the pneumonia classifier.

This module deliberately has NO knowledge of the test set — it only ever
operates on whatever (y_true, y_prob) arrays are handed to it, and the
calling script (`tune_threshold.py`) is responsible for making sure those
come from the VALIDATION split. The test set must only be touched once, by
`evaluate.py`, using the threshold this module recommends.

Because this is a screening/triage tool, the primary selection criterion is
NOT accuracy — it's "recall (sensitivity) on PNEUMONIA must be at least a
target level, and among thresholds meeting that bar, pick the one with the
best specificity" (i.e. minimize false alarms without letting real cases
slip through). This is a project modeling choice, not a claim of clinical
validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def load_recommended_threshold(reports_dir: Path, default: float = 0.5) -> Tuple[float, str]:
    """Load the threshold `tune_threshold.py` last recommended, from
    `<reports_dir>/threshold_recommendation.json`. Returns
    (threshold, provenance_message) — used by predict.py and app.py so the
    deployed decision threshold is never silently hard-coded: either it
    comes from an actual validation-tuning run, or the caller is told
    explicitly that it fell back to `default` and why.
    """
    import json

    path = Path(reports_dir) / "threshold_recommendation.json"
    if not path.exists():
        return default, (
            f"No threshold_recommendation.json found at {path} — using the neutral default {default}. "
            "Run tune_threshold.py first to get a validation-selected threshold."
        )
    try:
        with open(path, "r") as f:
            data = json.load(f)
        threshold = float(data["recommended"]["threshold"])
        return threshold, f"Loaded from {path} (tune_threshold.py recommendation)."
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        return default, f"Could not parse {path} ({e}) — using the neutral default {default}."


def sweep_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    step: float = 0.01,
) -> pd.DataFrame:
    """Compute the full metric suite at every threshold in the sweep range.

    Returns a DataFrame with one row per threshold and columns:
    threshold, accuracy, precision, recall, sensitivity, specificity, f1,
    balanced_accuracy, youden_j, tp, tn, fp, fn.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    thresholds = np.round(np.arange(min_threshold, max_threshold + step / 2, step), 4)
    rows: List[Dict] = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on PNEUMONIA (class 1)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = sensitivity
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2
        youden_j = sensitivity + specificity - 1

        rows.append(
            {
                "threshold": float(t),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
                "youden_j": youden_j,
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
            }
        )

    return pd.DataFrame(rows)


def select_sensitivity_constrained(df: pd.DataFrame, min_sensitivity: float = 0.95) -> Optional[Dict]:
    """Among thresholds with sensitivity >= min_sensitivity, pick the one
    with the highest specificity (ties broken by the higher threshold, i.e.
    the more conservative — fewer false alarms — of the tied options).

    Returns None if no threshold in the sweep reaches min_sensitivity at all
    (this can happen if the model's probabilities don't separate the classes
    well enough anywhere in the swept range).
    """
    eligible = df[df["sensitivity"] >= min_sensitivity]
    if eligible.empty:
        return None
    best = eligible.sort_values(["specificity", "threshold"], ascending=[False, False]).iloc[0]
    return best.to_dict()


def select_best_youden_j(df: pd.DataFrame) -> Dict:
    return df.loc[df["youden_j"].idxmax()].to_dict()


def select_best_f1(df: pd.DataFrame) -> Dict:
    return df.loc[df["f1"].idxmax()].to_dict()


def select_best_balanced_accuracy(df: pd.DataFrame) -> Dict:
    return df.loc[df["balanced_accuracy"].idxmax()].to_dict()


def probability_distribution_summary(y_true: np.ndarray, y_prob: np.ndarray, class_names=("NORMAL", "PNEUMONIA")) -> Dict:
    """Per-class percentile summary of predicted probabilities.

    Answers: "are probabilities for the two classes actually separated
    anywhere, or is the model just outputting a narrow band of high values
    for everyone?"
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    summary = {}
    for label, name in enumerate(class_names):
        probs = y_prob[y_true == label]
        if len(probs) == 0:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(len(probs)),
            "min": float(np.min(probs)),
            "max": float(np.max(probs)),
            "mean": float(np.mean(probs)),
            "median": float(np.median(probs)),
            "p25": float(np.percentile(probs, 25)),
            "p75": float(np.percentile(probs, 75)),
            "p95": float(np.percentile(probs, 95)),
        }
    return summary


def confusion_category_probability_summary(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, class_names=("NORMAL", "PNEUMONIA")
) -> Dict:
    """Per-audit-brief Experiment 9: splits predictions into the four
    confusion categories (TN/FP/TP/FN) at a given threshold and reports a
    percentile summary of predicted probability within each — for
    distinguishing whether errors look like a threshold problem (FP/FN
    probabilities close to the boundary), a calibration problem (probabilities
    confidently wrong), or something else entirely.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    categories = {
        f"{class_names[0]}_correct (TN)": (y_true == 0) & (y_pred == 0),
        f"{class_names[0]}_false_positive (FP)": (y_true == 0) & (y_pred == 1),
        f"{class_names[1]}_correct (TP)": (y_true == 1) & (y_pred == 1),
        f"{class_names[1]}_false_negative (FN)": (y_true == 1) & (y_pred == 0),
    }

    summary = {}
    for name, mask in categories.items():
        probs = y_prob[mask]
        if len(probs) == 0:
            summary[name] = {"count": 0}
            continue
        summary[name] = {
            "count": int(len(probs)),
            "min": float(np.min(probs)),
            "max": float(np.max(probs)),
            "mean": float(np.mean(probs)),
            "median": float(np.median(probs)),
            "p25": float(np.percentile(probs, 25)),
            "p75": float(np.percentile(probs, 75)),
            "p95": float(np.percentile(probs, 95)),
        }
    return summary


def plot_threshold_curves(df: pd.DataFrame, recommended_threshold: Optional[float], save_path: Path) -> Path:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    for col, label in [
        ("sensitivity", "Sensitivity (Recall)"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1", "F1 Score"),
        ("balanced_accuracy", "Balanced Accuracy"),
    ]:
        ax.plot(df["threshold"], df[col], label=label, linewidth=2)

    if recommended_threshold is not None:
        ax.axvline(recommended_threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Recommended threshold = {recommended_threshold:.2f}")

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("Metric vs. Threshold (Validation Set)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_probability_distribution(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names=("NORMAL", "PNEUMONIA"),
    recommended_threshold: Optional[float] = None,
    save_path: Path = Path("outputs/reports/probability_distribution.png"),
) -> Path:
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, name, color in [(0, class_names[0], "steelblue"), (1, class_names[1], "indianred")]:
        probs = y_prob[y_true == label]
        if len(probs) > 0:
            ax.hist(probs, bins=40, alpha=0.55, label=f"Actual {name} (n={len(probs)})", color=color, range=(0, 1))

    if recommended_threshold is not None:
        ax.axvline(recommended_threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Recommended threshold = {recommended_threshold:.2f}")

    ax.set_xlabel("Predicted probability of PNEUMONIA")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability Distribution by True Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
