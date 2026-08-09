"""
experiment_tracker.py
----------------------
Appends one row per controlled experiment to a persistent CSV log
(`outputs/reports/experiment_log.csv`) and can render it as a markdown
table — the tracking artifact requested by the audit brief, so every
experiment (class balancing variant, augmentation variant, focal loss,
fine-tuning depth, backbone, threshold) is comparable side by side instead
of living only in scrollback.

This module only ever records numbers you pass it — it does not compute or
guess metrics itself. Call it from tune_threshold.py (validation metrics,
once per experiment) or manually from a notebook/script.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

COLUMNS = [
    "experiment",
    "backbone",
    "class_weighting",
    "oversampling",
    "loss",
    "augmentation",
    "fine_tuning",
    "threshold",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "balanced_accuracy",
    "mcc",
    "roc_auc",
    "tp",
    "tn",
    "fp",
    "fn",
    "notes",
]


def log_experiment(
    log_path: Path,
    experiment: str,
    metrics: Dict,
    backbone: str = "",
    class_weighting: str = "",
    oversampling: str = "",
    loss: str = "",
    augmentation: str = "",
    fine_tuning: str = "",
    notes: str = "",
) -> Path:
    """Append one row. `metrics` should contain whatever subset of
    {threshold, sensitivity, specificity, precision, f1, balanced_accuracy,
    mcc, roc_auc, tp, tn, fp, fn} you have — missing keys are left blank
    rather than guessed.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "experiment": experiment,
        "backbone": backbone,
        "class_weighting": class_weighting,
        "oversampling": oversampling,
        "loss": loss,
        "augmentation": augmentation,
        "fine_tuning": fine_tuning,
        "notes": notes,
    }
    for key in ["threshold", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "mcc", "roc_auc", "tp", "tn", "fp", "fn"]:
        row[key] = metrics.get(key, "")

    file_exists = log_path.exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return log_path


def render_markdown_table(log_path: Path) -> str:
    """Render the CSV log as a compact markdown table (the format requested
    in the audit brief) for pasting into a report or PR description.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return "_No experiments logged yet._"

    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return "_No experiments logged yet._"

    display_cols = ["experiment", "backbone", "threshold", "sensitivity", "specificity", "f1", "balanced_accuracy", "mcc", "roc_auc", "fp", "fn"]
    header = "| " + " | ".join(display_cols) + " |"
    separator = "|" + "|".join(["---"] * len(display_cols)) + "|"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in display_cols) + " |")

    return "\n".join(lines)
