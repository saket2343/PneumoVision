"""
evaluator.py
------------
Computes the full evaluation metric suite and generates publication-quality
figures: confusion matrix, ROC curve, precision-recall curve, classification
report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")


class Evaluator:
    def __init__(self, class_names=("NORMAL", "PNEUMONIA"), output_dir: Path = Path("outputs")):
        self.class_names = class_names
        self.output_dir = Path(output_dir)

    def compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # = recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "mcc": matthews_corrcoef(y_true, y_pred),
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }
        return metrics

    def classification_report_dict(self, y_true, y_prob, threshold: float = 0.5) -> dict:
        y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
        return classification_report(
            y_true, y_pred, target_names=list(self.class_names), output_dict=True, zero_division=0
        )

    def plot_confusion_matrix(self, y_true, y_prob, threshold: float = 0.5, save: bool = True) -> Path:
        y_pred = (np.asarray(y_prob).ravel() >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=self.class_names, yticklabels=self.class_names, ax=ax,
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()

        path = self.output_dir / "confusion_matrix" / "confusion_matrix.png"
        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_roc_curve(self, y_true, y_prob, save: bool = True) -> Path:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.4f})", linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        fig.tight_layout()

        path = self.output_dir / "roc" / "roc_curve.png"
        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def plot_pr_curve(self, y_true, y_prob, save: bool = True) -> Path:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.4f})", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="lower left")
        fig.tight_layout()

        path = self.output_dir / "pr_curve" / "pr_curve.png"
        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=200)
        plt.close(fig)
        return path

    def full_report(self, y_true, y_prob, threshold: float = 0.5) -> Dict:
        metrics = self.compute_metrics(y_true, y_prob, threshold)
        report = self.classification_report_dict(y_true, y_prob, threshold)
        self.plot_confusion_matrix(y_true, y_prob, threshold)
        self.plot_roc_curve(y_true, y_prob)
        self.plot_pr_curve(y_true, y_prob)
        return {"metrics": metrics, "classification_report": report}
