"""
error_analysis.py
------------------
Automatically identifies false positives, false negatives, the most
confident mistakes, and the least confident (borderline) predictions,
then renders a visualization grid for qualitative review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class ErrorAnalyzer:
    def __init__(self, class_names=("NORMAL", "PNEUMONIA"), output_dir: Path = Path("outputs/error_analysis")):
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_errors(self, file_paths: List[str], y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()
        y_pred = (y_prob >= threshold).astype(int)

        fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
        fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]

        # Confidence = distance from decision boundary
        confidence = np.abs(y_prob - 0.5)
        wrong_idx = np.where(y_pred != y_true)[0]
        most_confident_mistakes = wrong_idx[np.argsort(-confidence[wrong_idx])][:10]
        least_confident = np.argsort(confidence)[:10]

        return {
            "false_positives": [file_paths[i] for i in fp_idx],
            "false_negatives": [file_paths[i] for i in fn_idx],
            "most_confident_mistakes": [(file_paths[i], float(y_prob[i]), int(y_true[i])) for i in most_confident_mistakes],
            "least_confident": [(file_paths[i], float(y_prob[i]), int(y_true[i])) for i in least_confident],
        }

    def top_normal_false_positives(
        self,
        file_paths: List[str],
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
        n: int = 20,
    ) -> List[Dict]:
        """The N actual-NORMAL images with the highest predicted PNEUMONIA
        probability — i.e. the cases the model is most confidently wrong
        about in the direction that drives false positives. Each entry
        records filename, predicted probability, predicted class, and the
        threshold used, per the audit brief's Experiment 8.

        Includes every NORMAL image ranked by probability (not just ones
        that cross `threshold`) so you can see how close the near-misses
        are too — the ones with y_pred == 1 are the actual false positives.
        """
        y_true = np.asarray(y_true).ravel()
        y_prob = np.asarray(y_prob).ravel()

        normal_idx = np.where(y_true == 0)[0]
        ranked = normal_idx[np.argsort(-y_prob[normal_idx])][:n]

        results = []
        for i in ranked:
            pred_class = self.class_names[int(y_prob[i] >= threshold)]
            results.append(
                {
                    "filename": file_paths[i],
                    "predicted_probability": float(y_prob[i]),
                    "predicted_class": pred_class,
                    "true_class": self.class_names[0],
                    "threshold": threshold,
                    "is_false_positive": bool(y_prob[i] >= threshold),
                }
            )
        return results

    def plot_normal_fp_montage(
        self,
        entries: List[Dict],
        load_fn,
        filename: str = "top_normal_false_positives.png",
        n_cols: int = 5,
    ):
        """Montage of the highest-probability NORMAL images (per
        `top_normal_false_positives`), each captioned with its predicted
        probability — for visually inspecting what's confusing the model.
        """
        n = len(entries)
        if n == 0:
            return None
        n_cols = min(n_cols, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, entry in zip(axes, entries):
            try:
                img = load_fn(entry["filename"])
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "load error", ha="center", va="center")
            marker = " (FP)" if entry["is_false_positive"] else ""
            ax.set_title(f"p={entry['predicted_probability']:.3f}{marker}", fontsize=9)
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Actual NORMAL images with highest predicted PNEUMONIA probability")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_grid(self, entries, title: str, filename: str, load_fn, n_cols: int = 5):
        """entries: list of (path, prob, true_label) tuples. load_fn(path)-> displayable RGB array."""
        n = len(entries)
        if n == 0:
            return None
        n_cols = min(n_cols, n)
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = np.array(axes).reshape(-1)

        for ax, (path, prob, true_label) in zip(axes, entries):
            try:
                img = load_fn(path)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "load error", ha="center", va="center")
            pred_label = self.class_names[int(prob >= 0.5)]
            true_name = self.class_names[true_label]
            ax.set_title(f"true={true_name}\npred={pred_label} ({prob:.2f})", fontsize=9)
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(title)
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
