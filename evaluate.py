"""
evaluate.py
-----------
CLI entry point for the ONE final, unbiased evaluation on the held-out test
set. Generates: metrics JSON, classification report, confusion matrix, ROC
curve, PR curve, probability-distribution diagnostics, and an error-analysis
visualization grid.

Correct workflow (see tune_threshold.py):
    python train.py            -> models/best_model.keras (selected by val AUC)
    python tune_threshold.py   -> recommends a threshold using ONLY the validation set
    python evaluate.py --model-path models/best_model.keras --threshold <recommended>
                                -> the one, final, unbiased test-set number

This script NEVER searches for a threshold itself — it only ever applies
whatever `--threshold` you pass (default 0.5, purely as a neutral fallback
for a quick sanity check; it is NOT a substitute for tune_threshold.py).
Threshold-specific outputs (test_evaluation, confusion matrix, error
analysis) are saved with the threshold in the filename so repeated runs at
different thresholds don't overwrite each other. ROC/PR curves are
threshold-independent and are saved once (they show performance across all
thresholds, not for one).

--temperature: if you fitted calibration on validation (calibrate.py) and
decided it helps, pass the same temperature here that you passed to
tune_threshold.py — this is a REPORTING choice, not a search: this script
still never fits or selects a temperature itself, it only applies whatever
value you supply, exactly like --threshold.

Usage:
    python evaluate.py
    python evaluate.py --model-path models/final_model.keras --threshold 0.72
    python evaluate.py --model-path models/best_model.keras --threshold 0.72 --temperature 1.8
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from config import CONFIG, ensure_directories
from src.calibration import apply_temperature
from src.dataset import PneumoniaDataset
from src.error_analysis import ErrorAnalyzer
from src.evaluator import Evaluator
from src.preprocessing import load_image
from src.threshold_optimizer import confusion_category_probability_summary, probability_distribution_summary
from src.utils import get_logger, save_json, set_seed


def _threshold_tag(threshold: float) -> str:
    """0.72 -> '0_72', 0.5 -> '0_50' — used to namespace per-threshold output files."""
    return f"{threshold:.2f}".replace(".", "_")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final, unbiased evaluation on the held-out test set.")
    parser.add_argument("--model-path", type=str, default=str(CONFIG.paths.best_model_path))
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold to apply. Default 0.5 is a neutral fallback only — "
             "run tune_threshold.py first and pass its recommended threshold here for "
             "a scientifically defensible evaluation.",
    )
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Validation-fitted temperature from calibrate.py (1.0 = no calibration).")
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size)
    parser.add_argument(
        "--i-know-this-is-not-threshold-tuning", action="store_true",
        help="Suppresses the reminder printed when --threshold is left at the 0.5 default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(CONFIG.train.seed)
    ensure_directories()
    logger = get_logger("evaluate", CONFIG.paths.logs_dir)

    if args.threshold == 0.5 and not args.i_know_this_is_not_threshold_tuning:
        logger.warning(
            "Using the default threshold of 0.5. If you haven't already, run "
            "`python tune_threshold.py` first (on the VALIDATION set) and pass its "
            "recommended threshold here via --threshold. Do NOT search for a threshold "
            "against this test set — that would bias the 'unbiased' evaluation."
        )

    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    test_data = PneumoniaDataset(
        CONFIG.paths.test_dir,
        image_size=CONFIG.data.image_size,
        batch_size=args.batch_size,
        class_names=CONFIG.data.class_names,
        use_clahe=CONFIG.data.use_clahe,
        cache=False,
    )
    test_ds = test_data.build(training=False)

    n_normal = sum(1 for l in test_data.labels if l == 0)
    n_pneumonia = sum(1 for l in test_data.labels if l == 1)
    logger.info(f"Test set composition: NORMAL={n_normal}, PNEUMONIA={n_pneumonia}, total={len(test_data)}")

    logger.info(f"Running predictions on {len(test_data)} test images.")
    y_prob_raw = model.predict(test_ds, verbose=1).ravel()
    y_true = np.array(test_data.labels)[: len(y_prob_raw)]

    if args.temperature != 1.0:
        logger.info(f"Applying temperature scaling (T={args.temperature}) — this value should have come from "
                    f"calibrate.py fit on the VALIDATION set, never fitted here on test data.")
        y_prob = apply_temperature(y_prob_raw, args.temperature)
    else:
        y_prob = y_prob_raw

    roc_auc = float(roc_auc_score(y_true, y_prob))
    logger.info(f"Test ROC-AUC: {roc_auc:.4f}")

    # Task 12: probability distribution diagnostics on the test set too.
    prob_summary = probability_distribution_summary(y_true, y_prob, CONFIG.data.class_names)
    logger.info("Predicted probability distribution by true class (test set):")
    for cls_name, stats in prob_summary.items():
        logger.info(f"  {cls_name}: {stats}")

    evaluator = Evaluator(class_names=CONFIG.data.class_names, output_dir=CONFIG.paths.outputs_dir)
    report = evaluator.full_report(y_true, y_prob, threshold=args.threshold)
    report["probability_distribution"] = prob_summary
    report["threshold_used"] = args.threshold
    report["temperature_used"] = args.temperature
    report["metrics"]["roc_auc"] = roc_auc  # ensure this reflects the (possibly calibrated) probabilities used here

    logger.info("Test metrics:")
    for k, v in report["metrics"].items():
        logger.info(f"  {k}: {v}")

    tag = _threshold_tag(args.threshold)

    # Task 8: threshold-specific report filename so repeated runs don't overwrite each other.
    report_path = CONFIG.paths.reports_dir / f"test_evaluation_threshold_{tag}.json"
    save_json(report, report_path)
    logger.info(f"Saved full report to {report_path}")

    # Confusion-category probability breakdown (Experiment 9, test-set version) — also threshold-tagged.
    category_summary = confusion_category_probability_summary(y_true, y_prob, args.threshold, CONFIG.data.class_names)
    save_json(category_summary, CONFIG.paths.reports_dir / f"confusion_category_probabilities_test_threshold_{tag}.json")

    # Rename the confusion matrix this run just wrote (Evaluator always writes the same
    # filename) into a threshold-specific one, so it isn't overwritten by the next run.
    generic_cm_path = CONFIG.paths.confusion_matrix_dir / "confusion_matrix.png"
    tagged_cm_path = CONFIG.paths.confusion_matrix_dir / f"confusion_matrix_threshold_{tag}.png"
    if generic_cm_path.exists():
        generic_cm_path.replace(tagged_cm_path)
        logger.info(f"Saved confusion matrix to {tagged_cm_path}")

    # Error analysis grid (also threshold-dependent -> also namespaced).
    analyzer = ErrorAnalyzer(class_names=CONFIG.data.class_names, output_dir=CONFIG.paths.error_analysis_dir)
    errors = analyzer.find_errors(test_data.files, y_true, y_prob, threshold=args.threshold)

    def _load_for_display(path):
        return load_image(path)

    analyzer.plot_grid(
        errors["most_confident_mistakes"], "Most Confident Mistakes",
        f"most_confident_mistakes_threshold_{tag}.png", _load_for_display,
    )
    analyzer.plot_grid(
        errors["least_confident"], "Least Confident (Borderline) Predictions",
        f"least_confident_threshold_{tag}.png", _load_for_display,
    )
    save_json(
        {"false_positives": errors["false_positives"], "false_negatives": errors["false_negatives"]},
        CONFIG.paths.error_analysis_dir / f"fp_fn_lists_threshold_{tag}.json",
    )

    logger.info(f"Evaluation complete at threshold={args.threshold:.2f}. See outputs/ for plots and reports.")


if __name__ == "__main__":
    main()
