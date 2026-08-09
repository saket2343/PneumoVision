"""
calibrate.py
------------
Fits temperature scaling on the VALIDATION set only (never test) and
reports whether it actually improves calibration (NLL, Brier score) and a
reliability diagram. Does NOT automatically change your deployment —
review the diagnostics, and if it helps, pass the printed --temperature
value into tune_threshold.py / evaluate.py / predict.py / app.py yourself.

Usage:
    python calibrate.py --model-path models/best_model.keras
"""

from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf

from config import CONFIG, ensure_directories
from src.calibration import fit_temperature, plot_reliability_diagram, reliability_bins
from src.dataset import PneumoniaDataset
from src.utils import get_logger, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Fit temperature scaling on the validation set only.")
    parser.add_argument("--model-path", type=str, default=str(CONFIG.paths.best_model_path))
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(CONFIG.train.seed)
    ensure_directories()
    logger = get_logger("calibrate", CONFIG.paths.logs_dir)

    logger.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    val_data = PneumoniaDataset(
        CONFIG.paths.val_dir,
        image_size=CONFIG.data.image_size,
        batch_size=args.batch_size,
        class_names=CONFIG.data.class_names,
        use_clahe=CONFIG.data.use_clahe,
        cache=False,
    )
    val_ds = val_data.build(training=False)

    logger.info(f"Running predictions on {len(val_data)} VALIDATION images (calibration must never use test data).")
    y_prob = model.predict(val_ds, verbose=1).ravel()
    y_true = np.array(val_data.labels)[: len(y_prob)]

    best_t, diag = fit_temperature(y_true, y_prob)

    print("\n" + "=" * 72)
    print("TEMPERATURE SCALING RESULT (fit on validation set)")
    print("=" * 72)
    print(f"Recommended temperature: {best_t:.3f}  (1.0 = no change; >1 softens overconfidence; <1 sharpens)")
    print(f"Negative log-likelihood: {diag['nll_before']:.4f} -> {diag['nll_after']:.4f} "
          f"({'improved' if diag['nll_after'] < diag['nll_before'] else 'did NOT improve'})")
    print(f"Brier score:             {diag['brier_before']:.4f} -> {diag['brier_after']:.4f} "
          f"({'improved' if diag['brier_after'] < diag['brier_before'] else 'did NOT improve'})")

    improved = diag["nll_after"] < diag["nll_before"] and diag["brier_after"] < diag["brier_before"]
    at_boundary = best_t <= 0.05 + 1e-9 or best_t >= 5.0 - 1e-9
    if at_boundary:
        print(
            f"\n⚠ The recommended temperature ({best_t:.3f}) sits right at the edge of the search "
            "range [0.05, 5.0]. This usually means the validation set is small enough (or separable "
            "enough) that NLL keeps improving by pushing probabilities toward the extremes without "
            "limit — a degenerate fit, not a meaningful calibration. Treat this result with caution; "
            "it's more likely a symptom of a small/easy validation set than a real calibration factor."
        )
    elif improved:
        print(
            "\nCalibration improved both NLL and Brier score on validation. This means the model's "
            "probabilities were miscalibrated (not necessarily wrong in ranking — ROC-AUC is unaffected "
            "by a monotonic rescaling like this). It does NOT by itself fix a systematic bias toward one "
            "class; re-run tune_threshold.py on calibrated probabilities to see whether the recommended "
            "operating threshold or the achievable specificity actually changes."
        )
    else:
        print(
            "\nCalibration did not improve both metrics — the raw probabilities may already be "
            "reasonably calibrated, or the miscalibration isn't well captured by a single global "
            "temperature. Consider this evidence AGAINST applying calibration, not a reason to force it."
        )

    reports_dir = CONFIG.paths.reports_dir
    save_json(
        {"diagnostics": diag, "reliability_before": reliability_bins(y_true, y_prob)},
        reports_dir / "calibration.json",
    )
    diagram_path = plot_reliability_diagram(y_true, y_prob, save_path=reports_dir / "reliability_diagram_before.png")
    logger.info(f"Saved calibration report to {reports_dir / 'calibration.json'}")
    logger.info(f"Saved reliability diagram to {diagram_path}")

    print(f"\nTo use this in threshold tuning / evaluation, pass: --temperature {best_t:.3f}")


if __name__ == "__main__":
    main()
