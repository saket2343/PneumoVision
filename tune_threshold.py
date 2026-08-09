"""
tune_threshold.py
------------------
Finds the deployment decision threshold using the VALIDATION set ONLY.
The test set is never touched by this script — it exists solely so
`evaluate.py` can report one final, unbiased number using whatever
threshold this script recommends.

Correct workflow:

    python train.py
        -> models/best_model.keras   (selected by validation AUC)
    python tune_threshold.py [--experiment-name my_experiment]
        -> outputs/reports/threshold_search.{json,csv}
        -> outputs/reports/threshold_curve.png
        -> outputs/reports/probability_distribution_val.png
        -> outputs/reports/top_normal_false_positives.{json,png}   (Experiment 8)
        -> outputs/reports/confusion_category_probabilities.json  (Experiment 9)
        -> outputs/reports/experiment_log.csv                     (if --experiment-name given)
        -> prints the recommended threshold + the exact evaluate.py command to run
    python evaluate.py --model-path models/best_model.keras --threshold <recommended>
        -> ONE final, unbiased test-set evaluation

Primary selection criterion (Task 5 of the pipeline spec): among thresholds
with validation sensitivity (recall on PNEUMONIA) >= --min-sensitivity
(default 0.95), pick the one with the highest specificity. This is a
project modeling choice — missing a true pneumonia case is treated as more
costly than a false alarm — not a claim that 95% is a clinically validated
number. Youden's J, best-F1, and best-balanced-accuracy thresholds are also
reported for comparison.

--temperature applies validation-fitted temperature scaling (see
calibrate.py) to the raw probabilities BEFORE the sweep — pass whatever
value calibrate.py recommended, if you decided calibration helps. Default
1.0 (no calibration) leaves probabilities untouched.
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
from src.experiment_tracker import log_experiment
from src.preprocessing import load_image
from src.threshold_optimizer import (
    confusion_category_probability_summary,
    plot_probability_distribution,
    plot_threshold_curves,
    probability_distribution_summary,
    select_best_balanced_accuracy,
    select_best_f1,
    select_best_youden_j,
    select_sensitivity_constrained,
    sweep_thresholds,
)
from src.utils import get_logger, save_json, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Find the deployment threshold on the validation set only.")
    parser.add_argument("--model-path", type=str, default=str(CONFIG.paths.best_model_path))
    parser.add_argument("--min-threshold", type=float, default=CONFIG.threshold.min_threshold)
    parser.add_argument("--max-threshold", type=float, default=CONFIG.threshold.max_threshold)
    parser.add_argument("--step", type=float, default=CONFIG.threshold.step)
    parser.add_argument("--min-sensitivity", type=float, default=CONFIG.threshold.min_sensitivity)
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size)
    parser.add_argument("--temperature", type=float, default=1.0,
                         help="Validation-fitted temperature from calibrate.py (1.0 = no calibration).")
    parser.add_argument("--top-n-fp", type=int, default=20, help="How many top NORMAL false positives to report.")
    parser.add_argument("--experiment-name", type=str, default=None,
                         help="If given, appends this run's validation metrics to outputs/reports/experiment_log.csv")
    parser.add_argument("--notes", type=str, default="", help="Free-text note stored alongside --experiment-name.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(CONFIG.train.seed)
    ensure_directories()
    logger = get_logger("tune_threshold", CONFIG.paths.logs_dir)

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

    logger.info(f"Running predictions on {len(val_data)} VALIDATION images (test set is not touched by this script).")
    y_prob_raw = model.predict(val_ds, verbose=1).ravel()
    y_true = np.array(val_data.labels)[: len(y_prob_raw)]

    if args.temperature != 1.0:
        logger.info(f"Applying validation-fitted temperature scaling (T={args.temperature}).")
        y_prob = apply_temperature(y_prob_raw, args.temperature)
    else:
        y_prob = y_prob_raw

    n_normal = int((y_true == 0).sum())
    n_pneumonia = int((y_true == 1).sum())
    logger.info(f"Validation set composition: NORMAL={n_normal}, PNEUMONIA={n_pneumonia}, total={len(y_true)}")

    roc_auc = float(roc_auc_score(y_true, y_prob))
    logger.info(f"Validation ROC-AUC: {roc_auc:.4f} (threshold-independent; unaffected by --temperature's monotonic rescaling)")

    # --- Task 12: probability distribution diagnostics ---
    prob_summary = probability_distribution_summary(y_true, y_prob, CONFIG.data.class_names)
    logger.info("Predicted probability distribution by true class (validation set):")
    for cls_name, stats in prob_summary.items():
        logger.info(f"  {cls_name}: {stats}")

    # --- Task 4: full threshold sweep ---
    df = sweep_thresholds(
        y_true, y_prob,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step,
    )

    reports_dir = CONFIG.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(reports_dir / "threshold_search.csv", index=False)
    save_json(
        {"probability_distribution": prob_summary, "roc_auc": roc_auc, "temperature": args.temperature, "sweep": df.to_dict(orient="records")},
        reports_dir / "threshold_search.json",
    )
    logger.info(f"Saved full threshold sweep to {reports_dir / 'threshold_search.csv'} and .json")

    # --- Task 5: candidate thresholds under different criteria ---
    sensitivity_constrained = select_sensitivity_constrained(df, min_sensitivity=args.min_sensitivity)
    youden = select_best_youden_j(df)
    best_f1 = select_best_f1(df)
    best_bal_acc = select_best_balanced_accuracy(df)

    print("\n" + "=" * 72)
    print("THRESHOLD CANDIDATES (validation set)" + (f"  [temperature={args.temperature}]" if args.temperature != 1.0 else ""))
    print("=" * 72)
    print(f"Validation ROC-AUC: {roc_auc:.4f}")
    print(f"Best Youden's J          : threshold={youden['threshold']:.2f}  "
          f"J={youden['youden_j']:.4f}  sens={youden['sensitivity']:.4f}  spec={youden['specificity']:.4f}")
    print(f"Best F1                  : threshold={best_f1['threshold']:.2f}  "
          f"F1={best_f1['f1']:.4f}  sens={best_f1['sensitivity']:.4f}  spec={best_f1['specificity']:.4f}")
    print(f"Best Balanced Accuracy   : threshold={best_bal_acc['threshold']:.2f}  "
          f"bal_acc={best_bal_acc['balanced_accuracy']:.4f}  sens={best_bal_acc['sensitivity']:.4f}  spec={best_bal_acc['specificity']:.4f}")

    if sensitivity_constrained is None:
        print(f"\n⚠ No threshold in [{args.min_threshold}, {args.max_threshold}] reaches "
              f"sensitivity >= {args.min_sensitivity:.2f} on the validation set.")
        print("This means the model's probabilities do not separate the classes well enough yet.")
        print("Falling back to the Youden's J threshold as the recommendation; consider more training/data.")
        recommended = youden
        reason = f"No threshold reached sensitivity >= {args.min_sensitivity:.0%}; used best Youden's J instead."
    else:
        recommended = sensitivity_constrained
        reason = (
            f"Sensitivity >= {args.min_sensitivity:.0%} on the validation set, "
            f"and this threshold gives the highest specificity ({recommended['specificity']:.4f}) "
            f"among all eligible thresholds."
        )

    print("\n" + "=" * 72)
    print(f"RECOMMENDED THRESHOLD: {recommended['threshold']:.2f}")
    print("=" * 72)
    print(f"Reason: {reason}")
    print(f"  accuracy={recommended['accuracy']:.4f}  precision={recommended['precision']:.4f}  "
          f"recall/sensitivity={recommended['sensitivity']:.4f}  specificity={recommended['specificity']:.4f}  "
          f"f1={recommended['f1']:.4f}  balanced_accuracy={recommended['balanced_accuracy']:.4f}")
    print(f"  TP={recommended['tp']}  TN={recommended['tn']}  FP={recommended['fp']}  FN={recommended['fn']}")

    # MCC at the recommended threshold (not produced by sweep_thresholds, computed here for the tracker/report)
    from sklearn.metrics import matthews_corrcoef
    y_pred_rec = (y_prob >= recommended["threshold"]).astype(int)
    mcc = float(matthews_corrcoef(y_true, y_pred_rec))
    print(f"  MCC={mcc:.4f}")

    save_json(
        {
            "recommended": recommended,
            "reason": reason,
            "min_sensitivity_target": args.min_sensitivity,
            "roc_auc": roc_auc,
            "mcc": mcc,
            "temperature": args.temperature,
            "alternatives": {
                "best_youden_j": youden,
                "best_f1": best_f1,
                "best_balanced_accuracy": best_bal_acc,
            },
        },
        reports_dir / "threshold_recommendation.json",
    )

    # --- Task 6: plots ---
    curve_path = plot_threshold_curves(df, recommended["threshold"], reports_dir / "threshold_curve.png")
    dist_path = plot_probability_distribution(
        y_true, y_prob, CONFIG.data.class_names, recommended["threshold"],
        reports_dir / "probability_distribution_val.png",
    )
    logger.info(f"Saved threshold curve to {curve_path}")
    logger.info(f"Saved probability distribution plot to {dist_path}")

    # --- Experiment 8: NORMAL false-positive report + montage ---
    analyzer = ErrorAnalyzer(class_names=CONFIG.data.class_names, output_dir=reports_dir)
    top_fp = analyzer.top_normal_false_positives(val_data.files, y_true, y_prob, threshold=recommended["threshold"], n=args.top_n_fp)
    save_json({"top_normal_false_positives": top_fp}, reports_dir / "top_normal_false_positives.json")
    montage_path = analyzer.plot_normal_fp_montage(top_fp, load_image, filename="top_normal_false_positives.png")
    logger.info(f"Saved top {len(top_fp)} NORMAL false-positive report to "
                f"{reports_dir / 'top_normal_false_positives.json'}"
                + (f" and montage to {montage_path}" if montage_path else ""))

    # --- Experiment 9: confusion-category probability distributions ---
    category_summary = confusion_category_probability_summary(y_true, y_prob, recommended["threshold"], CONFIG.data.class_names)
    save_json(category_summary, reports_dir / "confusion_category_probabilities.json")
    logger.info("Probability distribution by confusion category (at recommended threshold):")
    for cat, stats in category_summary.items():
        logger.info(f"  {cat}: {stats}")

    # --- Experiment tracking ---
    if args.experiment_name:
        log_path = log_experiment(
            reports_dir / "experiment_log.csv",
            experiment=args.experiment_name,
            metrics={
                "threshold": round(recommended["threshold"], 4),
                "sensitivity": round(recommended["sensitivity"], 4),
                "specificity": round(recommended["specificity"], 4),
                "precision": round(recommended["precision"], 4),
                "f1": round(recommended["f1"], 4),
                "balanced_accuracy": round(recommended["balanced_accuracy"], 4),
                "mcc": round(mcc, 4),
                "roc_auc": round(roc_auc, 4),
                "tp": recommended["tp"], "tn": recommended["tn"], "fp": recommended["fp"], "fn": recommended["fn"],
            },
            notes=args.notes,
        )
        logger.info(f"Logged this run to {log_path} as experiment '{args.experiment_name}'.")

    # --- Task 9: exact final command ---
    print("\n" + "=" * 72)
    print("NEXT STEP — run the ONE final, unbiased test-set evaluation:")
    print("=" * 72)
    temp_note = f" --temperature {args.temperature}" if args.temperature != 1.0 else ""
    print(f"    .venv/bin/python evaluate.py --model-path {args.model_path} --threshold {recommended['threshold']:.2f}{temp_note}")
    print()


if __name__ == "__main__":
    main()
