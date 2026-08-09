"""
predict.py
----------
CLI entry point for running inference on a single chest X-ray image.

By default, the decision threshold is loaded from
outputs/reports/threshold_recommendation.json (whatever tune_threshold.py
last recommended on the validation set) — never silently hard-coded.
Pass --threshold to override explicitly.

Usage:
    python predict.py --image path/to/xray.jpg
    python predict.py --image path/to/xray.jpg --tta --save-heatmap outputs/heatmaps/example.png
    python predict.py --image path/to/xray.jpg --threshold 0.68
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CONFIG, ensure_directories
from src.inference import InferenceEngine
from src.threshold_optimizer import load_recommended_threshold
from src.utils import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a single chest X-ray image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the chest X-ray image.")
    parser.add_argument("--model-path", type=str, default=str(CONFIG.paths.best_model_path))
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation.")
    parser.add_argument("--save-heatmap", type=str, default=None, help="Optional path to save the Grad-CAM overlay.")
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Decision threshold. Defaults to whatever tune_threshold.py last recommended "
             "(outputs/reports/threshold_recommendation.json), falling back to 0.5 if that "
             "doesn't exist yet.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_directories()
    logger = get_logger("predict", CONFIG.paths.logs_dir)

    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using explicitly-provided threshold: {threshold}")
    else:
        threshold, provenance = load_recommended_threshold(CONFIG.paths.reports_dir)
        logger.info(f"Threshold: {threshold} — {provenance}")

    engine = InferenceEngine(
        model_path=args.model_path,
        class_names=CONFIG.data.class_names,
        image_size=CONFIG.data.image_size,
        use_clahe=CONFIG.data.use_clahe,
        threshold=threshold,
    )

    result = engine.predict_path(args.image, tta=args.tta)

    print("=" * 40)
    print(f"Image:      {args.image}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print(f"Threshold:  {result['threshold_used']}")
    print("=" * 40)

    heatmap_path = Path(args.save_heatmap) if args.save_heatmap else CONFIG.paths.heatmaps_dir / f"{Path(args.image).stem}_gradcam.png"
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(heatmap_path, result["overlay"])
    logger.info(f"Grad-CAM overlay saved to {heatmap_path}")


if __name__ == "__main__":
    main()
