"""
export_model.py
----------------
Stretch goal: export the trained Keras model to TensorFlow SavedModel,
ONNX, and (optionally) TensorFlow Lite formats for portable deployment.

Usage:
    python export_model.py --model-path models/final_model.keras --formats savedmodel onnx tflite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from config import CONFIG
from src.utils import get_logger


def export_savedmodel(model: tf.keras.Model, out_dir: Path) -> Path:
    path = out_dir / "savedmodel"
    model.export(str(path))  # Keras 3 / TF2.16+ style export
    return path


def export_onnx(saved_model_path: Path, out_path: Path) -> Path:
    import subprocess

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--saved-model", str(saved_model_path),
        "--output", str(out_path),
        "--opset", "15",
    ]
    subprocess.run(cmd, check=True)
    return out_path


def export_tflite(model: tf.keras.Model, out_path: Path) -> Path:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export trained model to portable formats.")
    parser.add_argument("--model-path", type=str, default=str(CONFIG.paths.final_model_path))
    parser.add_argument("--formats", nargs="+", default=["savedmodel", "onnx", "tflite"],
                         choices=["savedmodel", "onnx", "tflite"])
    parser.add_argument("--out-dir", type=str, default=str(CONFIG.paths.models_dir / "exported"))
    args = parser.parse_args()

    logger = get_logger("export", CONFIG.paths.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(args.model_path)

    saved_model_path = None
    if "savedmodel" in args.formats:
        saved_model_path = export_savedmodel(model, out_dir)
        logger.info(f"SavedModel exported to {saved_model_path}")

    if "onnx" in args.formats:
        if saved_model_path is None:
            saved_model_path = export_savedmodel(model, out_dir)
        try:
            onnx_path = export_onnx(saved_model_path, out_dir / "model.onnx")
            logger.info(f"ONNX model exported to {onnx_path}")
        except Exception as e:  # pragma: no cover - depends on tf2onnx/opset compatibility
            logger.warning(f"ONNX export failed: {e}")

    if "tflite" in args.formats:
        tflite_path = export_tflite(model, out_dir / "model.tflite")
        logger.info(f"TFLite model exported to {tflite_path}")


if __name__ == "__main__":
    main()
