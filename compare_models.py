"""
compare_models.py
------------------
Stretch goal: train and compare several backbones (EfficientNet-B0/B3,
DenseNet121, ResNet50, MobileNetV3Large) under identical conditions and
produce a comparison table of test-set metrics.

Usage:
    python compare_models.py --epochs 10 --backbones EfficientNetB0 DenseNet121 MobileNetV3Large
"""

from __future__ import annotations

import argparse

import pandas as pd
import tensorflow as tf

from config import CONFIG, ensure_directories
from src.dataset import PneumoniaDataset
from src.evaluator import Evaluator
from src.model import build_model
from src.trainer import Trainer
from src.utils import get_logger, set_seed


def main():
    parser = argparse.ArgumentParser(description="Compare multiple backbones under identical settings.")
    parser.add_argument("--backbones", nargs="+", default=["EfficientNetB0", "EfficientNetB1", "EfficientNetB2", "EfficientNetB3"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size)
    args = parser.parse_args()

    set_seed(CONFIG.train.seed)
    ensure_directories()
    logger = get_logger("compare_models", CONFIG.paths.logs_dir)

    train_data = PneumoniaDataset(CONFIG.paths.train_dir, batch_size=args.batch_size, cache=True)
    val_data = PneumoniaDataset(CONFIG.paths.val_dir, batch_size=args.batch_size, cache=True)
    test_data = PneumoniaDataset(CONFIG.paths.test_dir, batch_size=args.batch_size, cache=False)

    train_ds = train_data.build(training=True)
    val_ds = val_data.build(training=False)
    test_ds = test_data.build(training=False)

    evaluator = Evaluator(class_names=CONFIG.data.class_names, output_dir=CONFIG.paths.outputs_dir)
    rows = []

    for backbone in args.backbones:
        logger.info(f"=== Training backbone: {backbone} ===")
        model = build_model(backbone_name=backbone, freeze_backbone=True)

        import dataclasses
        run_cfg = dataclasses.replace(CONFIG, train=dataclasses.replace(CONFIG.train, epochs=args.epochs, fine_tune_epochs=0))
        trainer = Trainer(model, run_cfg, logger=logger)
        trainer.fit(train_ds, val_ds, class_weight=train_data.class_weights, fine_tune=False)

        import numpy as np
        y_prob = model.predict(test_ds, verbose=0).ravel()
        y_true = np.array(test_data.labels)[: len(y_prob)]
        metrics = evaluator.compute_metrics(y_true, y_prob)
        metrics["backbone"] = backbone
        metrics["params"] = model.count_params()
        rows.append(metrics)
        logger.info(f"{backbone}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}")

    df = pd.DataFrame(rows).set_index("backbone")
    out_path = CONFIG.paths.reports_dir / "model_comparison.csv"
    df.to_csv(out_path)
    logger.info(f"Comparison table saved to {out_path}")
    print(df)


if __name__ == "__main__":
    main()
