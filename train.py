"""
train.py
--------
CLI entry point for training the pneumonia classifier end-to-end:
build datasets -> build model -> two-stage training -> save best/final weights.

Flags below exist to support the controlled-experiment workflow (class
balancing, loss function, backbone) — change ONE at a time and compare
against the baseline via tune_threshold.py on the VALIDATION set, per the
audit brief. Don't combine --oversample with class weighting without a
reason: both correct for the same imbalance, and combining them compounds
two separate corrections rather than being automatically "more correct".

Usage:
    python train.py
    python train.py --epochs 30 --fine-tune-epochs 10 --no-fine-tune
    python train.py --backbone EfficientNetB3 --attention se

    # Experiment 2 (class balancing) variants:
    python train.py                                          # A: current class-weighting baseline
    python train.py --oversample --oversample-ratio 0.5 --no-class-weights   # B: moderate oversampling only
    python train.py --oversample --oversample-ratio 0.5      # B+A combined (compounds — treat as its own variant)

    # Experiment 4 (focal loss):
    python train.py --loss focal --focal-gamma 2.0 --focal-alpha 0.25
"""

from __future__ import annotations

import argparse
import dataclasses

from config import CONFIG, ensure_directories
from src.dataset import PneumoniaDataset
from src.model import build_model
from src.trainer import Trainer
from src.utils import get_logger, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the pneumonia detection model.")
    parser.add_argument("--epochs", type=int, default=CONFIG.train.epochs)
    parser.add_argument("--fine-tune-epochs", type=int, default=CONFIG.train.fine_tune_epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.data.batch_size)
    parser.add_argument("--backbone", type=str, default=CONFIG.model.backbone)
    parser.add_argument("--attention", type=str, default=None, choices=[None, "se", "cbam"])
    parser.add_argument("--fine-tune-at-layer", type=int, default=CONFIG.model.fine_tune_at_layer,
                         help="Negative index: unfreeze the last N backbone layers during stage 2 (Experiment 5).")
    parser.add_argument("--no-fine-tune", action="store_true", help="Skip stage 2 fine-tuning.")
    parser.add_argument("--no-cache", action="store_true", help="Disable tf.data .cache().")
    parser.add_argument("--seed", type=int, default=CONFIG.train.seed)

    # Experiment 2: class balancing
    parser.add_argument("--oversample", action="store_true",
                         help="Moderately oversample the minority TRAIN class only (never val/test).")
    parser.add_argument("--oversample-ratio", type=float, default=0.5,
                         help="Target minority:majority ratio after oversampling (0.5 = moderate, not forced 50/50).")
    parser.add_argument("--no-class-weights", action="store_true",
                         help="Disable class-weighted loss (pass uniform weights to model.fit()).")

    # Experiment 4: focal loss
    parser.add_argument("--loss", type=str, default=CONFIG.train.loss, choices=["bce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=CONFIG.train.focal_gamma)
    parser.add_argument("--focal-alpha", type=float, default=CONFIG.train.focal_alpha)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_directories()

    logger = get_logger("train", CONFIG.paths.logs_dir)
    logger.info("Starting training run.")
    logger.info(
        f"Config: epochs={args.epochs}, fine_tune_epochs={args.fine_tune_epochs}, "
        f"batch_size={args.batch_size}, backbone={args.backbone}, attention={args.attention}, "
        f"fine_tune_at_layer={args.fine_tune_at_layer}, loss={args.loss}, "
        f"oversample={args.oversample} (ratio={args.oversample_ratio if args.oversample else 'n/a'}), "
        f"class_weights_disabled={args.no_class_weights}"
    )

    train_data = PneumoniaDataset(
        CONFIG.paths.train_dir,
        image_size=CONFIG.data.image_size,
        batch_size=args.batch_size,
        class_names=CONFIG.data.class_names,
        use_clahe=CONFIG.data.use_clahe,
        cache=not args.no_cache,
        seed=args.seed,
        oversample=args.oversample,
        oversample_ratio=args.oversample_ratio,
    )
    # NOTE: oversample is deliberately NEVER passed here — val must stay untouched.
    val_data = PneumoniaDataset(
        CONFIG.paths.val_dir,
        image_size=CONFIG.data.image_size,
        batch_size=args.batch_size,
        class_names=CONFIG.data.class_names,
        use_clahe=CONFIG.data.use_clahe,
        cache=not args.no_cache,
        seed=args.seed,
    )

    logger.info(f"Train images: {len(train_data)}"
                + (f" (oversampled from {train_data.original_file_count})" if args.oversample else "")
                + f" | Val images: {len(val_data)}")

    class_weight = {0: 1.0, 1: 1.0} if args.no_class_weights else train_data.class_weights
    logger.info(f"Class weights passed to model.fit(): {class_weight}"
                + (" (disabled via --no-class-weights)" if args.no_class_weights else ""))

    train_ds = train_data.build(training=True)
    val_ds = val_data.build(training=False)

    model = build_model(
        input_shape=CONFIG.model.input_shape,
        dropout_1=CONFIG.model.dropout_1,
        dropout_2=CONFIG.model.dropout_2,
        dense_1=CONFIG.model.dense_1,
        dense_2=CONFIG.model.dense_2,
        freeze_backbone=CONFIG.model.freeze_backbone,
        attention=args.attention,
        backbone_name=args.backbone,
    )
    model.summary(print_fn=logger.info)

    model_cfg = dataclasses.replace(CONFIG.model, fine_tune_at_layer=args.fine_tune_at_layer)
    train_cfg = dataclasses.replace(
        CONFIG.train,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )
    run_cfg = dataclasses.replace(CONFIG, model=model_cfg, train=train_cfg)

    trainer = Trainer(model, run_cfg, logger=logger)
    trainer.fit(train_ds, val_ds, class_weight=class_weight, fine_tune=not args.no_fine_tune)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
