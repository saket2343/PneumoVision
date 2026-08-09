"""
trainer.py
----------
Compiles and trains the model: AdamW optimizer, BCE loss, full metric
suite, mixed precision, gradient clipping, and the callback stack
(EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard).

Training happens in two stages, matching the spec:
  1. Frozen backbone, higher LR (feature-extraction stage).
  2. Unfrozen last N backbone layers, low LR (fine-tuning stage).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras import mixed_precision

from src.model import unfreeze_for_fine_tuning
from src.utils import get_logger


class Trainer:
    def __init__(self, model: tf.keras.Model, config, logger=None):
        self.model = model
        self.cfg = config
        self.logger = logger or get_logger("trainer", config.paths.logs_dir)
        self.history_stage1 = None
        self.history_stage2 = None

        if config.train.mixed_precision:
            try:
                mixed_precision.set_global_policy("mixed_float16")
                self.logger.info("Mixed precision enabled (mixed_float16).")
            except Exception as e:  # pragma: no cover - depends on hardware
                self.logger.warning(f"Could not enable mixed precision: {e}")

    def _build_optimizer(self, learning_rate: float) -> tf.keras.optimizers.Optimizer:
        try:
            optimizer = tf.keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=self.cfg.train.weight_decay,
                clipnorm=self.cfg.train.gradient_clip_norm,
            )
        except AttributeError:  # older TF without native AdamW
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate,
                clipnorm=self.cfg.train.gradient_clip_norm,
            )
            self.logger.warning("tf.keras.optimizers.AdamW unavailable; falling back to Adam (no decoupled weight decay).")
        return optimizer

    def _metrics(self):
        return [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ]

    def _callbacks(self, checkpoint_path: Path, stage: str):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        log_dir = self.cfg.paths.logs_dir / f"tensorboard_{stage}_{time.strftime('%Y%m%d-%H%M%S')}"

        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc",
                mode="max",
                patience=self.cfg.train.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.cfg.train.reduce_lr_factor,
                patience=self.cfg.train.reduce_lr_patience,
                min_lr=self.cfg.train.min_lr,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_auc",
                mode="max",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(log_dir=str(log_dir)),
            tf.keras.callbacks.CSVLogger(str(self.cfg.paths.logs_dir / f"history_{stage}.csv")),
        ]

    def _build_loss(self):
        loss_name = getattr(self.cfg.train, "loss", "bce")
        if loss_name == "focal":
            self.logger.info(
                f"Using BinaryFocalCrossentropy (gamma={self.cfg.train.focal_gamma}, "
                f"alpha={self.cfg.train.focal_alpha}, apply_class_balancing=True)."
            )
            return tf.keras.losses.BinaryFocalCrossentropy(
                apply_class_balancing=True,
                alpha=self.cfg.train.focal_alpha,
                gamma=self.cfg.train.focal_gamma,
                label_smoothing=self.cfg.train.label_smoothing,
            )
        if loss_name != "bce":
            self.logger.warning(f"Unknown loss '{loss_name}', falling back to BinaryCrossentropy.")
        return tf.keras.losses.BinaryCrossentropy(label_smoothing=self.cfg.train.label_smoothing)

    def compile_stage1(self):
        self.model.compile(
            optimizer=self._build_optimizer(self.cfg.train.learning_rate),
            loss=self._build_loss(),
            metrics=self._metrics(),
        )

    def compile_stage2(self):
        self.model = unfreeze_for_fine_tuning(self.model, self.cfg.model.fine_tune_at_layer)
        self.model.compile(
            optimizer=self._build_optimizer(self.cfg.train.fine_tune_learning_rate),
            loss=self._build_loss(),
            metrics=self._metrics(),
        )

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        class_weight: Optional[dict] = None,
        fine_tune: bool = True,
    ):
        loss_name = getattr(self.cfg.train, "loss", "bce")
        if loss_name == "focal" and class_weight is not None and len(set(round(v, 3) for v in class_weight.values())) > 1:
            self.logger.warning(
                "Using focal loss (apply_class_balancing=True, alpha-weighted) together with "
                "non-uniform class_weight compounds two separate imbalance corrections at once — "
                "the combined effect is not necessarily 'more correct' and should be treated as its "
                "own controlled experiment, not assumed better. Consider testing focal loss with "
                "uniform class_weight (train.py --no-class-weights) as a separate variant."
            )

        self.logger.info("Stage 1: training classification head with frozen backbone.")
        self.compile_stage1()
        self.history_stage1 = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.cfg.train.epochs,
            class_weight=class_weight,
            callbacks=self._callbacks(self.cfg.paths.best_model_path, stage="stage1"),
            verbose=1,
        )

        if fine_tune:
            self.logger.info("Stage 2: fine-tuning unfrozen backbone layers at low LR.")
            self.compile_stage2()
            self.history_stage2 = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.cfg.train.fine_tune_epochs,
                class_weight=class_weight,
                callbacks=self._callbacks(self.cfg.paths.best_model_path, stage="stage2"),
                verbose=1,
            )

        self.model.save(self.cfg.paths.final_model_path)
        self.logger.info(f"Final model saved to {self.cfg.paths.final_model_path}")
        return self.model
