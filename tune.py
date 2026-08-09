"""
tune.py
-------
Stretch goal: hyperparameter optimization with Optuna over learning rate,
dropout, batch size, and weight decay. Runs short trials (few epochs each)
on the frozen-backbone stage and optimizes validation AUC.

Usage:
    python tune.py --n-trials 20
"""

from __future__ import annotations

import argparse

import optuna
import tensorflow as tf

from config import CONFIG, ensure_directories
from src.dataset import PneumoniaDataset
from src.model import build_model
from src.utils import get_logger, set_seed


def objective(trial: optuna.Trial, args) -> float:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout_1 = trial.suggest_float("dropout_1", 0.2, 0.6)
    dropout_2 = trial.suggest_float("dropout_2", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    train_data = PneumoniaDataset(CONFIG.paths.train_dir, batch_size=batch_size, cache=True)
    val_data = PneumoniaDataset(CONFIG.paths.val_dir, batch_size=batch_size, cache=True)
    train_ds = train_data.build(training=True)
    val_ds = val_data.build(training=False)

    model = build_model(dropout_1=dropout_1, dropout_2=dropout_2, freeze_backbone=True)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.trial_epochs,
        class_weight=train_data.class_weights,
        verbose=0,
    )
    return max(history.history["val_auc"])


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search.")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--trial-epochs", type=int, default=3)
    args = parser.parse_args()

    set_seed(CONFIG.train.seed)
    ensure_directories()
    logger = get_logger("tune", CONFIG.paths.logs_dir)

    study = optuna.create_study(direction="maximize", study_name="pneumonia_efficientnet")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best val_auc: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    from src.utils import save_json
    save_json(
        {"best_value": study.best_value, "best_params": study.best_params},
        CONFIG.paths.reports_dir / "optuna_best_params.json",
    )


if __name__ == "__main__":
    main()
