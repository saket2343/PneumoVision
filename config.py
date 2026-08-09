"""
config.py
---------
Central configuration for the Pneumonia Detection project.

All hyperparameters, paths, and reproducibility settings live here so that
every script (train.py, evaluate.py, predict.py, app.py) shares a single
source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PathConfig:
    root: Path = Path(__file__).resolve().parent
    dataset_dir: Path = root / "dataset"
    train_dir: Path = dataset_dir / "train"
    val_dir: Path = dataset_dir / "val"
    test_dir: Path = dataset_dir / "test"

    models_dir: Path = root / "models"
    outputs_dir: Path = root / "outputs"
    confusion_matrix_dir: Path = outputs_dir / "confusion_matrix"
    roc_dir: Path = outputs_dir / "roc"
    pr_curve_dir: Path = outputs_dir / "pr_curve"
    heatmaps_dir: Path = outputs_dir / "heatmaps"
    reports_dir: Path = outputs_dir / "reports"
    error_analysis_dir: Path = outputs_dir / "error_analysis"
    logs_dir: Path = root / "logs"

    best_model_path: Path = models_dir / "best_model.keras"
    final_model_path: Path = models_dir / "final_model.keras"


@dataclass(frozen=True)
class DataConfig:
    image_size: Tuple[int, int] = (224, 224)
    channels: int = 3
    batch_size: int = 32
    shuffle_buffer: int = 1024
    class_names: Tuple[str, str] = ("NORMAL", "PNEUMONIA")
    positive_class: str = "PNEUMONIA"  # class == 1
    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "EfficientNetB0"
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    dropout_1: float = 0.4
    dropout_2: float = 0.3
    dense_1: int = 256
    dense_2: int = 128
    freeze_backbone: bool = True
    fine_tune_at_layer: int = -40  # unfreeze last N layers during fine-tuning


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    fine_tune_epochs: int = 15
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_lr: float = 1e-7
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    seed: int = 42
    loss: str = "bce"  # "bce" | "focal" — see src/trainer.py Trainer._build_loss
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25  # weight on the positive (PNEUMONIA) class in focal loss


@dataclass(frozen=True)
class ThresholdConfig:
    min_threshold: float = 0.01
    max_threshold: float = 0.99
    step: float = 0.01
    min_sensitivity: float = 0.95  # project modeling criterion, not a clinical claim


@dataclass(frozen=True)
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)


CONFIG = Config()


def ensure_directories() -> None:
    """Create every output directory declared in PathConfig if missing."""
    p = CONFIG.paths
    for attr in vars(p):
        path = getattr(p, attr)
        if isinstance(path, Path) and path.suffix == "":  # directories only
            path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print("Directories ensured under:", CONFIG.paths.root)
