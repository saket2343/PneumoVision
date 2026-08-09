"""
augmentations.py
-----------------
Phase 3 of the pipeline: training-only data augmentation using
Albumentations. Validation/test data must NEVER pass through this module.

NOTE (fixed 2026-08-08): the transform parameters here were verified against
the actually-installed Albumentations version's real constructor signatures
(inspected via `inspect.signature`), not assumed from memory or older docs.
Two real, version-specific breaking changes were found and fixed:
  - `RandomResizedCrop` takes `size=(h, w)` in current Albumentations, not
    separate `height=`/`width=` kwargs (that older signature now raises a
    pydantic "Field required: size" error and crashes training immediately).
  - `GaussNoise` takes `std_range=(lo, hi)` (a fraction of the image's max
    value — for our float32 [0,1] pipeline, a fraction of 1.0) in current
    Albumentations, not the older `var_limit=(lo, hi)` (variance on a 0-255
    scale), which no longer exists as a parameter at all.
`ShiftScaleRotate` + `Affine` were also consolidated into one `Affine` call
(shift/scale/shear together) since current Albumentations treats
`ShiftScaleRotate` as a deprecated special case of `Affine` and warns on
every use otherwise.
"""

from __future__ import annotations

from typing import Tuple

import albumentations as A
import numpy as np


def get_train_augmentations(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """Augmentation pipeline applied only to the training split.

    Transforms are intentionally conservative in magnitude: chest X-rays
    are anatomically constrained (lungs are always roughly centered and
    upright), so aggressive flips/rotations can create unrealistic,
    label-noisy samples.
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, border_mode=0, p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                shear=(-5, 5),
                rotate=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.RandomResizedCrop(
                size=(image_size[0], image_size[1]),
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                p=0.3,
            ),
            A.ElasticTransform(alpha=1, sigma=20, p=0.1),
            A.GaussNoise(std_range=(0.01, 0.03), p=0.15),  # fraction of max value (float32 pipeline -> fraction of 1.0)
            A.Resize(height=image_size[0], width=image_size[1]),
        ]
    )


def get_val_augmentations(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """No-op pipeline (resize only) for validation/test — deterministic."""
    return A.Compose([A.Resize(height=image_size[0], width=image_size[1])])


def apply_augmentation(image: np.ndarray, pipeline: A.Compose) -> np.ndarray:
    """Apply an Albumentations pipeline to a single float32 [0,1] RGB image."""
    result = pipeline(image=image)
    return result["image"]


def get_tta_augmentations(image_size: Tuple[int, int] = (224, 224)):
    """A small, label-preserving set of deterministic-ish transforms used
    for Test-Time Augmentation (stretch goal). Returns a list of callables.
    """
    transforms = [
        A.Compose([A.Resize(height=image_size[0], width=image_size[1])]),  # identity
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(height=image_size[0], width=image_size[1])]),
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Resize(height=image_size[0], width=image_size[1])]),
        A.Compose([A.Rotate(limit=5, border_mode=0, p=1.0), A.Resize(height=image_size[0], width=image_size[1])]),
    ]
    return [lambda img, t=t: t(image=img)["image"] for t in transforms]
