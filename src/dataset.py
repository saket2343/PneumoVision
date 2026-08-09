"""
dataset.py
----------
Phase 4 of the pipeline: a custom tf.data-based loader with lazy loading,
batching, prefetching, and optional caching, wired to the preprocessing
(preprocessing.py) and augmentation (augmentations.py) modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src.augmentations import apply_augmentation, get_train_augmentations, get_val_augmentations
from src.preprocessing import apply_clahe, resize_image, to_rgb


def _list_files_and_labels(directory: Path, class_names: Tuple[str, str]) -> Tuple[List[str], List[int]]:
    directory = Path(directory)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    files, labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = directory / class_name
        if not class_dir.exists():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in valid_ext:
                files.append(str(f))
                labels.append(label)
    return files, labels


class PneumoniaDataset:
    """Builds tf.data.Dataset objects for train / val / test splits.

    Design notes:
    - Decoding + CLAHE + augmentation run in a `tf.py_function` because
      Albumentations and CLAHE (OpenCV) are not native TF ops. This keeps
      preprocessing byte-for-byte identical between training and inference.
    - `.cache()` is applied on the *decoded* dataset before augmentation so
      repeated epochs don't re-hit disk, while augmentation still varies
      every epoch.
    - `.prefetch(AUTOTUNE)` overlaps preprocessing of the next batch with
      GPU/CPU model execution.
    """

    def __init__(
        self,
        data_dir: Path,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        class_names: Tuple[str, str] = ("NORMAL", "PNEUMONIA"),
        use_clahe: bool = True,
        shuffle_buffer: int = 1024,
        cache: bool = True,
        seed: int = 42,
        oversample: bool = False,
        oversample_ratio: float = 0.5,
    ):
        """
        oversample: if True, replicate minority-class file paths (sampling
            with replacement, seeded) until the minority class reaches
            `oversample_ratio` of the majority class's count. Off by default.

            IMPORTANT: this is a TRAIN-ONLY mechanism. Only pass
            oversample=True when constructing the dataset for `dataset/train`
            — never for `dataset/val` or `dataset/test`. This class has no
            way to detect which split `data_dir` is on its own, so the
            caller is responsible for this; `train.py` only ever sets it on
            the training dataset. Passing oversample=True for val/test would
            duplicate validation/test images, which the project brief
            explicitly forbids (it invalidates threshold selection and the
            final unbiased test evaluation).
        oversample_ratio: target ratio of (minority count / majority count)
            after oversampling, e.g. 0.5 = minority ends up at half the
            majority count ("moderate" correction, not forced 50/50). Only
            used when oversample=True.
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.class_names = class_names
        self.use_clahe = use_clahe
        self.shuffle_buffer = shuffle_buffer
        self.cache = cache
        self.seed = seed
        self.oversample = oversample
        self.oversample_ratio = oversample_ratio

        self.files, self.labels = _list_files_and_labels(self.data_dir, class_names)
        self.original_file_count = len(self.files)

        if self.oversample and self.original_file_count > 0:
            self.files, self.labels = self._apply_oversampling(self.files, self.labels)

        if len(self.files) == 0:
            # Empty dataset is allowed at scaffold time (no data downloaded yet).
            self.class_weights = {0: 1.0, 1: 1.0}
        else:
            self.class_weights = self._compute_class_weights()

    def _apply_oversampling(self, files: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
        """Replicate minority-class samples (with replacement) up to
        `oversample_ratio` of the majority count. Deterministic given `seed`.
        """
        labels_arr = np.array(labels)
        counts = {c: int((labels_arr == c).sum()) for c in range(len(self.class_names))}
        if len(counts) < 2 or min(counts.values()) == 0:
            return files, labels  # nothing sensible to do

        majority_class = max(counts, key=counts.get)
        minority_class = min(counts, key=counts.get)
        majority_count = counts[majority_class]
        minority_count = counts[minority_class]

        target_minority_count = int(majority_count * self.oversample_ratio)
        if target_minority_count <= minority_count:
            return files, labels  # already at/above the target ratio, nothing to add

        n_to_add = target_minority_count - minority_count
        minority_files = [f for f, l in zip(files, labels) if l == minority_class]

        rng = np.random.RandomState(self.seed)
        added = list(rng.choice(minority_files, size=n_to_add, replace=True))

        new_files = list(files) + added
        new_labels = list(labels) + [minority_class] * n_to_add
        return new_files, new_labels

    def _compute_class_weights(self) -> dict:
        labels = np.array(self.labels)
        n_total = len(labels)
        n_classes = len(self.class_names)
        weights = {}
        for c in range(n_classes):
            n_c = max((labels == c).sum(), 1)
            weights[c] = n_total / (n_classes * n_c)
        return weights

    def _decode_and_preprocess(self, path: tf.Tensor, label: tf.Tensor):
        def _py_decode(path_bytes):
            path_str = path_bytes.numpy().decode("utf-8")
            raw = tf.io.read_file(path_str)
            img = tf.image.decode_image(raw, channels=3, expand_animations=False).numpy()
            img = to_rgb(img)
            img = resize_image(img, self.image_size)
            if self.use_clahe:
                img = apply_clahe(img)
            return img.astype(np.uint8)

        img = tf.py_function(_py_decode, [path], Tout=tf.uint8)
        img.set_shape([self.image_size[0], self.image_size[1], 3])
        return img, label

    def _augment(self, img: tf.Tensor, label: tf.Tensor, training: bool):
        pipeline = get_train_augmentations(self.image_size) if training else get_val_augmentations(self.image_size)

        def _py_aug(img_uint8):
            img_float = img_uint8.numpy().astype(np.float32) / 255.0
            out = apply_augmentation(img_float, pipeline)
            return out.astype(np.float32)

        img = tf.py_function(_py_aug, [img], Tout=tf.float32)
        img.set_shape([self.image_size[0], self.image_size[1], 3])
        return img, label

    def build(self, training: bool = False, drop_remainder: bool = False) -> tf.data.Dataset:
        if len(self.files) == 0:
            raise FileNotFoundError(
                f"No images found under {self.data_dir}. "
                "Download the dataset and place it under dataset/<split>/<class>/ before building."
            )

        ds = tf.data.Dataset.from_tensor_slices((self.files, self.labels))

        if training:
            ds = ds.shuffle(min(self.shuffle_buffer, len(self.files)), seed=self.seed, reshuffle_each_iteration=True)

        ds = ds.map(self._decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

        if self.cache:
            ds = ds.cache()

        ds = ds.map(
            lambda img, label: self._augment(img, label, training=training),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(lambda img, label: (img, tf.cast(label, tf.float32)))
        ds = ds.batch(self.batch_size, drop_remainder=drop_remainder)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def __len__(self) -> int:
        return len(self.files)
