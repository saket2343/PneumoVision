"""
utils.py
--------
Shared, reusable utilities: reproducibility, logging setup, and small
I/O helpers used across training, evaluation, and inference.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set every relevant RNG seed for reproducible runs.

    Sets Python's `random`, NumPy, and TensorFlow/Keras seeds, and asks
    TensorFlow to run deterministic ops where supported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
        # Deterministic ops (may reduce throughput slightly, disabled by
        # default in most GPU pipelines but harmless on CPU).
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    except ImportError:
        pass


def get_logger(name: str, log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Create a logger that writes to stdout and, optionally, a timestamped file."""
    logger = logging.getLogger(name)
    if logger.handlers:  # avoid duplicate handlers on repeated calls
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        file_handler = logging.FileHandler(log_dir / f"{name}_{timestamp}.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def save_json(data: Dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


class Timer:
    """Simple context manager for timing code blocks.

    Example:
        with Timer("training") as t:
            ...
        print(t.elapsed)
    """

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.perf_counter() - self._start


def find_all_duplicate_groups(split_dirs: Dict[str, Path]) -> Dict[str, dict]:
    """Group every image file by MD5 content hash, across all given splits.

    Returns a dict mapping an MD5 hash to a small record for every hash that
    has more than one file (a "duplicate group"), regardless of whether the
    group is confined to one split or spans several:

        {hash: {"locations": [...], "spans_multiple_splits": bool}}

    This is the broader counterpart to `find_cross_split_duplicates` (which
    only returns cross-split groups) — use this one when you want the full
    picture (e.g. "30 duplicate groups, 62 images, of which 6 groups cross
    train/val") and `find_cross_split_duplicates` when you only care about
    the leakage-relevant subset.
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    hash_to_locations: Dict[str, list] = {}

    for split_name, split_dir in split_dirs.items():
        split_dir = Path(split_dir)
        if not split_dir.exists():
            continue
        for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            for f in class_dir.iterdir():
                if f.suffix.lower() not in valid_ext:
                    continue
                try:
                    digest = _file_md5(f)
                except OSError:
                    continue
                location = f"{split_name}/{class_dir.name}/{f.name}"
                hash_to_locations.setdefault(digest, []).append(location)

    groups = {}
    for digest, locations in hash_to_locations.items():
        if len(locations) < 2:
            continue
        seen_splits = {loc.split("/", 1)[0] for loc in locations}
        groups[digest] = {"locations": locations, "spans_multiple_splits": len(seen_splits) > 1}
    return groups


def find_cross_split_duplicates(split_dirs: Dict[str, Path]) -> Dict[str, list]:
    """Detect byte-identical files appearing in more than one split (e.g. the
    same X-ray present in both train/ and test/) via MD5 content hashing.

    `split_dirs` maps a split name ("train", "val", "test") to its directory
    (each expected to contain class subfolders). Returns a dict mapping an
    MD5 hash to the list of "split/class/filename" locations that share it,
    restricted to hashes that appear in more than one split — i.e. actual
    leakage candidates, not just within-split duplicates.
    """
    all_groups = find_all_duplicate_groups(split_dirs)
    return {h: g["locations"] for h, g in all_groups.items() if g["spans_multiple_splits"]}


_PATIENT_ID_PATTERN = re.compile(r"^person(\d+)_", re.IGNORECASE)


def extract_patient_id(filename: str) -> Optional[str]:
    """Extract a patient ID from a filename following the common
    `person<ID>_bacteria_<n>.jpeg` / `person<ID>_virus_<n>.jpeg` naming
    convention (as seen in the Kermany/Mooney chest X-ray dataset's
    PNEUMONIA files). Returns None if the filename doesn't match — e.g. most
    NORMAL images in that dataset follow an unrelated `IM-####-####.jpeg`
    convention with no reliable per-patient grouping, which this function
    correctly reports as "no patient ID found" rather than guessing.
    """
    match = _PATIENT_ID_PATTERN.match(Path(filename).name)
    return match.group(1) if match else None


def analyze_patient_leakage(split_dirs: Dict[str, Path]) -> Dict:
    """For every image across the given splits, extract a patient ID where
    the filename allows it (see `extract_patient_id`) and report:
      - unique patient IDs found per split
      - which patient IDs appear in more than one split (real leakage: the
        same patient's X-rays split across train/val/test lets the model
        partly "recognize" a patient it already trained on)
      - how many files per split had NO extractable patient ID (reported
        separately rather than silently ignored, since a naming convention
        that doesn't match is itself useful information, not a non-finding)
    """
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    split_to_patient_ids: Dict[str, set] = {}
    unmatched_counts: Dict[str, int] = {}
    patient_id_to_splits: Dict[str, set] = {}

    for split_name, split_dir in split_dirs.items():
        split_dir = Path(split_dir)
        split_to_patient_ids[split_name] = set()
        unmatched_counts[split_name] = 0
        if not split_dir.exists():
            continue
        for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            for f in class_dir.iterdir():
                if f.suffix.lower() not in valid_ext:
                    continue
                pid = extract_patient_id(f.name)
                if pid is None:
                    unmatched_counts[split_name] += 1
                    continue
                split_to_patient_ids[split_name].add(pid)
                patient_id_to_splits.setdefault(pid, set()).add(split_name)

    overlaps = {}
    split_names = list(split_dirs.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            shared = split_to_patient_ids[a] & split_to_patient_ids[b]
            overlaps[f"{a}/{b}"] = sorted(shared)

    return {
        "unique_patient_ids_per_split": {k: len(v) for k, v in split_to_patient_ids.items()},
        "unmatched_filenames_per_split": unmatched_counts,
        "overlaps": {k: {"count": len(v), "patient_ids": v} for k, v in overlaps.items()},
    }


def count_images(directory: Path) -> Dict[str, int]:
    """Count images per class subdirectory. Used by EDA and sanity checks."""
    directory = Path(directory)
    counts: Dict[str, int] = {}
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    if not directory.exists():
        return counts
    for class_dir in sorted(d for d in directory.iterdir() if d.is_dir()):
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in valid_ext)
        counts[class_dir.name] = n
    return counts


def _file_md5(path: Path, chunk_size: int = 65536) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
