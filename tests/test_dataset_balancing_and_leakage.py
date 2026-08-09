from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.dataset import PneumoniaDataset
from src.utils import analyze_patient_leakage, extract_patient_id, find_all_duplicate_groups


@pytest.fixture()
def imbalanced_dataset_dir(tmp_path):
    """20 NORMAL, 60 PNEUMONIA — mirrors the reported ~2.88x imbalance direction."""
    for cls, n in [("NORMAL", 20), ("PNEUMONIA", 60)]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            img = Image.fromarray((np.random.rand(20, 20, 3) * 255).astype("uint8"))
            img.save(d / f"{cls.lower()}_{i}.png")
    return tmp_path


class TestOversampling:
    def test_oversample_false_leaves_counts_unchanged(self, imbalanced_dataset_dir):
        ds = PneumoniaDataset(imbalanced_dataset_dir, oversample=False, cache=False)
        assert ds.original_file_count == 80
        assert len(ds.files) == 80

    def test_oversample_moderate_ratio_hits_target(self, imbalanced_dataset_dir):
        ds = PneumoniaDataset(imbalanced_dataset_dir, oversample=True, oversample_ratio=0.5, seed=42, cache=False)
        labels = np.array(ds.labels)
        # majority (PNEUMONIA=1) count of 60 * 0.5 = 30 target minority count
        assert (labels == 1).sum() == 60  # majority untouched
        assert (labels == 0).sum() == 30  # minority raised to target
        assert ds.original_file_count == 80
        assert len(ds.files) == 90  # 80 original + 10 added

    def test_oversample_deterministic_given_seed(self, imbalanced_dataset_dir):
        ds1 = PneumoniaDataset(imbalanced_dataset_dir, oversample=True, oversample_ratio=0.5, seed=7, cache=False)
        ds2 = PneumoniaDataset(imbalanced_dataset_dir, oversample=True, oversample_ratio=0.5, seed=7, cache=False)
        assert ds1.files == ds2.files

    def test_oversample_ratio_already_met_does_nothing(self, imbalanced_dataset_dir):
        # ratio=0.2 -> target minority count = 60*0.2=12, already have 20 -> no-op
        ds = PneumoniaDataset(imbalanced_dataset_dir, oversample=True, oversample_ratio=0.2, cache=False)
        assert len(ds.files) == 80

    def test_val_dataset_never_oversampled_by_default(self, imbalanced_dataset_dir):
        # oversample defaults to False -- simulates how train.py always constructs val_data
        ds = PneumoniaDataset(imbalanced_dataset_dir, cache=False)
        assert len(ds.files) == ds.original_file_count == 80


class TestPatientLeakage:
    def test_extract_patient_id_pneumonia_pattern(self):
        assert extract_patient_id("person1438_bacteria_3721.jpeg") == "1438"
        assert extract_patient_id("person124_virus_238.jpeg") == "124"

    def test_extract_patient_id_no_match_for_normal_convention(self):
        assert extract_patient_id("IM-0001-0001.jpeg") is None
        assert extract_patient_id("NORMAL2-IM-0374-0001.jpeg") is None

    def test_analyze_patient_leakage_detects_cross_split_overlap(self, tmp_path):
        splits = {}
        for split in ["train", "val", "test"]:
            d = tmp_path / split
            (d / "NORMAL").mkdir(parents=True)
            (d / "PNEUMONIA").mkdir(parents=True)
            splits[split] = d

        def make(path):
            Image.fromarray((np.random.rand(10, 10, 3) * 255).astype("uint8")).save(path)

        # person1430 appears in BOTH train and val -> should be flagged
        make(splits["train"] / "PNEUMONIA" / "person1430_bacteria_3696.jpeg")
        make(splits["val"] / "PNEUMONIA" / "person1430_bacteria_3695.jpeg")
        # person9999 only in train -> no overlap
        make(splits["train"] / "PNEUMONIA" / "person9999_virus_1.jpeg")

        report = analyze_patient_leakage(splits)
        assert report["overlaps"]["train/val"]["count"] == 1
        assert "1430" in report["overlaps"]["train/val"]["patient_ids"]
        assert report["overlaps"]["train/test"]["count"] == 0
        assert report["unique_patient_ids_per_split"]["train"] == 2


class TestDuplicateGroups:
    def test_finds_within_split_duplicate(self, tmp_path):
        d = tmp_path / "train" / "NORMAL"
        d.mkdir(parents=True)
        img = np.zeros((10, 10, 3), dtype="uint8")
        Image.fromarray(img).save(d / "a.png")
        Image.fromarray(img).save(d / "b.png")  # byte-identical copy

        groups = find_all_duplicate_groups({"train": tmp_path / "train"})
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert group["spans_multiple_splits"] is False
        assert len(group["locations"]) == 2

    def test_finds_cross_split_duplicate(self, tmp_path):
        train_dir = tmp_path / "train" / "NORMAL"
        test_dir = tmp_path / "test" / "NORMAL"
        train_dir.mkdir(parents=True)
        test_dir.mkdir(parents=True)
        img = np.ones((10, 10, 3), dtype="uint8") * 42
        Image.fromarray(img).save(train_dir / "a.png")
        Image.fromarray(img).save(test_dir / "a_copy.png")

        groups = find_all_duplicate_groups({"train": tmp_path / "train", "test": tmp_path / "test"})
        assert len(groups) == 1
        assert list(groups.values())[0]["spans_multiple_splits"] is True

    def test_no_duplicates_returns_empty(self, tmp_path):
        d = tmp_path / "train" / "NORMAL"
        d.mkdir(parents=True)
        for i in range(3):
            img = (np.random.rand(10, 10, 3) * 255).astype("uint8")
            Image.fromarray(img).save(d / f"{i}.png")
        groups = find_all_duplicate_groups({"train": tmp_path / "train"})
        assert groups == {}
