from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.dataset import PneumoniaDataset, _list_files_and_labels


@pytest.fixture()
def fake_dataset_dir(tmp_path):
    for cls, n in [("NORMAL", 6), ("PNEUMONIA", 3)]:
        d = tmp_path / cls
        d.mkdir()
        for i in range(n):
            img = Image.fromarray((np.random.rand(50, 50, 3) * 255).astype("uint8"))
            img.save(d / f"img_{i}.png")
    return tmp_path


def test_list_files_and_labels(fake_dataset_dir):
    files, labels = _list_files_and_labels(fake_dataset_dir, ("NORMAL", "PNEUMONIA"))
    assert len(files) == 9
    assert labels.count(0) == 6
    assert labels.count(1) == 3


def test_class_weights_favor_minority_class(fake_dataset_dir):
    ds = PneumoniaDataset(fake_dataset_dir, class_names=("NORMAL", "PNEUMONIA"), cache=False)
    assert ds.class_weights[1] > ds.class_weights[0]  # PNEUMONIA (minority) gets higher weight


def test_build_raises_on_empty_dir(tmp_path):
    ds = PneumoniaDataset(tmp_path, class_names=("NORMAL", "PNEUMONIA"), cache=False)
    with pytest.raises(FileNotFoundError):
        ds.build()
