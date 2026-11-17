import zipfile
from pathlib import PurePosixPath

import pytest

from src.data.fruitnet_inventory import extract_apple, is_apple_folder, target_path


def make_zip(path, names):
    with zipfile.ZipFile(path, "w") as zf:
        for n in names:
            zf.writestr(n, b"x")
    return path


def test_extract_apple_separa_mixed(tmp_path):
    z = make_zip(tmp_path / "f.zip", [
        "Processed/Apple_Good/a.jpg",
        "Processed/Apple_Bad/b.jpg",
        "Mixed Quality_Fruits/Apple/c.jpg",
        "Processed/Pineapple_Good/p.jpg",
        "Processed/Apple_Good/notes.txt",
    ])
    kg, mx = tmp_path / "kaggle", tmp_path / "mixed"
    assert extract_apple(z, kg, mx) == 3
    assert (kg / "Processed/Apple_Good/a.jpg").exists()
    assert (kg / "Processed/Apple_Bad/b.jpg").exists()
    assert (mx / "Apple/c.jpg").exists()
    assert not list(kg.rglob("c.jpg"))
    assert not list(mx.rglob("a.jpg"))
    assert not list(tmp_path.rglob("p.jpg"))


def test_extract_apple_rechaza_path_traversal(tmp_path):
    z = make_zip(tmp_path / "evil.zip", ["../Apple_Good/x.jpg"])
    kg, mx = tmp_path / "k" / "raw", tmp_path / "m"
    with pytest.raises(ValueError):
        extract_apple(z, kg, mx)
    assert not (tmp_path / "k" / "Apple_Good" / "x.jpg").exists()


def test_target_path(tmp_path):
    kg, mx = tmp_path / "k", tmp_path / "m"
    assert target_path(PurePosixPath("Mixed Quality/Apple/c.jpg"), kg, mx) == mx / "Apple" / "c.jpg"
    assert target_path(PurePosixPath("Train/Apple_Good/a.jpg"), kg, mx) == kg / "Train/Apple_Good/a.jpg"


def test_is_apple_folder():
    assert is_apple_folder("Apple") and is_apple_folder("Apple_Good") and is_apple_folder("apple_bad")
    assert not is_apple_folder("Pineapple_Good") and not is_apple_folder("Applesauce")
