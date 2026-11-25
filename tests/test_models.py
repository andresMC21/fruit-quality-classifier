"""Pruebas unitarias: segmentación, tamaño, características y fuga en el split."""
import numpy as np
import cv2
import pytest

from src.data.features import FEATURE_NAMES, extract_features
from src.data.segment import segment_image
from src.models.size_estimator import equivalent_diameter_px, find_reference, size_class, terciles, to_mm


def synthetic(radius=80, ref_radius=None, size=400):
    img = np.full((size, size, 3), 235, np.uint8)
    cv2.circle(img, (size // 2, size // 2), radius, (40, 60, 200), -1)         # "fruta" roja sobre fondo claro
    if ref_radius:
        cv2.circle(img, (size - 60, 60), ref_radius, (120, 150, 160), -1)      # "moneda"
    return img


def test_segmentation_finds_circle():
    r = segment_image(synthetic(80))
    assert r.ok and r.method == "otsu"
    assert abs(equivalent_diameter_px(r.mask) - 160) < 8


def test_size_with_reference_in_mm():
    img = synthetic(80, ref_radius=20)
    seg = segment_image(img)
    ref_px = find_reference(img, seg.mask)
    assert ref_px is not None and abs(ref_px - 40) < 4
    d_mm = to_mm(equivalent_diameter_px(seg.mask), ref_px, 25.0)          # moneda de 25 mm
    assert abs(d_mm - 100) < 8                                              # fruta de 160 px = 4 x moneda


def test_size_class_and_terciles():
    lo, hi = terciles(np.arange(100))
    assert size_class(5, (lo, hi)) == "pequeno" and size_class(50, (lo, hi)) == "mediano" and size_class(95, (lo, hi)) == "grande"
    assert to_mm(10, None, 25) is None


def test_features_fixed_dimension_and_finite():
    img = synthetic(80)
    bgra = np.dstack([cv2.resize(img, (128, 128)), np.full((128, 128), 255, np.uint8)])
    v = extract_features(bgra)
    assert v.shape == (len(FEATURE_NAMES),) and np.isfinite(v).all()


def test_split_has_no_group_leakage():
    import pandas as pd
    from pathlib import Path
    p = Path("data/processed/splits.csv")
    if not p.exists():
        pytest.skip("splits.csv no generado")
    s = pd.read_csv(p)
    assert (s.groupby("fruit_id")["split"].nunique() == 1).all()
    assert {"train", "val", "test"} <= set(s["split"])
