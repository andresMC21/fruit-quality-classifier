import cv2
import numpy as np
import pytest

from src.data.segment import segment_image
from src.models.size_estimator import (equivalent_diameter_px, feret_diameter_px, fit_thresholds,
                                       size_class, to_mm)


def disc_mask(r, size=None):
    size = size or 2 * r + 20
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), r, 1, -1)
    return m


def test_equivalente_disco():
    assert equivalent_diameter_px(disc_mask(100)) == pytest.approx(200, rel=0.02)


def test_feret_disco():
    assert feret_diameter_px(disc_mask(100)) == pytest.approx(200, rel=0.02)


def test_feret_elipse_eje_mayor():
    m = np.zeros((300, 500), np.uint8)
    cv2.ellipse(m, (250, 150), (200, 80), 0, 0, 360, 1, -1)
    assert feret_diameter_px(m) == pytest.approx(400, rel=0.02)
    assert equivalent_diameter_px(m) < feret_diameter_px(m)


def test_feret_mascara_vacia():
    assert feret_diameter_px(np.zeros((10, 10), np.uint8)) == 0.0


def test_to_mm_sin_referencia():
    assert to_mm(300, None, None) is None
    assert to_mm(300, 0, 50) is None
    assert to_mm(300, 200, None) is None


def test_to_mm_con_referencia():
    assert to_mm(300, 200, 50) == pytest.approx(75)


def test_to_mm_perspectiva():
    assert to_mm(300, 200, 50, camera_height_mm=1e12) == pytest.approx(75, rel=1e-6)
    assert to_mm(300, 200, 50, camera_height_mm=500) == pytest.approx(75 / (1 + 75 / 1000))
    assert to_mm(300, 200, 50, camera_height_mm=500) == pytest.approx(69.77, abs=0.01)


def test_fit_thresholds_terciles():
    assert fit_thresholds([1, 2, 3, 4, 5, 6, 7]) == pytest.approx((3.0, 5.0))
    q1, q2 = fit_thresholds(range(0, 101))
    assert q1 == pytest.approx(100 / 3) and q2 == pytest.approx(200 / 3)


def test_size_class_rangos_y_bordes():
    t = (10.0, 20.0)
    assert size_class(5, t) == "pequeno"
    assert size_class(15, t) == "mediano"
    assert size_class(25, t) == "grande"
    assert size_class(10, t) == "mediano"
    assert size_class(20, t) == "grande"


def test_integracion_disco_y_referencia_75mm():
    img = np.full((1000, 1000, 3), 128, np.uint8)
    cv2.circle(img, (620, 520), 165, (100, 100, 100), -1)  # sombra
    cv2.circle(img, (600, 500), 150, (40, 60, 200), -1)    # fruta, 300 px
    cv2.rectangle(img, (60, 60), (259, 259), (0, 0, 0), -1)  # referencia 200 px = 50 mm
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    res = segment_image(img)
    assert len(res) >= 1
    d_px = equivalent_diameter_px(res[0][1])
    assert to_mm(d_px, 200, 50) == pytest.approx(75, rel=0.01)
