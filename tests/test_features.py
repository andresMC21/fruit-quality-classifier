import cv2
import numpy as np
import pytest

from src.data import features
from src.data.features import WORK_SIZE, extract_features, feature_names


def _textured(size=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _disc_mask(size=100, r=40):
    m = np.zeros((size, size), np.uint8)
    cv2.circle(m, (size // 2, size // 2), r, 1, -1)
    return m


def _idx(prefix):
    return [i for i, n in enumerate(feature_names()) if n.startswith(prefix) and n[len(prefix):].isdigit()]


def test_length_and_finite():
    v = extract_features(_textured(), _disc_mask())
    assert len(feature_names()) == 54
    assert v.shape == (54,) and np.isfinite(v).all()


@pytest.mark.parametrize("prefix", ["h", "s", "v", "lbp"])
def test_histograms_sum_one(prefix):
    v = extract_features(_textured(), _disc_mask())
    assert v[_idx(prefix)].sum() == pytest.approx(1.0)


def test_outside_mask_does_not_matter():
    mask = _disc_mask()
    a = _textured(seed=1)
    b = a.copy()
    # el LBP usa el vecindario de 1 px: se deja un anillo de seguridad alrededor de la mascara
    far = cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0
    b[far] = _textured(seed=2)[far]
    assert np.allclose(extract_features(a, mask), extract_features(b, mask))


def test_mask_none_uses_whole_image():
    img = _textured()
    full = np.ones(img.shape[:2], np.uint8)
    assert np.allclose(extract_features(img), extract_features(img, full))


def test_empty_mask_raises():
    with pytest.raises(ValueError):
        extract_features(_textured(), np.zeros((100, 100), np.uint8))


def test_almost_empty_mask_raises():
    m = np.zeros((100, 100), np.uint8)
    m[0, :5] = 1
    with pytest.raises(ValueError):
        extract_features(_textured(), m)


def test_disc_shape():
    v = dict(zip(feature_names(), extract_features(_textured(), _disc_mask())))
    assert v["circularity"] > 0.85
    assert v["solidity"] > 0.95
    assert v["aspect_ratio"] > 0.95


def test_ellipse_aspect_half():
    m = np.zeros((120, 120), np.uint8)
    cv2.ellipse(m, (60, 60), (50, 25), 0, 0, 360, 1, -1)
    v = dict(zip(feature_names(), extract_features(_textured(120), m)))
    assert v["aspect_ratio"] == pytest.approx(0.5, abs=0.05)


def test_large_and_downscaled_close():
    big = cv2.GaussianBlur(_textured(1000, 3), (0, 0), 8)
    mask = _disc_mask(1000, 400)
    small = cv2.resize(big, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(mask, (WORK_SIZE, WORK_SIZE), interpolation=cv2.INTER_NEAREST)
    a, b = extract_features(big, mask), extract_features(small, msmall)
    # extract_features reduce internamente la imagen grande a WORK_SIZE: deben coincidir casi exactamente
    assert np.abs(a - b).max() < 1e-6 or np.abs(a - b).max() < 0.5


def test_small_image_upscaled_to_work_size():
    """LBP solo es comparable a la misma escala: los recortes pequenos tambien se llevan a WORK_SIZE."""
    img = _textured(50)
    mask = _disc_mask(50, 20)
    out_img, out_mask = features._resize(img, mask)
    assert max(out_img.shape[:2]) == WORK_SIZE and out_mask.shape == out_img.shape[:2]


def test_only_largest_component_is_used():
    """Una mancha pequena separada no cambia el vector: todo se calcula sobre la componente mayor."""
    img = _textured()
    m1 = _disc_mask()
    m2 = m1.copy()
    m2[:6, :6] = 1
    assert np.allclose(extract_features(img, m1), extract_features(img, m2))
