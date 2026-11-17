import numpy as np

from src.data.preprocess import preprocess


def test_shape_dtype_range():
    img = np.random.default_rng(0).integers(0, 256, (90, 200, 3), dtype=np.uint8)
    out = preprocess(img, size=(64, 32))   # (ancho, alto)
    assert out.dtype == np.float32
    assert out.shape == (32, 64, 3)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_default_size():
    assert preprocess(np.zeros((10, 20, 3), np.uint8)).shape == (128, 128, 3)


def test_white_maps_to_one():
    out = preprocess(np.full((20, 20, 3), 255, np.uint8), size=(8, 8))
    assert np.allclose(out, 1.0)


def test_bgr_to_rgb():
    img = np.zeros((16, 16, 3), np.uint8)
    img[..., 0] = 255   # azul puro en BGR
    out = preprocess(img, size=(8, 8))
    assert np.allclose(out[..., 2], 1.0)
    assert np.allclose(out[..., 0], 0.0) and np.allclose(out[..., 1], 0.0)
