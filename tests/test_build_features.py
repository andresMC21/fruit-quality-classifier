import cv2
import numpy as np

from src.data import build_features as B


def disc_image():
    rng = np.random.default_rng(0)
    img = np.full((200, 200, 3), 235, np.uint8)
    cv2.circle(img, (100, 100), 60, (30, 40, 200), -1)
    noise = rng.integers(-4, 5, img.shape)
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def test_disc_segments_ok():
    v, circ, ok = B.image_features(disc_image(), 0.7)
    assert ok is True or ok == True
    assert v is not None and len(v) == 54
    assert circ >= 0.7


def test_uniform_image_no_fruit():
    img = np.full((120, 120, 3), 200, np.uint8)
    v, circ, ok = B.image_features(img, 0.7)
    assert v is None and circ == 0 and ok is False
