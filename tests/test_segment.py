import cv2
import numpy as np
import pytest

from src.data.segment import (background_distance, find_reference_square, foreground_mask,
                              segment_image, touches_border)

BG = 128
FRUIT = (40, 60, 200)  # BGR


def noisy(img, sigma=6.0, seed=0):
    # Ruido leve y determinista: con imagenes perfectamente binarias Otsu devuelve t=0 y
    # foreground_mask descarta todo (ver test_imagen_binaria_perfecta_bug).
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def canvas(h=600, w=600, bg=BG):
    return np.full((h, w, 3), bg, np.uint8)


def disc(img, c, r, color=FRUIT):
    cv2.circle(img, c, r, color, -1)
    return img


def test_background_distance_cero_en_fondo_uniforme():
    d = background_distance(canvas())
    assert d.shape == (600, 600) and d.max() < 1e-3


def test_background_distance_alta_en_fruta():
    img = disc(canvas(), (300, 300), 100)
    d = background_distance(img)
    assert d[300, 300] > 30 and d[5, 5] < 1e-3


def test_foreground_mask_uniforme_vacia():
    assert foreground_mask(noisy(canvas())).sum() == 0


def test_touches_border():
    assert touches_border(0, 10, 20, 20, (100, 100))
    assert touches_border(10, 0, 20, 20, (100, 100))
    assert touches_border(80, 10, 20, 20, (100, 100))
    assert touches_border(10, 80, 20, 20, (100, 100))
    assert not touches_border(10, 10, 20, 20, (100, 100))


def test_disco_una_fruta_area():
    r = 150
    res = segment_image(noisy(disc(canvas(), (300, 300), r)))
    assert len(res) == 1
    assert abs(res[0][2] - np.pi * r * r) / (np.pi * r * r) < 0.02


def test_brillo_interno_sin_hueco():
    img = disc(canvas(), (300, 300), 150)
    cv2.circle(img, (330, 270), 30, (255, 255, 255), -1)
    (crop, mask, area), = segment_image(noisy(img))
    assert abs(area - np.pi * 150 ** 2) / (np.pi * 150 ** 2) < 0.02
    assert mask[mask.shape[0] // 2 - 30 + 0, mask.shape[1] // 2 + 30] == 1
    assert (mask == 0).sum() > 0  # esquinas fuera del disco
    yy, xx = np.nonzero(mask == 0)
    # ningun cero dentro del disco (hueco): todos los ceros estan lejos del centro
    cy, cx = mask.shape[0] / 2, mask.shape[1] / 2
    assert (np.hypot(yy - cy, xx - cx) > 140).all()


def test_sombra_no_incluida():
    img = canvas()
    cv2.circle(img, (310, 310), 170, (100, 100, 100), -1)  # sombra desplazada
    disc(img, (300, 300), 150)
    res = segment_image(noisy(img))
    assert len(res) >= 1
    area = np.pi * 150 ** 2
    assert abs(res[0][2] - area) / area < 0.05


def test_imagen_uniforme_lista_vacia():
    assert segment_image(noisy(canvas())) == []


def test_referencia_cuadrado_lado():
    img = canvas()
    L = 80
    cv2.rectangle(img, (100, 100), (100 + L - 1, 100 + L - 1), (0, 0, 0), -1)
    ref = find_reference_square(noisy(img))
    assert ref is not None
    assert abs(ref[0] - L) / L < 0.01


def test_referencia_no_es_fruta():
    img = canvas()
    cv2.rectangle(img, (50, 50), (129, 129), (0, 0, 0), -1)
    disc(img, (380, 360), 120)
    res = segment_image(noisy(img))
    assert len(res) == 1
    assert abs(res[0][2] - np.pi * 120 ** 2) / (np.pi * 120 ** 2) < 0.03


def test_fondo_oscuro_grande_no_es_referencia_y_fruta_detectada():
    # regresion: fondo oscuro que cubre casi toda la imagen, con margen claro y fruta clara
    img = canvas(bg=220)
    img[10:590, 10:590] = 20
    disc(img, (300, 300), 100, (200, 230, 240))
    assert find_reference_square(noisy(img)) is None
    res = segment_image(noisy(img))
    assert len(res) >= 1


def test_fondo_oscuro_completo_fruta_clara():
    img = canvas(bg=20)
    disc(img, (300, 300), 100, (200, 230, 240))
    assert find_reference_square(noisy(img)) is None
    res = segment_image(noisy(img))
    assert len(res) == 1
    assert abs(res[0][2] - np.pi * 100 ** 2) / (np.pi * 100 ** 2) < 0.03


def test_dos_discos_ordenados():
    img = canvas(600, 900)
    disc(img, (200, 300), 80)
    disc(img, (600, 300), 140)
    res = segment_image(noisy(img))
    assert len(res) == 2
    assert res[0][2] > res[1][2]


def test_borde_va_despues_aunque_sea_mayor():
    img = canvas(600, 900)
    disc(img, (150, 300), 90)       # no toca borde
    disc(img, (800, 300), 140)      # toca borde derecho (800+140 > 900), mayor
    res = segment_image(noisy(img))
    assert len(res) == 2
    assert res[0][2] < res[1][2]
    assert res[0][1].shape[1] < 200  # el pequeno primero


def test_recorte_y_mascara_misma_forma():
    img = canvas(600, 900)
    disc(img, (200, 300), 80)
    disc(img, (600, 300), 140)
    for crop, mask, area in segment_image(noisy(img)):
        assert crop.shape[:2] == mask.shape
        assert mask.sum() == area


def test_imagen_binaria_perfecta():
    """Regresion: sin ruido Otsu devuelve t=0; antes la mascara salia vacia."""
    assert len(segment_image(disc(canvas(), (300, 300), 150))) == 1
