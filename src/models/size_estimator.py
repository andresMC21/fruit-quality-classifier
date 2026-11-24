"""Estimación de tamaño a partir de la máscara de segmentación.

1. d_px = 2 * sqrt(A / pi)           diámetro equivalente (A = área de la máscara en píxeles)
2. d_mm = d_px * D_ref / d_ref_px    si hay un objeto de referencia redondo de diámetro conocido en el mismo plano
3. Clase pequeño/mediano/grande por umbrales = terciles del conjunto de TRAIN, por especie y por fuente.

Sin objeto de referencia (bases públicas) el resultado es SOLO RELATIVO: d_px / lado corto de la imagen depende de
la distancia a la cámara y no mide milímetros. Solo las fotos propias con referencia dan MAE en mm.
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE_NAMES = ("pequeno", "mediano", "grande")


def equivalent_diameter_px(mask) -> float:
    return 2.0 * float(np.sqrt((np.asarray(mask) > 0).sum() / np.pi))


def to_mm(d_px: float, ref_diameter_px: float | None, ref_diameter_mm: float | None):
    if not ref_diameter_px or not ref_diameter_mm:
        return None
    return float(d_px * ref_diameter_mm / ref_diameter_px)


def size_class(d: float, thresholds: tuple[float, float]) -> str:
    lo, hi = thresholds
    return SIZE_NAMES[0] if d < lo else (SIZE_NAMES[1] if d < hi else SIZE_NAMES[2])


def terciles(values) -> tuple[float, float]:
    """Umbrales (33 % y 67 %) calculados SOLO con datos de entrenamiento."""
    v = np.asarray(values, float)
    return float(np.percentile(v, 100 / 3)), float(np.percentile(v, 200 / 3))


def find_reference(img_bgr, fruit_mask, min_frac: float = 0.002, min_circularity: float = 0.82):
    """Busca un objeto redondo (moneda/tapa) distinto de la fruta. Devuelve su diámetro equivalente en px o None.

    Heurística: distancia en Lab al fondo -> Otsu -> componentes fuera de la fruta -> la más circular.
    Requiere fondo uniforme (protocolo de docs/guia_anotacion.md).
    """
    H, W = img_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(2, int(0.04 * min(H, W)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3), lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    d = np.linalg.norm(lab - np.median(border, axis=0), axis=2)
    d = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    grown = cv2.dilate((fruit_mask > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    m[grown > 0] = 0
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_c = None, min_circularity
    for c in cnts:
        a, p = cv2.contourArea(c), cv2.arcLength(c, True)
        if a < min_frac * H * W or p == 0:
            continue
        circ = 4 * np.pi * a / p ** 2
        if circ >= best_c:
            best, best_c = 2.0 * np.sqrt(a / np.pi), circ
    return best
