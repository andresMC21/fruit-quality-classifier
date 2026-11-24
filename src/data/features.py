"""Extracción de características manuales para ML clásico y MLP.

Entrada: recorte BGRA de 128x128 generado por preprocess.py (alfa = máscara de la fruta).
Todas las estadísticas de color/textura se calculan SOLO sobre los píxeles de la máscara, así el fondo
(papel, pasto, manos) no contamina el vector.

Vector (dimensión fija, ver FEATURE_NAMES):
  - Histogramas HSV normalizados: H (16), S (8), V (8)                         -> 32
  - Momentos de color en Lab: media y desviación de L, a, b                    -> 6
  - Fracción de píxeles "oscuros" (V<0.25), "saturados" (S>0.6) y "marrones"   -> 3   (manchas/podredumbre)
  - LBP uniforme (P=8, R=1) sobre gris dentro de la máscara                    -> 10
  - GLCM (contraste, homogeneidad, energía, correlación; distancia 1)          -> 4
  - Forma: relleno del recorte, circularidad, relación de aspecto, solidez     -> 4
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

H_BINS, S_BINS, V_BINS, LBP_P = 16, 8, 8, 8
FEATURE_NAMES = ([f"h{i}" for i in range(H_BINS)] + [f"s{i}" for i in range(S_BINS)] + [f"v{i}" for i in range(V_BINS)]
                 + [f"{c}_{m}" for c in "Lab" for m in ("mean", "std")]
                 + ["frac_dark", "frac_saturated", "frac_brown"]
                 + [f"lbp{i}" for i in range(LBP_P + 2)]
                 + ["glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_correlation"]
                 + ["fill", "circularity", "aspect", "solidity"])


def split_bgra(img):
    """(BGR, máscara bool). Si no hay canal alfa se asume que toda la imagen es fruta."""
    if img.ndim == 3 and img.shape[2] == 4:
        return img[..., :3], img[..., 3] > 0
    return img, np.ones(img.shape[:2], bool)


def shape_features(mask: np.ndarray) -> list:
    m = mask.astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return [0.0, 0.0, 1.0, 0.0]
    c = max(cnts, key=cv2.contourArea)
    area, per = cv2.contourArea(c), cv2.arcLength(c, True)
    hull_a = cv2.contourArea(cv2.convexHull(c))
    x, y, w, h = cv2.boundingRect(c)
    return [float(mask.mean()), float(4 * np.pi * area / per ** 2) if per > 0 else 0.0,
            float(max(w, h) / max(1, min(w, h))), float(area / hull_a) if hull_a > 0 else 0.0]


def extract_features(img, mask=None) -> np.ndarray:
    bgr, m = split_bgra(img)
    if mask is not None:
        m = mask > 0
    if m.sum() < 30:                                   # máscara degenerada: usar todo el recorte
        m = np.ones(bgr.shape[:2], bool)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    f = []
    for ch, bins, rng in ((0, H_BINS, 180), (1, S_BINS, 256), (2, V_BINS, 256)):
        h = np.histogram(hsv[..., ch][m], bins=bins, range=(0, rng))[0].astype(np.float32)
        f += list(h / max(1, h.sum()))
    for ch in range(3):
        v = lab[..., ch][m]
        f += [float(v.mean()), float(v.std())]
    hh, ss, vv = hsv[..., 0][m], hsv[..., 1][m] / 255.0, hsv[..., 2][m] / 255.0
    f += [float((vv < 0.25).mean()), float((ss > 0.6).mean()),
          float(((hh >= 5) & (hh <= 25) & (vv < 0.55) & (ss > 0.3)).mean())]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, 1, method="uniform")[m]
    h = np.bincount(lbp.astype(int), minlength=LBP_P + 2).astype(np.float32)
    f += list(h / max(1, h.sum()))
    q = (gray // 16).astype(np.uint8)                  # 16 niveles
    q = np.where(m, q, 0)
    glcm = graycomatrix(q, [1], [0, np.pi / 2], levels=16, symmetric=True, normed=True)
    f += [float(graycoprops(glcm, p).mean()) for p in ("contrast", "homogeneity", "energy", "correlation")]
    f += shape_features(m)
    return np.asarray(f, np.float32)


def build_feature_matrix(image_ids, seg_dir: str):
    """Matriz (n, d) leyendo los PNG de data/interim/segmented/."""
    from pathlib import Path
    X = []
    for i in image_ids:
        img = cv2.imread(str(Path(seg_dir) / f"{i}.png"), cv2.IMREAD_UNCHANGED)
        X.append(extract_features(img))
    return np.stack(X)
