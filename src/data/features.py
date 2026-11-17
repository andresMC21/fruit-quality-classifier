"""Extracción de características manuales para ML clásico/MLP:
histogramas HSV, textura (LBP/GLCM), forma (área, circularidad, aspect ratio).

Vector de 54 valores, calculado solo dentro de la mascara de la fruta (si no hay mascara, en toda
la imagen). Justificacion en docs/formulacion_matematica.md, seccion 3.2:
- Color (41): histograma HSV normalizado (H 16 bins, S 8, V 8) y momentos de Lab (media, desviacion,
  asimetria por canal). Las manchas y la podredumbre cambian tono y luminosidad.
- Textura (10): histograma LBP uniforme (P=8, R=1). Los golpes y el moho rompen la textura lisa de la piel.
- Forma (3): circularidad 4*pi*A/P^2, solidez A/A_casco y relacion de aspecto. No se usa el area:
  el tamano no es calidad y en Kaggle depende de la distancia a la camara.
La estandarizacion z = (x - mu_train) / sigma_train se hace despues, dentro del pipeline de sklearn,
para que mu y sigma salgan solo de train.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

WORK_SIZE = 128          # lado mayor al que se reduce el recorte, para que LBP sea comparable
H_BINS, S_BINS, V_BINS = 16, 8, 8
LBP_P, LBP_R = 8, 1


def feature_names():
    names = [f"h{i}" for i in range(H_BINS)] + [f"s{i}" for i in range(S_BINS)] + [f"v{i}" for i in range(V_BINS)]
    names += [f"{stat}_{ch}" for ch in "Lab" for stat in ("mean", "std", "skew")]
    names += [f"lbp{i}" for i in range(LBP_P + 2)]
    names += ["circularity", "solidity", "aspect_ratio"]
    return names


def _resize(img, mask):
    """Lleva el lado mayor a WORK_SIZE (tambien agranda): LBP mira vecinos a 1 px, asi que solo es
    comparable entre recortes si todos estan a la misma escala."""
    scale = WORK_SIZE / max(img.shape[:2])
    size = (max(1, round(img.shape[1] * scale)), max(1, round(img.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interp), cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)


def _largest_component(mask):
    """Color, textura y forma deben describir la misma region: la componente mas grande."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 2:
        return mask
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)


def _color(img, m):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[m]
    feats = []
    for ch, bins, top in ((0, H_BINS, 180), (1, S_BINS, 256), (2, V_BINS, 256)):
        hist, _ = np.histogram(hsv[:, ch], bins=bins, range=(0, top))
        feats.append(hist / hist.sum())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[m].astype(np.float64)
    mean, std = lab.mean(axis=0), lab.std(axis=0)
    skew = ((lab - mean) ** 3).mean(axis=0) / np.maximum(std, 1e-6) ** 3
    feats.append(np.column_stack([mean, std, skew]).ravel())   # L, a, b x (media, desv, asimetria)
    return np.concatenate(feats)


def _texture(img, m):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_P, LBP_R, method="uniform")   # valores 0..P+1
    # En el borde de la mascara los vecinos son fondo y darian "textura" falsa: se erosiona LBP_R px.
    inner = cv2.erode(m.astype(np.uint8), np.ones((2 * LBP_R + 1,) * 2, np.uint8)).astype(bool)
    if inner.sum() >= 10:
        m = inner
    hist, _ = np.histogram(lbp[m], bins=LBP_P + 2, range=(0, LBP_P + 2))
    return hist / hist.sum()


def _shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    hull_area = cv2.contourArea(cv2.convexHull(c))
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return np.array([
        4 * np.pi * area / max(perim ** 2, 1e-6),
        area / max(hull_area, 1e-6),
        min(w, h) / max(w, h, 1e-6),
    ])


def extract_features(img, mask=None):
    """img: BGR (uint8). mask: uint8 0/1 del mismo tamano o None. Devuelve np.ndarray de 54 valores."""
    if mask is None:
        mask = np.ones(img.shape[:2], np.uint8)
    mask = _largest_component(mask.astype(np.uint8))
    if mask.sum() < 10:
        raise ValueError("Mascara vacia o casi vacia: no se pueden extraer caracteristicas.")
    # La forma no depende de la escala y se mide en la mascara original: al agrandar con vecino
    # mas cercano el borde queda escalonado y el perimetro (y la circularidad) se distorsiona.
    shape = _shape(mask)
    img, mask = _resize(img, mask)
    m = mask.astype(bool)
    return np.concatenate([_color(img, m), _texture(img, m), shape])
