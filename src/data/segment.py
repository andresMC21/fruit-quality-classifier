"""Fase 3: segmentación de frutas individuales (bases públicas, mixed_quality y fotos propias).

Estrategia (de menos a más costosa):
 1. Distancia en Lab al color de fondo (mediana del borde) + umbral de Otsu + morfología + componente
    más grande y centrada + relleno de huecos. Funciona con fondo uniforme (Fruit grade, fotos propias).
 2. Si la máscara no pasa los controles de calidad (área, solidez), GrabCut inicializado con un rectángulo
    central (fondos variados de Kaggle).
 3. Si GrabCut tampoco pasa, `ok=False`: se reporta la tasa de fallo y la imagen se descarta del
    entrenamiento de los modelos de forma (se documenta en docs/03_preparacion_datos.md).
`segment_multi` separa varias frutas en una misma foto (mixed_quality, sobre pasto).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

L_WEIGHT = 0.45
WORK_SIDE = 384   # lado máximo de trabajo: acelera fotos de 12 MP sin perder precisión útil


@dataclass
class SegResult:
    mask: np.ndarray        # uint8 {0,255} a la resolución de entrada
    bbox: tuple             # x, y, w, h en píxeles de entrada
    area_px: float          # área de la máscara en píxeles de la imagen de entrada
    method: str             # 'otsu' | 'grabcut' | 'fail'
    ok: bool
    solidity: float
    area_frac: float


def _shrink(img, side=WORK_SIDE):
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    return (cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img), s


def _largest_central(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape
    best, best_score = 0, -1.0
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        d = np.hypot(cent[i][0] / w - 0.5, cent[i][1] / h - 0.5)
        score = a * (1.0 - min(d, 0.7))
        if score > best_score:
            best, best_score = i, score
    return np.where(lab == best, 255, 0).astype(np.uint8)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    cv2.drawContours(out, cnts, -1, 255, thickness=cv2.FILLED)
    return out


def _clean(mask: np.ndarray) -> np.ndarray:
    k = max(3, (min(mask.shape) // 60) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ker)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    return _fill_holes(_largest_central(mask))


def _solidity(mask: np.ndarray) -> float:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    ha = cv2.contourArea(hull)
    return float(cv2.contourArea(c) / ha) if ha > 0 else 0.0


def _bg_distance(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, int(0.04 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    diff = lab - bg
    diff[..., 0] *= L_WEIGHT                     # la luminancia pesa menos: las sombras cambian L, no el color
    d = np.linalg.norm(diff, axis=2)
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _otsu_mask(img_bgr: np.ndarray) -> np.ndarray:
    d = _bg_distance(img_bgr)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _clean(m)


def _grabcut_mask(img_bgr: np.ndarray, iters: int = 4) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    # semilla: región central probablemente fruta, borde seguramente fondo, núcleo del centro seguramente fruta
    m = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)] = cv2.GC_PR_FGD
    b = max(2, int(0.04 * min(h, w)))
    m[:b], m[-b:], m[:, :b], m[:, -b:] = cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_BGD
    m[int(0.4 * h):int(0.6 * h), int(0.4 * w):int(0.6 * w)] = cv2.GC_FGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img_bgr, m, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    return _clean(np.where((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD), 255, 0).astype(np.uint8))


def _check(mask: np.ndarray, min_frac=0.04, max_frac=0.85, min_solidity=0.88):
    frac = float((mask > 0).mean())
    sol = _solidity(mask)
    return (min_frac <= frac <= max_frac and sol >= min_solidity), sol, frac


# Controles de calidad por especie: el banano es curvo (solidez baja) y la lima es pequeña en el encuadre.
QC_BY_SPECIES = {"banana": dict(min_solidity=0.70), "lime": dict(min_frac=0.012, min_solidity=0.85),
                "grape": dict(min_solidity=0.60), "strawberry": dict(min_solidity=0.80)}


def segment_image(img_bgr: np.ndarray, allow_grabcut: bool = True, species: str | None = None) -> SegResult:
    """Segmenta UNA fruta. Devuelve máscara a la resolución de entrada."""
    qc = QC_BY_SPECIES.get(species, {})
    H, W = img_bgr.shape[:2]
    small, s = _shrink(img_bgr)
    best = None
    for name, fn in (("otsu", _otsu_mask), ("grabcut", _grabcut_mask)):
        if name == "grabcut" and not allow_grabcut:
            break
        m = fn(small)
        ok, sol, frac = _check(m, **qc)
        if best is None or (ok and not best[1]):
            best = (m, ok, name, sol, frac)
        if ok:
            break
    m, ok, name, sol, frac = best
    method = name if ok else "fail"
    mask = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) if s < 1 else m
    x, y, w, h = cv2.boundingRect(mask) if mask.any() else (0, 0, W, H)
    return SegResult(mask, (x, y, w, h), float((mask > 0).sum()), method, ok, sol, frac)


def segment_multi(img_bgr: np.ndarray, min_frac: float = 0.004, max_frac: float = 0.3):
    """Varias frutas sobre un fondo dominante (p. ej. pasto en mixed_quality). Devuelve lista de SegResult."""
    H, W = img_bgr.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img_bgr, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.median(lab.reshape(-1, 3), axis=0)         # el fondo ocupa la mayor parte de la imagen
    d = cv2.normalize(np.linalg.norm(lab - bg, axis=2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m = cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, (min(H, W) // 80) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (min_frac * H * W <= a <= max_frac * H * W):
            continue
        comp = _fill_holes(np.where(lab_img == i, 255, 0).astype(np.uint8))
        ok, sol, frac = _check(comp, min_frac=min_frac, max_frac=max_frac, min_solidity=0.8)
        x, y, w, h = cv2.boundingRect(comp)
        out.append(SegResult(comp, (x, y, w, h), float((comp > 0).sum()), "multi" if ok else "fail", ok, sol, frac))
    return out


def equivalent_diameter_px(area_px: float) -> float:
    """Diámetro de la circunferencia con la misma área que la máscara: d = 2 sqrt(A / pi)."""
    return 2.0 * float(np.sqrt(area_px / np.pi))
