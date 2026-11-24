"""Fase 3: segmentación por lotes, recorte con fondo neutro y medidas de tamaño.

Por imagen se guarda en data/interim/segmented/<image_id>.png un recorte cuadrado de `image_size` con:
  - la fruta (fuera de la máscara -> gris 128) para quitar el atajo de la fuente (fondo, manos, papel);
  - margen del 8 % alrededor de la caja de la máscara, conservando la proporción.
La máscara va en el canal alfa del PNG (la usan las características de forma).
Y en data/interim/segmentation.csv: método, ok, área, diámetro equivalente (px) y relativo (d_eq / lado corto).
Si la segmentación falla (ok=False) se guarda un recorte central SIN enmascarar; esas filas se excluyen del
modelado y su tasa se reporta (docs/03_preparacion_datos.md).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.segment import equivalent_diameter_px, segment_image

GRAY = 128
MARGIN = 0.08


def load_bgr(path: str, max_side: int = 1600):
    """Lee la imagen reduciendo (fotos de 12 MP) y devuelve (img, escala respecto al original)."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2 if "Fruit grade" in path else cv2.IMREAD_COLOR)
    return img


def masked_square_crop(img_bgr, mask, bbox, size=(128, 128), apply_mask=True):
    """Devuelve (recorte BGR con fondo gris, máscara uint8 del recorte)."""
    x, y, w, h = bbox
    side = int(max(w, h) * (1 + 2 * MARGIN))
    cx, cy = x + w // 2, y + h // 2
    x0, y0 = cx - side // 2, cy - side // 2
    out = np.full((side, side, 3), GRAY, np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(img_bgr.shape[1], x0 + side), min(img_bgr.shape[0], y0 + side)
    patch = img_bgr[sy0:sy1, sx0:sx1].copy()
    m = mask[sy0:sy1, sx0:sx1] > 0
    if apply_mask:
        patch[~m] = GRAY
    out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = patch
    mfull = np.zeros((side, side), np.uint8)
    mfull[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = np.where(m, 255, 0)
    return (cv2.resize(out, size, interpolation=cv2.INTER_AREA),
            cv2.resize(mfull, size, interpolation=cv2.INTER_NEAREST))


def center_square_crop(img_bgr, size=(128, 128)):
    h, w = img_bgr.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return (cv2.resize(img_bgr[y0:y0 + s, x0:x0 + s], size, interpolation=cv2.INTER_AREA),
            np.full(size[::-1], 255, np.uint8))


def _job(args):
    image_id, path, out_dir, size, species = args
    out = Path(out_dir) / f"{image_id}.png"
    # fotos de 12 MP: decodificación JPEG reducida a 1/4 (≈4x más rápida); `scale` permite volver a píxeles originales
    big = "Fruit grade" in path or "Sultana" in path
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4 if big else cv2.IMREAD_COLOR)
    scale = 4 if big else 1
    if img is None:
        return dict(image_id=image_id, method="unreadable", ok=False)
    H, W = img.shape[:2]
    s = segment_image(img, species=species)
    crop, cmask = masked_square_crop(img, s.mask, s.bbox, size) if s.ok else center_square_crop(img, size)
    cv2.imwrite(str(out), np.dstack([crop, cmask]))        # PNG BGRA: alfa = máscara de la fruta
    d = equivalent_diameter_px(s.area_px)
    return dict(image_id=image_id, method=s.method, ok=s.ok, scale=scale, area_px=s.area_px, d_eq_px=d,
                rel_diameter=d / min(H, W), area_frac=s.area_frac, solidity=s.solidity,
                bbox_x=s.bbox[0], bbox_y=s.bbox[1], bbox_w=s.bbox[2], bbox_h=s.bbox[3], img_w=W, img_h=H)   # medidas en píxeles de la imagen leída; x scale = original


def process_all(df: pd.DataFrame, cfg: dict, root: str = ".", workers: int = 2, deadline: float | None = None):
    """Segmenta y recorta todo lo pendiente; reanudable (la tabla se guarda por lotes)."""
    import time
    out_dir = Path(root) / cfg["paths"]["interim"] / "segmented"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = Path(root) / cfg["paths"]["interim"] / "segmentation.csv"
    done = pd.read_csv(table) if table.exists() else pd.DataFrame(columns=["image_id"])
    todo = df[~df["image_id"].isin(done["image_id"])]
    size = tuple(cfg["image_size"])
    jobs = [(r.image_id, str(Path(root) / r.filename), str(out_dir), size, r.fruit_species) for r in todo.itertuples()]
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for k in range(0, len(jobs), 100):
            if deadline is not None and time.time() > deadline:
                break
            rows += list(ex.map(_job, jobs[k:k + 100], chunksize=4))
            pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(table, index=False)
    res = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    res.to_csv(table, index=False)
    print(f"segmentadas {len(res)}/{len(df)}", flush=True)
    return res
