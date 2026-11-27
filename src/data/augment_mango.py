"""Aumento dirigido para la especie (v2.2): mangos rojo-amarillos sintéticos.

Los mangos de entrenamiento (Fruit grade, FruitVision, Fruits-360) son casi todos verdes o amarillos; un mango tipo
Tommy (rojo con amarillo, el común en Colombia) se clasificaba como manzana. A cada mango de TRAIN se le crean dos
variantes desplazando el tono SOLO en los píxeles de la fruta (máscara del canal alfa): una naranja-amarilla uniforme
y una con degradado rojo -> amarillo en una dirección al azar. Solo entran al entrenamiento de la cabeza de especie;
validación y test no se tocan. Salida: data/interim/aug_mango/<image_id>_a{1,2}.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _shift(bgra: np.ndarray, shift_map: np.ndarray, sat: float = 1.15) -> np.ndarray:
    bgr, a = bgra[:, :, :3], bgra[:, :, 3] > 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = (hsv[:, :, 0] + shift_map) % 180                      # OpenCV: tono 0-180 (2 grados por unidad)
    hsv[:, :, 0] = np.where(a, h, hsv[:, :, 0])
    hsv[:, :, 1] = np.where(a, np.clip(hsv[:, :, 1] * sat, 0, 255), hsv[:, :, 1])
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return np.dstack([out, bgra[:, :, 3]])


def augment(image_ids, seg_dir: str, out_dir: str, seed: int = 42):
    rng = np.random.RandomState(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    made = []
    for i in image_ids:
        bgra = cv2.imread(str(Path(seg_dir) / f"{i}.png"), cv2.IMREAD_UNCHANGED)
        if bgra is None or bgra.shape[2] < 4:
            continue
        H, W = bgra.shape[:2]
        cv2.imwrite(str(out / f"{i}_a1.png"), _shift(bgra, np.full((H, W), -rng.uniform(12, 20))))
        ang = rng.uniform(0, 2 * np.pi)
        yy, xx = np.mgrid[0:H, 0:W]
        t = ((xx - W / 2) * np.cos(ang) + (yy - H / 2) * np.sin(ang)) / (max(H, W) / 2)   # -1..1
        t = (np.clip(t, -1, 1) + 1) / 2
        cv2.imwrite(str(out / f"{i}_a2.png"), _shift(bgra, -(34 - 22 * t)))                 # rojo -> amarillo
        made += [f"{i}_a1", f"{i}_a2"]
    return made
