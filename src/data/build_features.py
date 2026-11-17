"""Tabla de caracteristicas para los modelos clasicos (data/processed/features.csv, no versionado).

Para cada imagen de splits.csv: segmenta, toma la fruta principal y extrae las 54 caracteristicas.
seg_ok indica si la segmentacion pasa el control de calidad (circularidad >= segmentation.min_circularity):
por debajo, la mascara casi siempre incluye mano o fondo (docs/04_modelado.md). Los modelos clasicos
se entrenan solo con seg_ok; la CNN no usa esta tabla.
Ademas se guardan, con prefijo full_, color y textura de la imagen COMPLETA (sin mascara ni forma): es el
experimento de contraste para medir cuanto aprende un modelo del fondo y la mano.

Uso:
    python -m src.data.build_features
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.features import extract_features, feature_names
from src.data.segment import segment_image
from src.utils.config import load_config

RAW_DIR = Path("data/raw")
SHAPE = ("circularity", "solidity", "aspect_ratio")


def full_feature_names():
    """Color y textura de la imagen completa. Sin forma: seria la del rectangulo de la foto."""
    return ["full_" + n for n in feature_names() if n not in SHAPE]


def full_image_features(img):
    v = dict(zip(feature_names(), extract_features(img, None)))
    return [v[n] for n in feature_names() if n not in SHAPE]


def image_features(img, min_circularity):
    """(vector de 54 o None, circularidad, seg_ok) de una imagen BGR."""
    fruits = segment_image(img)
    if not fruits:
        return None, 0.0, False
    crop, mask, _ = fruits[0]
    v = extract_features(crop, mask)
    circularity = float(v[feature_names().index("circularity")])
    return v, circularity, circularity >= min_circularity


def build_feature_table(cfg):
    splits = pd.read_csv(cfg["split"]["output"])
    names = feature_names()
    rows = []
    for _, r in splits.iterrows():
        img = cv2.imread(str(RAW_DIR / r["filename"]))
        if img is None:
            raise FileNotFoundError(RAW_DIR / r["filename"])
        v, circ, ok = image_features(img, cfg["segmentation"]["min_circularity"])
        row = {k: r[k] for k in ("image_id", "source", "fruit_id", "quality_class", "split")}
        row.update(seg_ok=ok, circularity_qc=circ)
        row.update(dict(zip(names, v if v is not None else np.full(len(names), np.nan))))
        row.update(dict(zip(full_feature_names(), full_image_features(img))))
        rows.append(row)
    table = pd.DataFrame(rows)
    out = Path(cfg["paths"]["features"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"{len(table)} imagenes -> {out}")
    print(pd.crosstab([table["split"], table["seg_ok"]], table["quality_class"]))
    return table


if __name__ == "__main__":
    build_feature_table(load_config())
