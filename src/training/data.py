"""Carga de datos para entrenamiento: unión splits + anotaciones + características (con caché)."""
from __future__ import annotations

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import FEATURE_NAMES, extract_features
import cv2


def _feat(args):
    path = args
    return extract_features(cv2.imread(path, cv2.IMREAD_UNCHANGED))


def load_table(cfg: dict) -> pd.DataFrame:
    sp = pd.read_csv(cfg["split"]["output"])
    ann = pd.read_csv(cfg["paths"]["annotations"])[["image_id", "filename", "size_class", "diameter_mm"]]
    seg = pd.read_csv(Path(cfg["paths"]["interim"]) / "segmentation.csv")[["image_id", "method", "rel_diameter", "d_eq_px", "scale"]]
    return sp.merge(ann, on="image_id").merge(seg, on="image_id").reset_index(drop=True)


def feature_matrix(df: pd.DataFrame, cfg: dict, workers: int = 2, deadline: float | None = None) -> np.ndarray:
    """Matriz de características alineada con df; se cachea por image_id en data/processed/features.npz.
    Se calcula por lotes (reanudable): si se pasa `deadline`, lanza TimeoutError y basta con repetir."""
    import time
    cache = Path(cfg["paths"]["processed"]) / "features.npz"
    known = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["ids"].tolist(), z["X"]))
    seg_dir = Path(cfg["paths"]["interim"]) / "segmented"
    missing = [i for i in df["image_id"] if i not in known]
    with Pool(workers) as p:
        for k in range(0, len(missing), 500):
            part = missing[k:k + 500]
            for i, v in zip(part, p.map(_feat, [str(seg_dir / f"{i}.png") for i in part], chunksize=32)):
                known[i] = v
            np.savez(cache, ids=np.array(list(known)), X=np.stack(list(known.values())))
            if deadline is not None and time.time() > deadline and k + 500 < len(missing):
                raise TimeoutError(f"características {k + 500}/{len(missing)}: repetir para continuar")
    return np.stack([known[i] for i in df["image_id"]])


def species_onehot(df: pd.DataFrame, species: list[str]) -> np.ndarray:
    return np.stack([(df["fruit_species"] == s).to_numpy(np.float32) for s in species], axis=1)


def has_quality(df: pd.DataFrame) -> np.ndarray:
    """Filas con etiqueta de calidad (Fruits-360 solo aporta especie)."""
    return df["quality_class"].notna().to_numpy() & (df["quality_class"].astype(str) != "")


def load_xy(cfg: dict, deadline: float | None = None):
    """(df, X, Xs) con X = características visuales y Xs = X + one-hot de especie (la clase other queda en ceros)."""
    df = load_table(cfg)
    X = feature_matrix(df, cfg, deadline=deadline)
    Xs = np.hstack([X, species_onehot(df, cfg["fruit_species"])])
    return df, X, Xs
