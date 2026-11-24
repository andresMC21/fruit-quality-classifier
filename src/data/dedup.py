"""Detección de casi-duplicados para evitar fuga de datos entre train y test.

Las bases públicas contienen varias fotos muy parecidas de la misma fruta física (ráfagas de la cámara).
Si caen en splits distintos, el test mide memoria y no generalización.

Dos imágenes de la misma carpeta (fuente + clase) se unen en un mismo grupo (union-find) si:
  (a) distancia de Hamming de pHash <= dedup.hamming_threshold, o
  (b) fueron tomadas con <= dedup.burst_seconds entre sí (ráfaga, por la marca de tiempo del nombre de archivo)
      y su distancia de pHash <= dedup.burst_hamming.
La regla (b) es deliberadamente conservadora (agrupa de más): se prefiere perder variedad en train a filtrar fuga.
Salida: columna `dup_group`, usada por split.py como unidad de partición.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def phash_bits(path: str, hash_size: int = 8, highfreq: int = 4) -> np.ndarray:
    """pHash de 64 bits como vector booleano (DCT sobre imagen gris 32x32)."""
    import imagehash
    im = Image.open(path)
    im.draft("L", (256, 256))                      # decodificación JPEG reducida: acelera fotos de 12 MP
    return imagehash.phash(im.convert("L"), hash_size=hash_size, highfreq_factor=highfreq).hash.flatten()


def compute_hashes(paths, root: str = ".") -> np.ndarray:
    return np.stack([phash_bits(str(Path(root) / p)) for p in paths])


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def compute_groups(df: pd.DataFrame, hashes: np.ndarray, hamming_threshold: int = 6,
                   burst_seconds: float = 3, burst_hamming: int = 18) -> np.ndarray:
    """Devuelve un id de grupo por fila de df (mismo orden). df necesita: source, orig_label, timestamp."""
    uf = _UF(len(df))
    ts = pd.to_datetime(df["timestamp"])
    for _, idx in df.groupby(["source", "orig_label", "fruit_species"]).indices.items():
        h = hashes[idx].astype(np.uint8)
        d = (h[:, None, :] != h[None, :, :]).sum(-1)           # Hamming por pares dentro de la carpeta
        for a, b in np.argwhere(np.triu(d <= hamming_threshold, 1)):
            uf.union(idx[a], idx[b])
        # ráfagas: vecinos consecutivos en el tiempo
        t = ts.iloc[idx]
        order = idx[np.argsort(t.values)] if t.notna().any() else []
        for i, j in zip(order[:-1], order[1:]):
            if pd.notna(ts.iloc[i]) and pd.notna(ts.iloc[j]):
                gap = abs((ts.iloc[j] - ts.iloc[i]).total_seconds())
                if gap <= burst_seconds and (hashes[i] != hashes[j]).sum() <= burst_hamming:
                    uf.union(i, j)
    roots = np.array([uf.find(i) for i in range(len(df))])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def add_dup_groups(df: pd.DataFrame, cfg: dict, root: str = ".", cache: str | None = "data/interim/phash.npz",
                   deadline: float | None = None):
    """Añade `dup_group`. El caché de hashes se indexa por nombre de archivo: agregar especies no recalcula."""
    cache_p = Path(root) / cache if cache else None
    known = {}
    if cache_p is not None and cache_p.exists():
        z = np.load(cache_p, allow_pickle=False)
        known = dict(zip(z["names"].tolist(), z["hashes"]))
    missing = [f for f in df["filename"] if f not in known]
    if missing:
        import time
        for k in range(0, len(missing), 200):                    # por lotes: reanudable si se acaba el tiempo
            for f, h in zip(missing[k:k + 200], compute_hashes(missing[k:k + 200], root)):
                known[f] = h
            if cache_p is not None:
                cache_p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache_p, names=np.array(list(known)), hashes=np.stack(list(known.values())))
            if deadline is not None and time.time() > deadline and len(known) < len(df):
                raise TimeoutError(f"hashes {len(known)}/{len(df)}: repetir el comando para continuar")
    hashes = np.stack([known[f] for f in df["filename"]])
    c = cfg["dedup"]
    gid = compute_groups(df, hashes, c["hamming_threshold"], c.get("burst_seconds", 0), c.get("burst_hamming", 0))
    out = df.copy()
    out["dup_group"] = [f"{s[:3].upper()}_{sp[:3]}_g{g:05d}" for s, sp, g in zip(out["source"], out["fruit_species"], gid)]
    return out, hashes
