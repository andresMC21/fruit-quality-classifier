"""Detección de casi-duplicados para evitar fuga de datos entre train y test.

Las bases públicas contienen varias fotos muy parecidas de la misma fruta física (ráfagas de la cámara).
Si caen en splits distintos, el test mide memoria y no generalización.

Dos imágenes de la misma carpeta (fuente + clase) se unen en un mismo grupo (union-find) si:
  (a) distancia de Hamming de pHash <= dedup.hamming_threshold, o
  (b) fueron tomadas con <= dedup.burst_seconds entre sí (ráfaga, por la marca de tiempo del nombre de archivo)
      y su distancia de pHash <= dedup.burst_hamming.
Fuentes sin marca de tiempo y con varias fotos seguidas de la misma fruta girada (FruitVision): el pHash no une
vistas rotadas, así que la unidad de partición (`fruit_id`) es un TRAMO de ~dedup.sequence_chunk fotos consecutivas
(orden del nombre de archivo), con cada corte movido al mayor salto de pHash dentro de ±dedup.sequence_window
posiciones; luego se une con los grupos de pHash. Fruits-360: la unidad es la carpeta (una variedad, a menudo
una sola fruta fotografiada girando).
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
    from src.data.manifest import id_prefix
    out["dup_group"] = [f"{id_prefix(s)}_{sp[:3]}_g{g:05d}" for s, sp, g in zip(out["source"], out["fruit_species"], gid)]
    return out, hashes


def partition_units(df: pd.DataFrame, hashes: np.ndarray, cfg: dict):
    """(fruit_id, dup_group) por fila. Por defecto fruit_id = dup_group; ver docstring del módulo."""
    c = cfg["dedup"]
    seq_src = set(c.get("sequence_sources", []))
    chunk, win = c.get("sequence_chunk", 30), c.get("sequence_window", 5)
    unit = df["dup_group"].astype(str).copy()
    uf = _UF(len(df))
    for (src, lab, sp), idx in df.groupby(["source", "orig_label", "fruit_species"]).indices.items():
        if src in ("fruits360", "wgisd"):                 # unidad = carpeta (Fruits-360) o foto de viñedo (WGISD)
            for k in idx[1:]:
                uf.union(idx[0], k)
            continue
        if src not in seq_src:
            continue
        order = idx[np.argsort(df["filename"].to_numpy()[idx])]
        n = len(order)
        jump = np.array([(hashes[order[i]] != hashes[order[i + 1]]).sum() for i in range(n - 1)])
        cuts, b = [], chunk
        while b < n - chunk // 2:
            lo, hi = max(1, b - win), min(n - 1, b + win)
            cuts.append(lo + int(np.argmax(jump[lo - 1:hi])))    # corte tras el mayor salto de la ventana
            b = cuts[-1] + chunk
        seg = np.searchsorted(cuts, np.arange(n), side="right")
        for i in range(1, n):
            if seg[i] == seg[i - 1]:
                uf.union(order[i - 1], order[i])
    # unir también los grupos de duplicados (un duplicado lejano en la secuencia arrastra su tramo). En las fuentes
    # de secuencia la fruta es pequeña sobre fondo gris uniforme y el pHash casi no cambia entre frutas distintas
    # (encadenaba carpetas enteras), así que ahí el grupo de duplicados es el de ARCHIVOS idénticos (mismo MD5).
    is_seq = df["source"].isin(seq_src).to_numpy()
    dup = df["dup_group"].astype(str).copy()
    if is_seq.any():
        import hashlib
        from src.data.manifest import id_prefix
        first = {}
        for i in np.where(is_seq)[0]:
            r = df.iloc[i]
            h = hashlib.md5(Path(r["filename"]).read_bytes()).hexdigest()
            j = first.setdefault(h, i)
            dup.iloc[i] = f"{id_prefix(r['source'])}_{r['fruit_species'][:3]}_x{j:05d}"
    for _, idx in pd.Series(range(len(df))).groupby(dup.to_numpy()).indices.items():
        for k in idx[1:]:
            uf.union(idx[0], k)
    roots = np.array([uf.find(i) for i in range(len(df))])
    is_new = df["source"].isin(seq_src | {"fruits360", "wgisd"}).to_numpy()
    first = {}
    for i, r in enumerate(roots):
        first.setdefault(r, i)
    unit[is_new] = [_rename(unit.iloc[first[r]], "u") for r in roots[is_new]]
    return unit, dup


def _rename(gid: str, tag: str) -> str:
    """FVI_app_g00012 -> FVI_app_<tag>00012 (solo el último bloque)."""
    head, _, tail = gid.rpartition("_")
    return f"{head}_{tag}{tail[1:]}"


def cross_source_dups(df: pd.DataFrame, hashes: np.ndarray, sources, threshold: int = 6) -> np.ndarray:
    """Máscara de imágenes de `sources` con un casi-duplicado (pHash <= threshold) de la MISMA especie en otra
    fuente. Se excluyen (keep=False): podrían ser la misma foto redistribuida y filtrarse entre train y los tests."""
    out = np.zeros(len(df), bool)
    H = hashes.astype(np.uint8)
    for sp, idx in df.groupby("fruit_species").indices.items():
        src = df["source"].to_numpy()[idx]
        new, old = idx[np.isin(src, list(sources))], idx[~np.isin(src, list(sources))]
        if not len(new) or not len(old):
            continue
        d = (H[new][:, None, :] != H[old][None, :, :]).sum(-1)
        out[new[(d <= threshold).any(1)]] = True
    return out
