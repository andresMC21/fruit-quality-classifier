"""División train/val/test FIJA, AGRUPADA y estratificada.

- Unidad de partición = `fruit_id`: grupo de duplicados (bases públicas, ver dedup.py) o la fruta física (fotos propias).
  Un grupo nunca aparece en dos splits (se verifica con assert).
- Estratificación por especie x calidad (StratifiedGroupKFold).
- Test = fracción de grupos públicos (split.test_public) + fracción de frutas PROPIAS (split.test_own), para medir
  también la generalización a datos locales. Las fotos propias se reportan aparte en la evaluación.
- Solo entran imágenes con segmentación ok. El resultado se versiona (data/processed/splits.csv) y NO se regenera
  después de mirar el test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

OWN_SOURCES = {"own", "other_group"}


def _pick_fold(df: pd.DataFrame, frac: float, seed: int) -> np.ndarray:
    """Índices (posiciones) de ~frac de los grupos de df, estratificados por `strat`."""
    k = max(2, round(1 / frac))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(df, df["strat"], df["fruit_id"]))
    return held


def make_splits(cfg: dict, root: str = ".") -> pd.DataFrame:
    root = Path(root)
    ann = pd.read_csv(root / cfg["paths"]["annotations"])
    seg = pd.read_csv(root / cfg["paths"]["interim"] / "segmentation.csv")
    df = ann.merge(seg[["image_id", "ok", "method"]], on="image_id", how="inner")
    df = df[(df["ok"] == True) & (df["keep"] == True)].copy()           # noqa: E712
    df["fruit_id"] = df["fruit_id"].astype(str)
    df["strat"] = df["fruit_species"] + "|" + df["quality_class"]
    seed, sc = cfg["seed"], cfg["split"]
    df["split"] = "train"
    ext = sc.get("external")
    if ext:                                    # fuente externa en especies compartidas: nunca entra a train/val/test público
        is_ext = (df["source"] == ext["source"]) & df["fruit_species"].isin(ext["species"])
        df.loc[is_ext, "split"] = "external"
        df = df.copy()
    else:
        is_ext = pd.Series(False, index=df.index)

    df_all, df = df, df[~is_ext].copy()
    own = df[df["source"].isin(OWN_SOURCES)]
    pub = df[~df["source"].isin(OWN_SOURCES)].copy()

    # 1) test: fracción de grupos públicos
    pub = pub.reset_index()
    test_idx = _pick_fold(pub, sc["test_public"], seed)
    df.loc[pub.loc[test_idx, "index"], "split"] = "test"
    # 2) validación: fracción de los grupos públicos restantes
    rest = pub.drop(index=test_idx).reset_index(drop=True)
    val_idx = _pick_fold(rest, sc["val"] / (1 - sc["test_public"]), seed + 1)
    df.loc[rest.loc[val_idx, "index"], "split"] = "val"

    # 3) fotos propias: test_own de las frutas (por fruit_id), estratificado; el resto, train/val
    if len(own):
        own = own.reset_index()
        g = own.groupby("fruit_id").first()
        n_test = max(1, int(round(sc["test_own"] * len(g))))
        rng = np.random.RandomState(seed)
        test_ids = set(rng.permutation(g.index.to_numpy())[:n_test])
        df.loc[own.loc[own["fruit_id"].isin(test_ids), "index"], "split"] = "test"
        left = own[~own["fruit_id"].isin(test_ids)]
        val_ids = set(rng.permutation(left["fruit_id"].unique())[:max(1, int(0.2 * left["fruit_id"].nunique()))])
        df.loc[left.loc[left["fruit_id"].isin(val_ids), "index"], "split"] = "val"

    df = pd.concat([df, df_all[df_all["split"] == "external"]])
    # verificación de fuga: ningún grupo en dos splits
    leak = df.groupby("fruit_id")["split"].nunique()
    assert (leak == 1).all(), f"fuga: {leak[leak > 1].index.tolist()[:5]}"
    out = df[["image_id", "split", "fruit_id", "fruit_species", "quality_class", "source"]]
    path = root / sc["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def summarize(splits: pd.DataFrame) -> pd.DataFrame:
    return (splits.groupby(["split", "fruit_species", "quality_class"]).size().unstack(fill_value=0))
