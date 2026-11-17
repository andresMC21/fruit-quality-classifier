"""Division train/val/test FIJA, AGRUPADA y estratificada.

- Grupo = fruit_id (fotos propias) o grupo de duplicados de dedup.py (Kaggle). Un grupo nunca aparece en dos splits.
- Estratificar por quality_class.
- El test incluye una fraccion de frutas PROPIAS (config: split.test_own), para medir la generalizacion a datos locales.
- Guardar el resultado en data/processed/splits.csv (se versiona) y no regenerarlo despues de mirar el test.

Implementacion: reparto por clase (no GroupShuffleSplit ni StratifiedGroupKFold: con pocos grupos por clase
no garantizan que cada clase tenga test). Cada grupo recibe la clase mas grave de sus fotos (mala > regular >
buena; dos vistas de la misma fruta pueden diferir). Dentro de cada clase se barajan los grupos con la
semilla del config y se asignan a test y val segun split_sizes; el resto va a train.
Las fotos propias y las publicas (kaggle, mixed_quality, other_group) se reparten por separado, asi que
val tendra pocas fotos propias (15 % de unas 30 frutas de train+val): sirve sobre todo para ajustar con Kaggle.

Uso:
    python -m src.data.split            (se niega a sobrescribir splits.csv)
    python -m src.data.split --force    (solo antes de haber mirado el test)
"""
import argparse
import random
from pathlib import Path

import pandas as pd

from src.utils.config import load_config


def group_labels(df, classes):
    """Clase de cada fruit_id: la mas grave (ultima en el orden de classes) entre sus fotos."""
    severity = {c: i for i, c in enumerate(classes)}
    worst = df.groupby("fruit_id")["quality_class"].agg(lambda s: max(s, key=severity.__getitem__))
    return worst.to_dict()


def split_sizes(n, test_frac, val_frac):
    """Cuantos grupos de una clase van a test y a val.

    Con pocos grupos, round() puede dejar test o val vacios, y entonces el recall de esa clase
    no se puede medir. Por eso: con n >= 3 hay al menos 1 en test, 1 en val y 1 en train;
    con n == 2, uno a test y otro a train; con n == 1, a train.
    """
    if n < 2:
        return 0, 0
    if n == 2:
        return 1, 0
    n_test = max(1, round(test_frac * n))
    n_val = max(1, round(val_frac * n))
    n_val = min(n_val, n - n_test - 1)   # siempre queda al menos 1 en train
    return n_test, max(n_val, 0)


def assign_groups(labels, test_frac, val_frac, rng):
    """{fruit_id: clase} -> {fruit_id: split}, estratificado por clase."""
    by_class = {}
    for gid in sorted(labels):
        by_class.setdefault(labels[gid], []).append(gid)
    out = {}
    for cls, gids in sorted(by_class.items()):
        rng.shuffle(gids)
        n_test, n_val = split_sizes(len(gids), test_frac, val_frac)
        if len(gids) < 3:
            print(f"Aviso: la clase '{cls}' tiene solo {len(gids)} grupo(s); val y/o test quedan sin ella.")
        for i, gid in enumerate(gids):
            out[gid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return out


def make_splits(annotations_csv: str, cfg: dict):
    df = pd.read_csv(annotations_csv)
    df = df[df["quality_class"].isin(cfg["classes"])].copy()   # sin etiqueta o excluidas: fuera
    rng = random.Random(cfg["seed"])
    s = cfg["split"]
    is_own = df["source"] == "own"
    shared = set(df.loc[is_own, "fruit_id"]) & set(df.loc[~is_own, "fruit_id"])
    if shared:
        # Se reparten por separado: un fruit_id repetido en ambas partes se pisaria sin aviso.
        raise ValueError(f"fruit_id presentes en fotos propias y publicas: {sorted(shared)[:5]}")
    split_of = {}
    for part, test_frac in ((df[is_own], s["test_own"]), (df[~is_own], s["test_public"])):
        if len(part):
            split_of.update(assign_groups(group_labels(part, cfg["classes"]), test_frac, s["val"], rng))
    df["split"] = df["fruit_id"].map(split_of)
    return df[["image_id", "filename", "source", "fruit_id", "quality_class", "split"]]


def write_splits(cfg, force=False):
    out = Path(cfg["split"]["output"])
    if out.exists() and not force:
        raise SystemExit(f"{out} ya existe. El split es fijo: no se regenera despues de mirar el test.")
    splits = make_splits(cfg["paths"]["annotations"], cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(out, index=False)
    print(pd.crosstab([splits["source"], splits["quality_class"]], splits["split"], margins=True))
    return splits


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Genera data/processed/splits.csv")
    ap.add_argument("--force", action="store_true")
    write_splits(cfg, ap.parse_args().force)


if __name__ == "__main__":
    main()
