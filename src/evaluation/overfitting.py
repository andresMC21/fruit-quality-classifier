"""Brecha de generalización de los modelos de calidad: F1 macro en train (re-sustitución), en CV agrupada (de la
búsqueda de hiperparámetros) y en validación. Salida: experiments/results/tables/sobreajuste.csv
Uso: python -m src.evaluation.overfitting"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score

from src.training.data import has_quality, load_xy
from src.utils.config import load_config

CK = Path("experiments/checkpoints")


def run(cfg: dict):
    df, X, Xs = load_xy(cfg); lab = has_quality(df)
    rows = []
    for m in ("logreg", "svm", "rf"):
        mod = joblib.load(CK / f"quality_{m}.joblib")
        f = {}
        for sp in ("train", "val"):
            k = (df["split"] == sp).to_numpy() & lab
            f[sp] = f1_score(df.loc[k, "quality_class"], mod.predict(Xs[k]), average="macro")
        t = json.loads((CK / f"tune_quality_{m}.json").read_text())
        cv = max(v[0] for v in t.values())
        rows.append(dict(modelo=m, f1_train=round(f["train"], 4), f1_cv=round(cv, 4), f1_val=round(f["val"], 4),
                         brecha_train_val=round(f["train"] - f["val"], 4)))
    t = pd.DataFrame(rows); t.to_csv("experiments/results/tables/sobreajuste.csv", index=False); return t


if __name__ == "__main__":
    print(run(load_config()).to_string())
