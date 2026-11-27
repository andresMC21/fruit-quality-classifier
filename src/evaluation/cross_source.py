"""Validación DEJANDO UNA FUENTE FUERA (solo train + val; el test no se usa).

Para cada fuente S en {kaggle, afruitdb, fruitvision}: se entrena con las demás fuentes y se mide en S. Mide la
generalización a una cámara nueva, que es lo que pide la app (las fotos del usuario son otra fuente). Es el
criterio de selección de la versión 2.1:
  - especie: exactitud; candidatos = SVM, RF, regresión logística sobre las 59 características y
    MobileNetV2 congelada + regresión logística / cabeza densa propia sobre su vector de 1280;
  - calidad: F1 macro (con la especie verdadera en one-hot); candidatos = SVM, RF, cabeza sobre MobileNetV2.
Uso: python -m src.evaluation.cross_source {especie|calidad} CANDIDATO  (resultados en
experiments/results/tables/validacion_entre_fuentes.csv)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models.classical import build_model
from src.training.data import has_quality, load_xy
from src.utils.config import load_config

FUENTES = ["kaggle", "afruitdb", "fruitvision"]
OUT = Path("experiments/results/tables/validacion_entre_fuentes.csv")


def _head(F, y, classes, cw=None, epochs=15, seed=42):
    import tensorflow as tf
    from src.models.transfer import build_head
    mu, sd = F.mean(0), F.std(0) + 1e-6
    h = build_head(F.shape[1], len(classes), seed)
    h.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy")
    h.fit((F - mu) / sd, np.searchsorted(classes, y), epochs=epochs, batch_size=128, verbose=0, class_weight=cw)
    return lambda Z: classes[h.predict((Z - mu) / sd, verbose=0).argmax(1)]


def run(tarea: str, cand: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    df, X, Xs = load_xy(cfg)
    emb = cand.startswith("emb")
    if emb:
        from src.models.transfer import embeddings
        E = embeddings(df["image_id"].tolist(), cfg)
    src = df["source"].to_numpy()
    dev = df["split"].isin(["train", "val"]).to_numpy()
    if tarea == "calidad":
        dev &= has_quality(df)
        y = df["quality_class"].to_numpy()
        F = np.hstack([E, Xs[:, X.shape[1]:]]) if emb else Xs
    else:
        y = df["fruit_species"].to_numpy()
        F = E if emb else X
    res = {}
    for S in FUENTES:
        tr = dev & (src != S)
        te = dev & (src == S) & np.isin(y, np.unique(y[tr]))
        if cand in ("svm", "emb_svm"):
            C = 100 if tarea == "calidad" else 10
            p = build_model("svm", 42, probability=False, clf__C=C, clf__gamma="scale").fit(F[tr], y[tr]).predict(F[te])
        elif cand == "rf":
            p = build_model("rf", 42, probability=False, clf__n_estimators=200, clf__min_samples_leaf=3,
                            **({} if tarea == "calidad" else {"clf__max_depth": 20})).fit(F[tr], y[tr]).predict(F[te])
        elif cand in ("logreg", "emb_logreg"):
            p = build_model("logreg", 42, clf__C=1 if emb else 10).fit(F[tr], y[tr]).predict(F[te])
        elif cand in ("emb_head", "emb_head_cw"):
            classes = np.array(sorted(set(y[tr])))
            cw = None
            if tarea == "calidad" or cand == "emb_head_cw":     # pesos inversos a la frecuencia de cada clase
                cnt = np.bincount(np.searchsorted(classes, y[tr]), minlength=len(classes)).astype(float)
                cw = {i: cnt.sum() / (len(classes) * max(c, 1)) for i, c in enumerate(cnt)}
            p = _head(F[tr], y[tr], classes, cw)(F[te])
        else:
            raise ValueError(cand)
        if tarea == "calidad":
            res[S] = float(f1_score(y[te], p, labels=sorted(set(y[te])), average="macro"))
        else:
            res[S] = float((p == y[te]).mean())
    row = dict(tarea=tarea, candidato=cand, **{k: round(v, 4) for k, v in res.items()}, media=round(float(np.mean(list(res.values()))), 4))
    tab = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    if len(tab):
        tab = tab[~((tab["tarea"] == tarea) & (tab["candidato"] == cand))]
    pd.concat([tab, pd.DataFrame([row])], ignore_index=True).to_csv(OUT, index=False)
    return row


if __name__ == "__main__":
    print(run(sys.argv[1], sys.argv[2]))
