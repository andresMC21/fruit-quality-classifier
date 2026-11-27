"""Entrenamiento de los modelos clásicos y registro en experiments/results/tables/experiments_log.csv.

Flujo por modelo: tuning con CV agrupada sobre TRAIN -> reajuste con los mejores parámetros -> métricas en VAL.
El conjunto de TEST no se toca aquí (solo src/evaluation/evaluate.py, una vez, con el modelo congelado).
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.evaluation.metrics import breakdown, summarize
from src.models.classical import build_model
from src.training.data import has_quality, load_xy
from src.training.tune import tune

LOG = Path("experiments/results/tables/experiments_log.csv")
LOG_COLS = ["exp_id", "date", "model", "hyperparams", "cv_f1_macro", "val_f1_macro", "val_recall_mala",
            "val_cost", "fit_seconds", "test_f1_macro", "notes"]


def log_row(row: dict):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    if LOG.exists():
        old = pd.read_csv(LOG)
        old = old.reindex(columns=LOG_COLS)
    else:
        old = pd.DataFrame(columns=LOG_COLS)
    row = {**{c: "" for c in LOG_COLS}, **row}
    pd.concat([old, pd.DataFrame([row])], ignore_index=True)[LOG_COLS].to_csv(LOG, index=False)


def train_classical(model_name: str, cfg: dict, task: str = "quality", deadline: float | None = None):
    """task: 'quality' (X + especie; solo filas con calidad) | 'species' (solo características visuales; incluye
    la clase `other` de Fruits-360)."""
    df, X, Xs = load_xy(cfg, deadline=deadline)
    lab = has_quality(df) if task == "quality" else np.ones(len(df), bool)
    tr, va = (df["split"] == "train").to_numpy() & lab, (df["split"] == "val").to_numpy() & lab
    feats = Xs if task == "quality" else X
    y = df["quality_class"] if task == "quality" else df["fruit_species"]
    strat = df["fruit_species"] + "|" + df["quality_class"].fillna("-").astype(str)
    t0 = time.time()
    if model_name == "majority":
        best, cv_f1, res = {}, float("nan"), None
    else:
        best, cv_f1, res, _ = tune(model_name, feats[tr], y[tr].to_numpy(), df.loc[tr, "fruit_id"].to_numpy(),
                                   strat[tr].to_numpy(), cfg, task=task, deadline=deadline)
        print(res.head(5).to_string())
    model = build_model(model_name, cfg["seed"], probability=True, **best)
    model.fit(feats[tr], y[tr])
    fit_s = time.time() - t0
    pred = model.predict(feats[va])
    if task == "quality":
        s = summarize(y[va].to_numpy(), pred, cfg)
        print({k: round(v, 3) for k, v in s.items()})
        print(breakdown(df[va], pred, cfg, "fruit_species").to_string())
        print(breakdown(df[va], pred, cfg, "source").to_string())
    else:
        s = dict(f1_macro=float((pred == y[va].to_numpy()).mean()), accuracy=float((pred == y[va].to_numpy()).mean()))
        print("accuracy especie (val):", round(s["accuracy"], 4))
    ck = Path("experiments/checkpoints"); ck.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ck / f"{task}_{model_name}.joblib")
    log_row(dict(exp_id=f"{task}_{model_name}_{date.today():%m%d}_v2", date=f"{date.today()}", model=f"{task}:{model_name}",
                 hyperparams=json.dumps({k.replace("clf__", ""): v for k, v in best.items()}),
                 cv_f1_macro=round(cv_f1, 4) if cv_f1 == cv_f1 else "", val_f1_macro=round(s["f1_macro"], 4),
                 val_recall_mala=round(s.get("recall_mala", float("nan")), 4), val_cost=round(s.get("cost", float("nan")), 4),
                 fit_seconds=round(fit_s, 1), notes="v2: + FruitVision (+ Fruits-360 en especie); val agrupada"))
    return model, s
