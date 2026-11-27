"""Resume la búsqueda de hiperparámetros de la CNN (solo VALIDACIÓN): una fila por configuración entrenada con
src.training.run_cnn. Salida: experiments/results/tables/tuning_cnn.csv. Uso: python -m src.training.tune_cnn"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import summarize
from src.models.cnn import build_cnn
from src.training.data import has_quality, load_table
from src.training.train_cnn import load_images
from src.utils.config import load_config

CONFIGS = [("cnn_v4", 2e-3, 0.3), ("cnn_lr1e3", 1e-3, 0.3)]
OUT = Path("experiments/results/tables/tuning_cnn.csv")


def run(cfg: dict):
    df = load_table(cfg); df = df[has_quality(df)].reset_index(drop=True)
    va = (df["split"] == "val").to_numpy()
    X = load_images(df[va], cfg).astype(np.float32) / 255.0
    rows = []
    for tag, lr, do in CONFIGS:
        hp, wp = Path(f"experiments/checkpoints/{tag}_history.json"), Path(f"experiments/checkpoints/{tag}.best.weights.h5")
        if not (hp.exists() and wp.exists()):
            continue
        h = json.loads(hp.read_text())
        m = build_cnn(len(cfg["fruit_species"]), len(cfg["classes"]), cfg["seed"], dropout=do); m.load_weights(wp)
        pq = np.array(cfg["classes"])[m.predict(X, verbose=0)["quality"].argmax(1)]
        s = summarize(df.loc[va, "quality_class"].to_numpy(), pq, cfg)
        rows.append(dict(config=tag, lr=lr, dropout=do, epochs=len(h["loss"]), best_val_loss=round(h["best"], 4),
                         val_f1_macro=round(s["f1_macro"], 4), val_recall_mala=round(s["recall_mala"], 4),
                         val_cost=round(s["cost"], 4), train_acc_q=round(h["train_quality_acc"][-1], 4),
                         val_acc_q=round(h["val_quality_acc"][-1], 4), seconds=round(h["seconds"], 1)))
    t = pd.DataFrame(rows); t.to_csv(OUT, index=False); return t


if __name__ == "__main__":
    print(run(load_config()).to_string())
