"""Búsqueda en rejilla de la cabeza del clasificador de especie (MobileNetV2 congelada).

Rejilla: unidades {128, 256, 512} x dropout {0,3; 0,5} x tasa de aprendizaje {1e-3; 3e-4}. Cada combinación se
entrena con train y parada temprana en la pérdida de validación; se elige por exactitud en VALIDACIÓN (el test no
se usa). Resultado: experiments/results/tables/tuning_cabeza_especie.csv. Reanudable.
Uso: python -m src.training.tune_species_tl
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.transfer import build_head, embeddings
from src.training.data import load_table
from src.utils.config import load_config

OUT = Path("experiments/results/tables/tuning_cabeza_especie.csv")
GRID = {"units": [128, 256, 512], "dropout": [0.3, 0.5], "lr": [1e-3, 3e-4]}


def run(cfg: dict, deadline: float | None = None):
    import tensorflow as tf
    from tensorflow.keras import layers
    df = load_table(cfg)
    E = embeddings(df["image_id"].tolist(), cfg)
    classes = np.array(sorted(df["fruit_species"].unique()))
    y = np.searchsorted(classes, df["fruit_species"].to_numpy())
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["units", "dropout", "lr"])
    for u, d, lr in itertools.product(*GRID.values()):
        if ((done["units"] == u) & (done["dropout"] == d) & (done["lr"] == lr)).any():
            continue
        if deadline is not None and time.time() > deadline:
            return done
        norm = layers.Normalization(); norm.adapt(E[tr])
        head = build_head(E.shape[1], len(classes), cfg["seed"], units=u, dropout=d)
        z = layers.Input((E.shape[1],)); m = tf.keras.Model(z, head(norm(z)))
        m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = m.fit(E[tr], y[tr], validation_data=(E[va], y[va]), epochs=60, batch_size=128, verbose=0,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        acc = float((m.predict(E[va], verbose=0).argmax(1) == y[va]).mean())
        tr_acc = float((m.predict(E[tr], verbose=0).argmax(1) == y[tr]).mean())
        row = dict(units=u, dropout=d, lr=lr, val_accuracy=round(acc, 4), train_accuracy=round(tr_acc, 4),
                   val_loss=round(min(h.history["val_loss"]), 4), epochs=len(h.history["loss"]))
        done = pd.concat([done, pd.DataFrame([row])], ignore_index=True); done.to_csv(OUT, index=False)
        print(row, flush=True)
    return done.sort_values("val_accuracy", ascending=False)


if __name__ == "__main__":
    print(run(load_config(), deadline=time.time() + 130).to_string())
