"""Entrenamiento de la CNN (especie + calidad). Test NO se toca aquí.

Solo usa imágenes con etiqueta de calidad (Fruits-360, que solo trae especie, queda fuera de la CNN)."""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluation.metrics import breakdown, summarize
from src.training.data import has_quality, load_table
from src.training.train import log_row


def load_images(df: pd.DataFrame, cfg: dict, size: int = 64) -> np.ndarray:
    cache = Path(cfg["paths"]["processed"]) / f"images{size}.npz"
    known = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["ids"].tolist(), z["X"]))
    seg = Path(cfg["paths"]["interim"]) / "segmented"
    miss = [i for i in df["image_id"] if i not in known]
    for i in miss:
        im = cv2.imread(str(seg / f"{i}.png"), cv2.IMREAD_COLOR)           # descarta alfa: fondo ya es gris
        known[i] = cv2.cvtColor(cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    if miss:
        np.savez(cache, ids=np.array(list(known)), X=np.stack(list(known.values())))
    return np.stack([known[i] for i in df["image_id"]])


def train_cnn(cfg: dict, epochs: int = 40, batch: int = 64, budget_s: int = 150, resume: bool = True,
              lr: float = 2e-3, dropout: float = 0.3, tag: str = "cnn_v4"):
    """tag nombra los checkpoints; lr y dropout son los hiperparámetros de la búsqueda (ver run_cnn)."""
    import tensorflow as tf
    from src.models.cnn import augmenter, build_cnn
    df = load_table(cfg)
    df = df[has_quality(df)].reset_index(drop=True)
    species, classes = cfg["fruit_species"], cfg["classes"]
    X = load_images(df, cfg).astype(np.float32) / 255.0
    ys = df["fruit_species"].map({s: i for i, s in enumerate(species)}).to_numpy()
    yq = df["quality_class"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    # pesos por clase de calidad (balanceo) en train
    cnt = np.bincount(yq[tr], minlength=len(classes)).astype(float)
    cw = cnt.sum() / (len(classes) * np.maximum(cnt, 1))
    sw_q = cw[yq[tr]]
    ck = Path("experiments/checkpoints"); ck.mkdir(parents=True, exist_ok=True)
    wpath, hpath = ck / f"{tag}.weights.h5", ck / f"{tag}_history.json"
    model = build_cnn(len(species), len(classes), cfg["seed"], dropout=dropout)
    hist = json.loads(hpath.read_text()) if (resume and hpath.exists()) else {"loss": [], "val_loss": [], "val_quality_acc": [], "train_quality_acc": [], "best": 1e9, "wait": 0, "seconds": 0.0}
    if resume and wpath.exists():
        model.load_weights(wpath)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={"species": "sparse_categorical_crossentropy", "quality": "sparse_categorical_crossentropy"},
                  loss_weights={"species": 0.5, "quality": 1.0},
                  metrics={"species": "accuracy", "quality": "accuracy"})
    aug = augmenter(cfg["seed"])
    ds = (tf.data.Dataset.from_tensor_slices((X[tr], {"species": ys[tr], "quality": yq[tr]}, {"species": np.ones(tr.sum(), np.float32), "quality": sw_q.astype(np.float32)}))
          .shuffle(4096, seed=cfg["seed"]).batch(batch).map(lambda x, y, w: (aug(x, training=True), y, w)).prefetch(2))
    dv = tf.data.Dataset.from_tensor_slices((X[va], {"species": ys[va], "quality": yq[va]})).batch(256)
    t_call, t0 = time.time(), time.time()
    best_w = ck / f"{tag}.best.weights.h5"
    for ep in range(len(hist["loss"]), epochs):
        if time.time() - t_call > budget_s:
            print("PENDIENTE: presupuesto de tiempo; repetir para continuar"); break
        # coseno por época (coherente entre reanudaciones): 2e-3 -> 4e-5
        model.optimizer.learning_rate.assign(lr / 50 + 0.5 * (lr - lr / 50) * (1 + np.cos(np.pi * ep / epochs)))
        h = model.fit(ds, validation_data=dv, epochs=1, verbose=0)
        vl = h.history["val_loss"][0]
        hist["loss"].append(h.history["loss"][0]); hist["val_loss"].append(vl)
        hist["train_quality_acc"].append(h.history["quality_accuracy"][0]); hist["val_quality_acc"].append(h.history["val_quality_accuracy"][0])
        print(f"ep {ep+1} loss {hist['loss'][-1]:.3f} val_loss {vl:.3f} tr_acc_q {hist['train_quality_acc'][-1]:.3f} val_acc_q {hist['val_quality_acc'][-1]:.3f}", flush=True)
        if vl < hist["best"] - 1e-4:
            hist["best"], hist["wait"] = vl, 0; model.save_weights(best_w)
        else:
            hist["wait"] += 1
        model.save_weights(wpath); hist["seconds"] += time.time() - t0; t0 = time.time()
        hpath.write_text(json.dumps(hist))
        if hist["wait"] >= 12:
            print("parada temprana"); hist["done"] = True; hpath.write_text(json.dumps(hist)); break
    else:
        hist["done"] = True; hpath.write_text(json.dumps(hist))
    return model, hist


def finalize_cnn(cfg: dict, tag: str = "cnn_v4", dropout: float = 0.3, lr: float = 2e-3, log: bool = True):
    """Carga los mejores pesos, mide en VALIDACIÓN y registra la fila en experiments_log.csv."""
    from src.models.cnn import build_cnn
    df = load_table(cfg)
    df = df[has_quality(df)].reset_index(drop=True)
    species, classes = cfg["fruit_species"], cfg["classes"]
    X = load_images(df, cfg).astype(np.float32) / 255.0
    va = (df["split"] == "val").to_numpy()
    hist = json.loads(Path(f"experiments/checkpoints/{tag}_history.json").read_text())
    model = build_cnn(len(species), len(classes), cfg["seed"], dropout=dropout)
    model.load_weights(f"experiments/checkpoints/{tag}.best.weights.h5")
    out = model.predict(X[va], verbose=0)
    pq = np.array(classes)[out["quality"].argmax(1)]
    ps = np.array(species)[out["species"].argmax(1)]
    s = summarize(df.loc[va, "quality_class"].to_numpy(), pq, cfg)
    print({k: round(v, 3) for k, v in s.items()}, "| acc especie:", round(float((ps == df.loc[va, "fruit_species"].to_numpy()).mean()), 4))
    print(breakdown(df[va], pq, cfg, "fruit_species").to_string()); print(breakdown(df[va], pq, cfg, "source").to_string())
    log_row(dict(exp_id=f"cnn_{date.today():%m%d}_v2", date=f"{date.today()}", model="quality+species:cnn_3conv",
                 hyperparams=json.dumps({"input": 64, "filters": [32, 64, 128], "epochs_run": len(hist["loss"]), "optimizer": "adam_cosine_2e-3"}),
                 val_f1_macro=round(s["f1_macro"], 4), val_recall_mala=round(s.get("recall_mala", float("nan")), 4),
                 val_cost=round(s["cost"], 4), fit_seconds=round(hist["seconds"], 1), notes="aug leve; parada temprana en val_loss"))
    return s
