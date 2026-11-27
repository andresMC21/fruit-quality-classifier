"""Entrena la cabeza propia del clasificador de especie sobre MobileNetV2 congelada (src/models/transfer.py).

Train: split train (incluye Fruits-360 y su clase `other`); parada temprana con la pérdida de VALIDACIÓN.
Guarda models/species_tl.keras = [entrada 128x128 RGB en [-1,1]] -> MobileNetV2 congelada -> normalización ->
cabeza (Dropout, Dense 256, Dropout, softmax), y registra la fila en experiments_log.csv.
Ejecutar: python -m src.training.train_species_tl
"""
from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import numpy as np

from src.models.transfer import SIZE, backbone, build_head, embeddings
from src.training.data import load_table
from src.training.train import log_row
from src.utils.config import load_config, set_seed


def train_species_tl(cfg: dict, out: str = "models/species_tl.keras", class_weight: bool = True, aug_mango: bool = False):
    """v2.2: pesos por clase inversos a la frecuencia (la manzana, la clase más numerosa, absorbía las dudas)."""
    import tensorflow as tf
    from tensorflow.keras import layers
    df = load_table(cfg)
    E = embeddings(df["image_id"].tolist(), cfg)
    classes = np.array(sorted(df["fruit_species"].unique()))
    y = np.searchsorted(classes, df["fruit_species"].to_numpy())
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    Xtr, ytr = E[tr], y[tr]
    if aug_mango:                     # mangos rojo-amarillos sintéticos (src/data/augment_mango.py), solo en train
        import glob, os
        aug_ids = sorted(os.path.basename(f)[:-4] for f in glob.glob("data/interim/aug_mango/*.png"))
        Ea = embeddings(aug_ids, cfg, seg_dir="data/interim/aug_mango", cache_name="emb_mnv2_aug.npz")
        Xtr = np.vstack([Xtr, Ea]); ytr = np.concatenate([ytr, np.full(len(Ea), int(np.searchsorted(classes, "mango")))])
        print("aumento de mango:", len(Ea))
    norm = layers.Normalization(name="norm"); norm.adapt(Xtr)
    head = build_head(E.shape[1], len(classes), cfg["seed"])
    z_in = layers.Input((E.shape[1],)); clf = tf.keras.Model(z_in, head(norm(z_in)))
    clf.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t = time.time()
    cnt = np.bincount(ytr, minlength=len(classes)).astype(float)
    cw = {i: cnt.sum() / (len(classes) * max(c, 1)) for i, c in enumerate(cnt)} if class_weight else None
    h = clf.fit(Xtr, ytr, validation_data=(E[va], y[va]), epochs=60, batch_size=128, verbose=0, class_weight=cw,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    fit_s = time.time() - t
    pv = classes[clf.predict(E[va], verbose=0).argmax(1)]
    acc = float((pv == df.loc[va, "fruit_species"].to_numpy()).mean())
    by_src = {s: round(float((pv[g] == df.loc[va, "fruit_species"].to_numpy()[g]).mean()), 4)
              for s in df.loc[va, "source"].unique() for g in [(df.loc[va, "source"] == s).to_numpy()]}
    print("exactitud especie (val):", round(acc, 4), by_src, "épocas:", len(h.history["loss"]))
    # modelo completo para la app: imagen -> especie
    x_in = layers.Input((SIZE, SIZE, 3))
    full = tf.keras.Model(x_in, clf(backbone()(x_in, training=False)), name="especie_mobilenetv2")
    full.save(out)
    Path(out).with_suffix(".classes.json").write_text(json.dumps(classes.tolist()))
    hist = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    Path("experiments/checkpoints/species_tl_history.json").write_text(json.dumps(hist))
    log_row(dict(exp_id=f"species_tl_{date.today():%m%d}_v22", date=f"{date.today()}", model="species:mobilenetv2_congelada+cabeza",
                 hyperparams=json.dumps({"backbone": "MobileNetV2 ImageNet 128, congelada", "head": "Dropout .3, Dense 256, Dropout .3", "class_weight": class_weight, "aug_mango": aug_mango,
                                         "epochs_run": len(h.history["loss"]), "optimizer": "adam_1e-3"}),
                 val_f1_macro=round(acc, 4), fit_seconds=round(fit_s, 1),
                 notes="v2.2: + WGISD (racimos) y más mango de Fruits-360; pesos por clase; validación entre fuentes 0,681 (aumento de mango rojo probado y descartado: val 0,922)"))
    return acc


if __name__ == "__main__":
    cfg = load_config(); set_seed(cfg["seed"]); train_species_tl(cfg)
