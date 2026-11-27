"""Clasificador de ESPECIE con transfer learning (permitido por el enunciado si se congelan capas y se añaden
capas propias): MobileNetV2 preentrenada en ImageNet, CONGELADA, como extractor (vector de 1280) + cabeza densa
propia entrenada con nuestros datos.

Por qué: con características manuales la especie no generaliza entre cámaras (validación dejando una fuente
fuera: ~44 % de exactitud media con el SVM). Los pesos se descargan del repositorio público de MobileNetV2 para
Keras y se guardan en models/pretrained/.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
WEIGHTS = Path("models/pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5")
SIZE = 128


def backbone():
    import tensorflow as tf
    m = tf.keras.applications.MobileNetV2(input_shape=(SIZE, SIZE, 3), include_top=False, weights=str(WEIGHTS),
                                          pooling="avg")
    m.trainable = False
    return m


def to_input(bgr: np.ndarray) -> np.ndarray:
    """Recorte BGR (fondo gris) -> RGB en [-1, 1], como espera MobileNetV2."""
    rgb = cv2.cvtColor(cv2.resize(bgr[:, :, :3], (SIZE, SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32) / 127.5 - 1.0


def embeddings(ids, cfg: dict, deadline: float | None = None, batch: int = 64, seg_dir: str | None = None,
               cache_name: str = "emb_mnv2.npz") -> np.ndarray:
    """Vectores de 1280 por image_id; caché reanudable en data/processed/<cache_name>."""
    cache = Path(cfg["paths"]["processed"]) / cache_name
    known = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        known = dict(zip(z["ids"].tolist(), z["E"]))
    miss = [i for i in ids if i not in known]
    if miss:
        bb = backbone()
        seg = Path(seg_dir) if seg_dir else Path(cfg["paths"]["interim"]) / "segmented"
        for k in range(0, len(miss), 512):
            part = miss[k:k + 512]
            x = np.stack([to_input(cv2.imread(str(seg / f"{i}.png"), cv2.IMREAD_COLOR)) for i in part])
            for i, e in zip(part, bb.predict(x, batch_size=batch, verbose=0)):
                known[i] = e.astype(np.float32)
            np.savez(cache, ids=np.array(list(known)), E=np.stack(list(known.values())))
            if deadline is not None and time.time() > deadline and k + 512 < len(miss):
                raise TimeoutError(f"embeddings {k + 512}/{len(miss)}: repetir para continuar")
    return np.stack([known[i] for i in ids])


def build_head(n_in: int, n_out: int, seed: int = 42, units: int = 256, dropout: float = 0.3):
    """Capas propias sobre el extractor congelado."""
    import tensorflow as tf
    from tensorflow.keras import layers
    tf.keras.utils.set_random_seed(seed)
    x_in = layers.Input((n_in,))
    x = layers.Dropout(dropout)(x_in)
    x = layers.Dense(units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_out, activation="softmax", name="species")(x)
    return tf.keras.Model(x_in, out, name="cabeza_especie")
