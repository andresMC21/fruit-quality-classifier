"""CNN pequeña desde cero (sin transfer learning), con dos salidas: especie y calidad.

Entrada 64x64x3 (recorte con fondo gris). 3 bloques Conv-BN-ReLU-MaxPool (32/64/128 filtros), pooling global,
Dropout y una capa densa compartida. ~100 mil parámetros: cabe en CPU y es difícil que memorice 5 mil imágenes.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf
from tensorflow.keras import layers, models

IMG = 64


def build_cnn(n_species: int, n_quality: int, seed: int = 42) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    x_in = layers.Input((IMG, IMG, 3))
    x = x_in
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    species = layers.Dense(n_species, activation="softmax", name="species")(x)
    quality = layers.Dense(n_quality, activation="softmax", name="quality")(x)
    return models.Model(x_in, {"species": species, "quality": quality}, name="cnn_pequena")


def augmenter(seed: int = 42):
    """Aumento leve: la forma y el color son la señal, así que no se deforma ni se cambia el tono."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomRotation(0.05, fill_mode="constant", fill_value=128 / 255, seed=seed),
        layers.RandomZoom(0.05, fill_mode="constant", fill_value=128 / 255, seed=seed),
        layers.RandomBrightness(0.1, value_range=(0, 1), seed=seed),
    ])
