"""Pipeline ÚNICO de inferencia, usado por la app y por evaluate.py.

Si la app y la evaluación preprocesan distinto, el modelo se comporta peor en la cámara que en el test.
Por eso todo pasa por aquí:
    imagen BGR -> segment_image -> recorte enmascarado -> características -> especie -> calidad -> tamaño
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import joblib
import numpy as np

from src.data.features import extract_features
from src.data.preprocess import masked_square_crop
from src.data.segment import segment_image
from src.models.size_estimator import equivalent_diameter_px, find_reference, size_class, to_mm

MIN_SPECIES_CONF = 0.60       # por debajo, la fruta se declara "no reconocida"
MIN_QUALITY_CONF = 0.50       # por debajo, se pide revisión humana


@dataclass
class Prediction:
    species: str | None          # None -> fruta no reconocida
    species_confidence: float
    quality: str | None          # clase predicha (None si no hay predicción fiable)
    probabilities: dict          # {clase: probabilidad}
    size_class: str | None       # pequeno | mediano | grande
    diameter_mm: float | None    # None si no hay objeto de referencia
    relative_diameter: float | None
    segmentation_ok: bool        # False -> la app pide repetir la foto
    needs_review: bool = False
    message: str = ""
    mask: np.ndarray | None = field(default=None, repr=False)
    crop: np.ndarray | None = field(default=None, repr=False)


def load_model(model_dir: str = "models"):
    d = Path(model_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    qs = {k: joblib.load(d / f) for k, f in meta.get("quality_models", {"default": meta["quality_model"]}).items()}
    return dict(meta=meta, species=joblib.load(d / meta["species_model"]), qualities=qs,
                quality=qs[meta.get("default_quality", next(iter(qs)))])


def _thresholds(meta: dict, species: str, has_ref: bool):
    t = meta["size_thresholds"].get(species)
    return None if t is None else tuple(t["mm" if has_ref and "mm" in t else "relative"])


def predict(img_bgr, model, ref_diameter_mm: float | None = None, quality_model: str | None = None) -> Prediction:
    meta = model["meta"]
    qm = model["qualities"][quality_model] if quality_model else model["quality"]
    seg = segment_image(img_bgr)
    if not seg.ok:                       # la especie aún no se conoce: reintentar con el control de forma del banano/lima
        seg = segment_image(img_bgr, species="banana")
    if not seg.ok:
        return Prediction(None, 0.0, None, {}, None, None, None, False, True,
                          "No se pudo aislar la fruta: use un fondo liso y una sola fruta en la foto.", seg.mask)
    crop, cmask = masked_square_crop(img_bgr, seg.mask, seg.bbox, tuple(meta["image_size"]))
    bgra = np.dstack([crop, cmask])
    x = extract_features(bgra)[None]
    sp_proba = model["species"].predict_proba(x)[0]
    sp_classes = list(model["species"].classes_)
    k = int(sp_proba.argmax())
    species, sp_conf = sp_classes[k], float(sp_proba[k])
    if sp_conf < MIN_SPECIES_CONF:
        return Prediction(None, sp_conf, None, {}, None, None, None, True, True,
                          f"Fruta no reconocida (confianza {sp_conf:.0%}). Especies soportadas: {', '.join(meta['species'])}.", seg.mask, crop)
    onehot = np.array([[1.0 if s == species else 0.0 for s in meta["species"]]], np.float32)
    q_proba = qm.predict_proba(np.hstack([x, onehot]))[0]
    q_classes = list(qm.classes_)
    probs = {c: float(p) for c, p in zip(q_classes, q_proba)}
    quality = max(probs, key=probs.get)
    # tamaño
    H, W = img_bgr.shape[:2]
    d_px = equivalent_diameter_px(seg.mask)
    rel = d_px / min(H, W)
    d_mm = None
    if ref_diameter_mm:
        ref_px = find_reference(img_bgr, seg.mask)
        d_mm = to_mm(d_px, ref_px, ref_diameter_mm)
    th = _thresholds(meta, species, d_mm is not None)
    sc = size_class(d_mm if d_mm is not None else rel, th) if th else None
    review = probs[quality] < MIN_QUALITY_CONF
    msg = "Confianza baja: se recomienda revisión humana." if review else ""
    return Prediction(species, sp_conf, quality, probs, sc, d_mm, rel, True, review, msg, seg.mask, crop)
