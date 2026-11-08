"""Pipeline UNICO de inferencia, usado por la app y por evaluate.py.

Si la app y la evaluacion preprocesan distinto (otro tamano, otra normalizacion, otra segmentacion),
el modelo se comporta peor en la camara que en el test. Por eso todo pasa por aqui:
    imagen BGR -> segment_image -> preprocess/features -> modelo -> estimate_size -> resultado
"""
from dataclasses import dataclass


@dataclass
class Prediction:
    quality: str                 # clase predicha
    probabilities: dict          # {clase: probabilidad}
    size_class: str              # pequeno | mediano | grande
    diameter_mm: float | None    # None si no hay objeto de referencia
    segmentation_ok: bool        # False -> la app pide repetir la foto


def load_model(model_dir: str = "models"):
    """TODO: cargar el modelo final (joblib o keras) y sus metadatos (clases, umbrales de tamano)."""
    raise NotImplementedError


def predict(img_bgr, model) -> Prediction:
    """TODO: aplicar exactamente el mismo preprocesamiento que en entrenamiento."""
    raise NotImplementedError
