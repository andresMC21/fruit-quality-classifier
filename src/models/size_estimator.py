"""Estimacion de tamano a partir de la mascara de segmentacion.

1. d_px = 2 * sqrt(A / pi)          (diametro equivalente; A = area de la mascara en pixeles)
2. d_mm = d_px * D_ref / d_ref_px   (si hay objeto de referencia de diametro conocido D_ref en la foto)
3. Clase por especie con umbrales en los terciles de train.
Sin referencia (Kaggle), el resultado es solo relativo: area/imagen depende de la distancia a la camara.
Validar: MAE en mm frente a diameter_mm medido con regla o calibrador.

Ademas (docs/formulacion_matematica.md 3.4):
- feret_diameter_px: maxima distancia entre dos puntos del contorno. El calibrador mide el diametro
  maximo, y el equivalente lo subestima si la fruta no es redonda; se comparan ambos con el MAE.
- Correccion de perspectiva con la altura H de la camara: d = a / (1 + a / (2H)), con a = d_px * s0.
"""
import cv2
import numpy as np


def equivalent_diameter_px(mask) -> float:
    return float(2.0 * np.sqrt(np.count_nonzero(mask) / np.pi))


def feret_diameter_px(mask) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    hull = cv2.convexHull(max(contours, key=cv2.contourArea)).reshape(-1, 2).astype(np.float64)
    d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(axis=2))
    # +1: los centros de los pixeles extremos estan a (ancho - 1) de distancia
    return float(d.max() + 1.0)


def to_mm(d_px: float, ref_diameter_px: float | None, ref_diameter_mm: float | None,
          camera_height_mm: float | None = None) -> float | None:
    """Convierte a mm con la referencia; si se da la altura de la camara, corrige la perspectiva.

    ref_diameter_* es la medida lineal de la referencia: con el cuadrado impreso, su lado
    (find_reference_square devuelve el lado en pixeles; el lado en mm se mide con regla).
    """
    if not ref_diameter_px or not ref_diameter_mm:
        return None
    a = d_px * ref_diameter_mm / ref_diameter_px
    if camera_height_mm:
        # La referencia esta sobre la mesa y el contorno de la fruta a ~d/2 por encima: se ve mas grande.
        return a / (1.0 + a / (2.0 * camera_height_mm))
    return a


def fit_thresholds(diameters) -> tuple[float, float]:
    """Terciles de los diametros de TRAIN (no usar val ni test)."""
    q1, q2 = np.quantile(np.asarray(diameters, dtype=float), [1 / 3, 2 / 3])
    return float(q1), float(q2)


def size_class(d: float, thresholds: tuple[float, float]) -> str:
    if d < thresholds[0]:
        return "pequeno"
    if d < thresholds[1]:
        return "mediano"
    return "grande"
