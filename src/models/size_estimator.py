"""Estimacion de tamano a partir de la mascara de segmentacion.

1. d_px = 2 * sqrt(A / pi)          (diametro equivalente; A = area de la mascara en pixeles)
2. d_mm = d_px * D_ref / d_ref_px   (si hay objeto de referencia de diametro conocido D_ref en la foto)
3. Clase por especie con umbrales en los terciles de train.
Sin referencia (Kaggle), el resultado es solo relativo: area/imagen depende de la distancia a la camara.
Validar: MAE en mm frente a diameter_mm medido con regla o calibrador.
"""


def equivalent_diameter_px(mask) -> float:
    raise NotImplementedError


def to_mm(d_px: float, ref_diameter_px: float | None, ref_diameter_mm: float | None) -> float | None:
    raise NotImplementedError


def size_class(d: float, thresholds: tuple[float, float]) -> str:
    raise NotImplementedError
