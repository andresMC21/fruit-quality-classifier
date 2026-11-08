"""Deteccion de casi-duplicados para evitar fuga de datos entre train y test.

FruitNet/Kaggle contiene varias fotos muy parecidas de la misma fruta. Si caen en splits distintos,
el test mide memoria y no generalizacion.
Idea: hash perceptual (imagehash.phash) por imagen; dos imagenes con distancia de Hamming <= umbral
(configs/config.yaml: dedup.hamming_threshold) pertenecen al mismo grupo (union-find).
Comparar dentro de cada carpeta fruta/clase para que el costo O(n^2) sea manejable.
Salida: columna group_id usada por split.py. Reportar en el EDA cuantos grupos y duplicados hay.
"""


def compute_groups(image_paths, hamming_threshold: int = 6) -> dict:
    """TODO: devolver {ruta_imagen: id_grupo}."""
    raise NotImplementedError
