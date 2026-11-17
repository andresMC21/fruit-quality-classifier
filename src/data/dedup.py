"""Deteccion de casi-duplicados para evitar fuga de datos entre train y test.

FruitNet/Kaggle contiene varias fotos muy parecidas de la misma fruta. Si caen en splits distintos,
el test mide memoria y no generalizacion.
Idea: hash perceptual (imagehash.phash) por imagen; dos imagenes con distancia de Hamming <= umbral
(configs/config.yaml: dedup.hamming_threshold) pertenecen al mismo grupo (union-find).
Comparar dentro de cada carpeta fruta/clase para que el costo O(n^2) sea manejable.
Salida: columna group_id usada por split.py. Reportar en el EDA cuantos grupos y duplicados hay.
"""
import imagehash
from PIL import Image


def compute_groups(image_paths, hamming_threshold: int = 6) -> dict:
    """Devuelve {ruta_imagen: id_grupo}.

    El id de grupo es la posicion (en image_paths) de la primera imagen del grupo, asi que
    el resultado es el mismo siempre que image_paths llegue en el mismo orden.
    La union es transitiva: si A~B y B~C, A, B y C quedan juntas aunque A y C esten lejos.
    """
    hashes = []
    for path in image_paths:
        with Image.open(path) as im:
            hashes.append(imagehash.phash(im))

    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hashes[i] - hashes[j] <= hamming_threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    return {path: find(i) for i, path in enumerate(image_paths)}
