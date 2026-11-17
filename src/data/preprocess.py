"""Fase 3: redimensionado, normalización y aumento de datos.

preprocess es el paso comun para la CNN y para la app (ver src/inference/predict.py): misma entrada,
misma salida, en entrenamiento y en inferencia.
"""
import cv2
import numpy as np


def preprocess(img, size=(128, 128)):
    """Recorte BGR -> RGB float32 en [0, 1] de tamano size (ancho, alto).

    Dividir por 255 deja las entradas en [0, 1], la escala para la que estan pensadas las
    inicializaciones habituales de los pesos; con pixeles en [0, 255] los gradientes iniciales
    serian mucho mayores.
    """
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
