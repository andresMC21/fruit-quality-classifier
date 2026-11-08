"""Fase 3: segmentación de frutas individuales (carpeta mixed_quality y fotos propias).
Idea base: umbral de color/Otsu sobre fondo uniforme + componentes conexas + recorte por bounding box.
Devuelve además el área de la máscara, usada para estimar el tamaño.
"""

def segment_image(img_bgr):
    """TODO: devolver lista de (recorte, máscara, área_px)."""
    raise NotImplementedError
