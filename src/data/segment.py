"""Fase 3: segmentacion de frutas individuales (carpeta mixed_quality y fotos propias).
Idea base: umbral de color/Otsu sobre fondo uniforme + componentes conexas + recorte por bounding box.
Devuelve ademas el area de la mascara, usada para estimar el tamano.

Pasos (justificacion en docs/formulacion_matematica.md, seccion 3.1):
1. Color del fondo = mediana del borde de la imagen en Lab (el protocolo deja margen alrededor de la fruta).
2. Distancia de color de cada pixel al fondo, con menos peso en L para que las sombras no cuenten como fruta.
3. Umbral de Otsu sobre esa distancia, apertura/cierre morfologico y relleno de huecos (brillos de la piel).
4. Componentes conexas con area >= min_area_frac de la imagen, de mayor a menor.
Antes se borra el cuadrado de referencia (find_reference_square) para que no aparezca como fruta.
"""
import cv2
import numpy as np
from scipy import ndimage

L_WEIGHT = 0.5        # peso de la luminosidad en la distancia de color
MIN_CONTRAST = 12.0   # distancia media minima (unidades Lab) del primer plano al fondo
BORDER_FRAC = 0.05    # ancho del marco usado para estimar el fondo
REF_MAX_GRAY = 90     # el interior del cuadrado impreso es oscuro...
REF_MAX_STD = 25      # ...y uniforme; una manzana oscura tiene brillos y textura


def as_bgr(img):
    """Acepta imagenes en grises (2D) o BGRA y las devuelve en BGR."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def border_median(img):
    """Mediana por canal del marco exterior (BORDER_FRAC del lado menor)."""
    h, w = img.shape[:2]
    b = max(1, int(BORDER_FRAC * min(h, w)))
    c = img.shape[2]
    border = np.concatenate([img[:b].reshape(-1, c), img[-b:].reshape(-1, c),
                             img[:, :b].reshape(-1, c), img[:, -b:].reshape(-1, c)])
    return np.median(border, axis=0)


def background_distance(img_bgr):
    """Distancia de color (Lab ponderado) de cada pixel a la mediana del borde."""
    lab = cv2.cvtColor(as_bgr(img_bgr), cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab - border_median(lab)
    diff[..., 0] *= L_WEIGHT
    return np.sqrt((diff ** 2).sum(axis=2))


def foreground_mask(img_bgr):
    """Mascara binaria (uint8 0/1) de todo lo que no es fondo; vacia si no hay contraste."""
    dist = background_distance(img_bgr)
    scaled = np.clip(dist * 255.0 / max(dist.max(), 1e-6), 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(scaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Se mide el contraste del primer plano y no el umbral: con una imagen perfectamente
    # bimodal Otsu puede devolver t = 0 aunque la fruta se distinga sin problema.
    if not mask.any() or dist[mask == 1].mean() < MIN_CONTRAST:
        return np.zeros(dist.shape, np.uint8)
    k = max(3, int(0.01 * min(mask.shape)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return ndimage.binary_fill_holes(mask).astype(np.uint8)


def touches_border(x, y, w, h, shape):
    return x == 0 or y == 0 or x + w >= shape[1] or y + h >= shape[0]


def find_reference_square(img_bgr, min_area_frac=0.002, max_area_frac=0.2):
    """Busca el cuadrado oscuro impreso de referencia.

    Devuelve (lado_px, contorno) del cuadrado mas grande, o None si no hay. Criterios: contorno de
    4 vertices, convexo, relacion de aspecto 0.85-1.15, que llene >= 90 % de su rectangulo minimo,
    que no toque el borde, que ocupe como mucho max_area_frac de la imagen (si no, un fondo oscuro
    entero pasaria por "cuadrado") y con interior oscuro y uniforme (para no confundirlo con una
    manzana oscura de silueta cuadrada).
    """
    gray = cv2.cvtColor(as_bgr(img_bgr), cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    min_area, max_area = min_area_frac * gray.size, max_area_frac * gray.size
    for c in contours:
        area = cv2.contourArea(c)
        if not min_area <= area <= max_area or touches_border(*cv2.boundingRect(c), gray.shape):
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        if not 0.85 <= rw / max(rh, 1e-6) <= 1.15 or area / max(rw * rh, 1e-6) < 0.9:
            continue
        inside = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(inside, [c], -1, 1, thickness=cv2.FILLED)
        inside = cv2.erode(inside, np.ones((5, 5), np.uint8))  # sin el borde antialias
        mean, std = cv2.meanStdDev(gray, mask=inside)
        if mean[0, 0] > REF_MAX_GRAY or std[0, 0] > REF_MAX_STD:
            continue
        if best is None or area > best[0]:
            # +1: el contorno pasa por los centros de los pixeles del borde, a (lado - 1) de distancia
            best = (area, c, (rw + rh) / 2.0 + 1.0)
    if best is None:
        return None
    return float(best[2]), best[1]


def segment_image(img_bgr, min_area_frac=0.01):
    """Devuelve lista de (recorte, mascara, area_px).

    recorte: BGR de la caja envolvente; mascara: uint8 0/1 del mismo tamano que el recorte;
    area_px: pixeles de la mascara. Primero van las componentes que no tocan el borde de la imagen
    (el protocolo deja la fruta centrada) y, dentro de cada grupo, de mayor a menor area.
    Frutas que se tocan quedan en una sola componente: no sirve para separar fotos de mixed_quality.
    Lista vacia si no se encuentra nada sobre el fondo.
    """
    img_bgr = as_bgr(img_bgr)
    img = img_bgr.copy()
    ref = find_reference_square(img)
    if ref is not None:
        # Se pinta la referencia con el color del fondo para que no salga como fruta; el trazo
        # grueso cubre el halo borroso del borde, que crece con la resolucion.
        color = border_median(img).tolist()
        cv2.drawContours(img, [ref[1]], -1, color, thickness=cv2.FILLED)
        cv2.drawContours(img, [ref[1]], -1, color, thickness=max(3, int(0.05 * ref[0])))

    mask = foreground_mask(img)
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    min_area = min_area_frac * mask.size
    fruits = []
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(comp)
        crop = (img_bgr[y:y + h, x:x + w].copy(), comp[y:y + h, x:x + w], area)
        fruits.append((touches_border(x, y, w, h, mask.shape), crop))
    fruits.sort(key=lambda f: (f[0], -f[1][2]))
    return [crop for _, crop in fruits]
