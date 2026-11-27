"""Fase 6: interfaz Streamlit (cámara o carga de imagen) que muestra especie, calidad y tamaño.
Ejecutar desde la raíz del repo:  streamlit run src/app/streamlit_app.py

Modo local (por defecto): usa src/inference/predict.py directamente.
Modo cliente: si existe la variable de entorno API_URL, envía la imagen a un backend FastAPI (POST /predict).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
import streamlit as st

from src.inference.predict import load_model, predict

st.set_page_config(page_title="Calidad de frutas", page_icon="🍎")


@st.cache_resource
def get_model():
    return load_model("models")


model = get_model()
meta = model["meta"]
names = {"svm": "SVM (RBF)", "rf": "Random Forest", "logreg": "Regresión logística (línea base)"}

st.title("Clasificador de calidad de frutas")
st.caption("Prototipo académico: no certifica calidad comercial. "
           f"Frutas soportadas: {', '.join(meta['species'])}.")
with st.expander("Alcance validado y limitaciones"):
    st.markdown(
        "- Entrenado con fotos de India, Bangladesh y Brasil; **no validado con fruta colombiana**.\n"
        "- **Mango:** se reconoce el mango asiático verde o amarillo; el **mango rojo-amarillo (tipo Tommy) puede leerse como manzana**.\n"
        "- **Guayaba:** fiable solo en fotos parecidas a las de entrenamiento (14 % en otra fuente).\n"
        "- **Uva:** grano suelto o racimo; lima y fresa vienen de una sola fuente.\n"
        "- **Calidad:** 'regular' solo en manzana, banano y mango; fuera de dominio detecta ~75 % de la fruta mala.\n"
        "- **Tamaño** relativo al encuadre (mm solo con objeto de referencia). Una fruta por foto, fondo liso.")

if model.get("species_warning"):
    st.warning(model["species_warning"])

with st.sidebar:
    qkey = st.selectbox("Modelo de calidad", list(model["qualities"]), format_func=lambda k: names.get(k, k),
                        index=list(model["qualities"]).index(meta.get("default_quality", "svm")))
    ref_mm = st.number_input("Diámetro del objeto de referencia (mm, 0 = sin referencia)", 0.0, 200.0, 0.0, 0.1,
                             help="Moneda o tapa colocada junto a la fruta, en el mismo plano. Con ella se estima el diámetro en mm.")
    st.markdown("**Cómo tomar la foto:** una sola fruta, fondo liso (cartulina blanca o gris), buena luz, sin manos.")

src = st.radio("Origen de la imagen", ["Cámara", "Subir archivo"], horizontal=True)
f = st.camera_input("Capture la fruta") if src == "Cámara" else st.file_uploader("Cargue una imagen", type=["jpg", "jpeg", "png"])

if f is not None:
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    with st.spinner("Analizando…"):
        p = predict(img, model, ref_diameter_mm=ref_mm or None, quality_model=qkey)
    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Imagen original", width="stretch")
    if p.crop is not None:
        c2.image(cv2.cvtColor(p.crop, cv2.COLOR_BGR2RGB), caption="Fruta aislada (entrada del modelo)", width="stretch")
    if not p.segmentation_ok or p.species is None:
        st.warning(p.message)
    else:
        st.subheader(f"{p.species} · calidad: {p.quality}")
        st.caption(f"Confianza de especie: {p.species_confidence:.0%}")
        st.bar_chart({k: v for k, v in p.probabilities.items()})
        if p.needs_review:
            st.warning(p.message)
        size_txt = (p.size_class or "sin umbrales") + (f" · ≈ {p.diameter_mm:.0f} mm" if p.diameter_mm else
                                                       f" (relativo al encuadre: {p.relative_diameter:.2f})")
        st.info(f"Tamaño: {size_txt}")
        if not p.diameter_mm:
            st.caption("Sin objeto de referencia el tamaño es relativo a la foto, no una medida en milímetros.")
