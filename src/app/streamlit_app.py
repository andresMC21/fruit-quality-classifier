"""Fase 6: interfaz Streamlit (carga de imagen o st.camera_input) que muestra calidad y tamaño predichos.
Ejecutar: streamlit run src/app/streamlit_app.py
"""
import streamlit as st

st.title("Clasificador de calidad de frutas")
img = st.camera_input("Capture la fruta") or st.file_uploader("...o cargue una imagen", type=["jpg", "jpeg", "png"])
if img:
    st.image(img)
    st.info("TODO: segmentar -> predecir calidad y tamaño -> mostrar resultado y confianza.")
