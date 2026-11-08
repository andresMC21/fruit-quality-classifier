"""Fase 6: interfaz Streamlit (carga de imagen o st.camera_input) que muestra calidad y tamano predichos.
Ejecutar desde la raiz del repo: streamlit run src/app/streamlit_app.py

st.camera_input toma una foto (minimo viable). Para ver la prediccion en vivo, como pide la fase 6,
la extension es streamlit-webrtc o un bucle con cv2.VideoCapture.
"""
import streamlit as st

st.title("Clasificador de calidad de frutas")
st.caption("Prototipo académico: no certifica calidad comercial.")

img = st.camera_input("Capture la fruta") or st.file_uploader("...o cargue una imagen", type=["jpg", "jpeg", "png"])
if img:
    st.image(img)
    # TODO: decodificar a BGR, llamar a src.inference.predict.predict(img_bgr, model) y mostrar
    # clase, probabilidades, tamano y la mascara; si segmentation_ok es False o la confianza es baja,
    # pedir otra foto o revision humana.
    st.info("Pendiente: conectar con src/inference/predict.py")
