# Fase 6 - Despliegue
- **Pipeline único:** `src/inference/predict.py` (segmentar → recorte enmascarado → características → especie → calidad → tamaño). Lo usan la app y la evaluación.
- **Modelo exportado:** `models/` (`python -m src.main export`): SVM de especie, SVM/RF/regresión logística de calidad, `meta.json` con clases, tamaño de recorte y umbrales de tamaño.
- **App:** `streamlit run src/app/streamlit_app.py`. Cámara (`st.camera_input`) o archivo, selector de modelo, objeto de referencia opcional en mm, imagen del recorte que ve el modelo, probabilidades y avisos.
- **Rechazos explícitos:** segmentación fallida → pide repetir la foto; especie con confianza < 60 % → "fruta no reconocida"; calidad con confianza < 50 % → revisión humana.
- **Aviso de robustez:** en fotos de otra fuente el SVM pierde mucho recall de "mala" (0,45); la app debe mostrarse como prototipo y el modelo por defecto está pendiente de decisión (`05_evaluacion.md`).
- **Tamaño:** sin objeto de referencia es relativo al encuadre; con referencia, en mm. Los umbrales relativos se calculan con la fuente más parecida al protocolo (Fruit grade para manzana, banano y mango; Kaggle para guayaba, naranja y lima; uva y fresa sin umbrales: la app solo muestra el tamaño relativo).
- **Extensión opcional (no implementada):** video en vivo con `streamlit-webrtc`, Raspberry Pi, API FastAPI + Docker (ver `arquitectura.md`).
