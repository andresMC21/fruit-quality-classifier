# Impactos (PI2)
Se piden dos versiones: un **análisis inicial** (antes de modelar) y un **análisis final** (con los resultados).
Cada celda debe apoyarse en un dato o un argumento técnico, no en una opinión.

| Dimensión | Impactos positivos | Riesgos | Mitigación | Evidencia o dato que la respalda |
|---|---|---|---|---|
| **Social** | Estandariza la calidad; consumidor mejor informado. | Desplazamiento de clasificadores manuales; desconfianza de pequeños productores; fotos en mercados (privacidad). | Herramienta de apoyo, no reemplazo; criterios transparentes; protocolo de captura. | |
| **Económica** | Menos pérdida por mala clasificación; hardware barato (celular o webcam). | Errores con costo asimétrico (ver la matriz de costos); el modelo no transfiere a otra variedad sin reentrenar. | Elegir el umbral según el costo; reentrenar con datos locales. | Costo esperado del modelo frente a la línea base, calculado con la matriz de costos. |
| **Ambiental** | Reduce el desperdicio de alimentos. | Energía de entrenamiento e inferencia; residuos electrónicos si se usa hardware dedicado. | Preferir el modelo más liviano si el F1 es similar; inferencia en CPU. | Medir energía y CO₂ por modelo (p. ej. con `codecarbon`) y el tiempo de inferencia. |
| **Global** | Aplicable donde hay celulares y no hay equipos industriales. | Normas de calidad distintas por país (NTC de Icontec, estándares internacionales); variedades locales; datos entrenados en otro país. | Clases configurables; documentar el alcance; validar con datos de cada región. | Desempeño por fuente: Kaggle (India) frente a imágenes propias (Colombia). |
