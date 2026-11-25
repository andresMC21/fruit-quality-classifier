# Impactos (PI2)
Se piden dos versiones: un **análisis inicial** (antes de modelar) y un **análisis final** (con los resultados).
Cada celda debe apoyarse en un dato o un argumento técnico, no en una opinión.

| Dimensión | Impactos positivos | Riesgos | Mitigación | Evidencia o dato que la respalda |
|---|---|---|---|---|
| **Social** | Estandariza la calidad; consumidor mejor informado. | Desplazamiento de clasificadores manuales; desconfianza de pequeños productores; fotos en mercados (privacidad). | Herramienta de apoyo, no reemplazo; criterios transparentes; protocolo de captura. | |
| **Económica** | Menos pérdida por mala clasificación; hardware barato (celular o webcam). | Errores con costo asimétrico (ver la matriz de costos); el modelo no transfiere a otra variedad sin reentrenar. | Elegir el umbral según el costo; reentrenar con datos locales. | Costo esperado en validación: SVM 0,133 frente a 1,520 de la clase mayoritaria y 0,296 de la regresión logística (`05_evaluacion.md`). |
| **Ambiental** | Reduce el desperdicio de alimentos. | Energía de entrenamiento e inferencia; residuos electrónicos si se usa hardware dedicado. | Preferir el modelo más liviano si el F1 es similar; inferencia en CPU. | Tiempo de ajuste en CPU de 2 núcleos: logreg 3 s, SVM 11 s, Random Forest 81 s, CNN 689 s (`04_modelado.md`); el SVM rinde más que la CNN con ~60 veces menos cómputo. No se midió energía (recortado en el plan). |
| **Global** | Aplicable donde hay celulares y no hay equipos industriales. | Normas de calidad distintas por país (NTC de Icontec, estándares internacionales); variedades locales; datos entrenados en otro país. | Clases configurables; documentar el alcance; validar con datos de cada región. | Desempeño por fuente en validación: Kaggle 0,965 vs Fruit grade 0,812 de F1; pendiente el contraste con imágenes propias (Colombia). |


## Análisis inicial vs. final
Las columnas "Evidencia" con cifras corresponden al **análisis inicial** (validación). El **análisis final** se completa con el test y el test externo (`05_evaluacion.md`); no hay fotos propias ni MAE en mm. Cifra clave para las cuatro dimensiones: en otra fuente de fotos, el SVM detecta solo 45 % de la fruta mala (RF 78 %), por lo que el sistema es un apoyo, no un sustituto de la revisión humana.
