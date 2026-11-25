# Clasificación automática de calidad y tamaño de frutas con visión por computadora
*Borrador (7 páginas máx. en la plantilla IEEE `conference-template-letter.docx`). Las marcas **[PENDIENTE]** son datos que solo el grupo puede aportar. No hay fotos propias.*

## Resumen
Se desarrolló un sistema que, a partir de la foto de una fruta individual, predice su especie (manzana, banano, mango, guayaba, naranja, lima, uva o fresa), su calidad (buena, regular o mala) y su tamaño relativo. Se combinaron tres bases públicas (Kaggle *Fruit Quality Classification*, Fruit grade/AFruitDB y Sultana) con segmentación, deduplicación y un split agrupado por fruta, y se reservó una fuente (Sultana: manzana, banano, naranja, guayaba) como test externo. Se compararon una línea base, regresión logística, SVM, Random Forest y una CNN pequeña. En test, el SVM con 59 características obtuvo F1 macro 0,889 (IC 95 % 0,863-0,912) y recall de "mala" 0,959, frente a 0,235 de la clase mayoritaria; **pero en la fuente externa cayó a 0,600 (recall de "mala" 0,45) y el Random Forest resultó más robusto (0,764)**. El sistema se despliega en una app Streamlit con cámara.

## 1. Introducción
La clasificación manual de fruta fresca es lenta y subjetiva, y sus errores tienen costo asimétrico: descartar fruta buena genera desperdicio, y vender fruta mala genera riesgo para el consumidor. El profesor pidió un clasificador de varias frutas comunes; con los datos disponibles se cubren ocho (manzana, banano, mango, guayaba, naranja, lima, uva, fresa) y se declaran fuera de alcance las demás (pera, piña, sandía, mandarina, mora, durazno, limón). El problema es difícil por la variedad de fondos y cámaras entre bases, las fronteras difusas de la clase "regular" y el cambio de dominio frente a fotos locales.

## 2. Fundamentos teóricos
Solo lo necesario para leer el resto: segmentación por umbral de Otsu [8] y GrabCut [9]; descriptores de color HSV/Lab, textura LBP [10] y GLCM; SVM con kernel RBF y Random Forest [13]; CNN con entropía cruzada ponderada; F1 macro, recall de la clase crítica y costo esperado con matriz de costos; kappa de Cohen [11] y McNemar [12]. Formulación en `docs/formulacion_matematica.md`.

## 3. Metodología
CRISP-DM adaptado (figura `docs/diagramas/crisp_dm_personalizado.pdf`).
- **Datos:** 15 309 imágenes de 8 especies y 3 fuentes; la clase "regular" solo viene de Fruit grade; uva y fresa solo de Sultana. Etiquetas públicas débiles (de la carpeta o del grado). Se evaluaron y descartaron FruQ-DB/FruQ-multi (fotogramas de videos de una misma fruta, nombres de carpeta que no coinciden con la fruta) y FruityHub (fotogramas con contador de días superpuesto que delata la clase).
- **Deduplicación y split:** pHash más ráfagas dan 8 340 grupos; se limitan a 3 imágenes por grupo (10 647) y se parte por grupo, estratificado por especie y calidad: train 5 784, val 1 139, test 1 162 y externo 1 493. Sin fuga entre splits (verificado en pruebas).
- **Segmentación y recorte:** Otsu con respaldo GrabCut; fondo reemplazado por gris para reducir el atajo de la fuente; fallos 0-29 % por especie/fuente (fresa de Sultana la peor).
- **Modelos:** dos clasificadores en cascada (especie, y calidad con la especie como entrada); SVM, Random Forest y CNN de 3 convoluciones; hiperparámetros por validación cruzada agrupada.
- **Tamaño:** diámetro equivalente de la máscara; en mm con objeto de referencia; si no, relativo al encuadre (terciles por especie y fuente). Sin fotos propias no hay mm verificados ni MAE; uva y fresa no tienen umbrales.
- **Etiquetas:** débiles; sin anotación propia no se calculó kappa.

## 4. Resultados
Validación (F1 macro): mayoritaria 0,237; regresión logística 0,756; Random Forest 0,873; CNN 0,806; **SVM 0,903**. Exactitud de especie en validación: 96,2 %.

**Test público** (una sola evaluación; IC 95 % bootstrap): mayoritaria 0,235; regresión logística 0,774 [0,741-0,802]; Random Forest 0,867 [0,838-0,892]; **SVM 0,889 [0,863-0,912]**; McNemar SVM vs RF p = 4e-4. Por fuente (SVM): Kaggle 0,972, Sultana 0,969, Fruit grade 0,824.

**Test externo** (Sultana, 1 493 imágenes): mayoritaria 0,322; regresión logística 0,632; SVM 0,600 (recall "mala" 0,45); **Random Forest 0,764** (recall "mala" 0,78). Exactitud de especie 54,6 %.

Tablas y matrices de confusión: `docs/05_evaluacion.md`, `experiments/results/tables/`. **[PENDIENTE: figuras para el informe (matriz de confusión, curvas de la CNN)]**

## 5. Análisis de resultados
- Las características manuales con SVM superan a la CNN (64x64, desde cero, CPU) en validación.
- El F1 global esconde diferencias grandes entre fuentes: las especies con "regular" (manzana, banano, mango) rinden menos porque la frontera entre clases es difusa.
- **Atajo de fuente confirmado por el test externo:** la fuente se predice con 98,8 % de exactitud desde las características, y al cambiar de fuente el SVM cae de 0,889 a 0,600; falla sobre todo detectando fruta mala (manzana 18 %, naranja 8 %). El Random Forest generaliza mejor (0,764), aunque también cae. El F1 del test público sobreestima el desempeño con fotos nuevas. El clasificador de especie tampoco transfiere (54,6 %).
- Uva y fresa solo existen en una fuente: sus métricas buenas no prueban generalización.
- Decisión pendiente: el modelo por defecto de la app es el SVM (mejor en validación y test); la evidencia externa favorece al Random Forest. Elegirlo ahora con el test externo ya visto sería selección con datos de evaluación, y debe declararse así.
- Limitación: la deduplicación no enlaza vistas distintas de una misma manzana en Fruit grade, por lo que puede haber fuga residual en esa fuente.
- **[PENDIENTE]** Comparación con la literatura (FruitNet, Sultana et al.) y errores con ejemplos visuales.

## 6. Ética e impactos
Tabla de decisiones en `docs/etica.md` (9 filas: privacidad en mercados, sesgo por origen de los datos, atajo de fuente, costo asimétrico, honestidad de métricas, frutas fuera de alcance) y matriz social/económica/ambiental/global en `docs/impactos.md` con evidencia cuantitativa (costo esperado, tiempos de cómputo).

## 7. Conclusiones y trabajo futuro
Se construyó un pipeline reproducible de datos a app con evaluación agrupada y por fuente. La evaluación en otra fuente mostró que el rendimiento en test público es optimista. Trabajo futuro: fotos propias con las tres clases (en especial "regular"), entrenar con varias fuentes por especie, aumento de datos o normalización de color para reducir el atajo de fuente, validar la detección de "fruta no reconocida" con frutas ajenas, y ampliar a las especies faltantes.

## Referencias
Ver `docs/referencias.md` (formato IEEE). **Verificar** [3], [7] (FruitNet y Fruit grade) y [15] (Sultana) antes de entregar.
