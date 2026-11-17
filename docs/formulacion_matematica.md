# Formulación y fundamentos matemáticos (PI3)
Borrador de la estructura que pide el documento de competencias. Las ecuaciones son las que corresponden al pipeline
planeado; hay que ajustarlas a lo que finalmente se implemente y completar con los resultados.

## 1. Formulación del problema
**Datos.** D = {(xᵢ, yᵢ, dᵢ, gᵢ)}ᵢ₌₁ᴺ, donde
- xᵢ ∈ ℝ^{H×W×3}: imagen RGB de una fruta;
- yᵢ ∈ {1, …, K}: clase de calidad, K ≥ 3 y ordinal (buena > regular > mala);
- dᵢ ∈ ℝ⁺: diámetro real en mm (solo disponible en las frutas propias);
- gᵢ: grupo de la fruta física (`fruit_id` o grupo de duplicados).

**Tareas.**
1. Calidad: aprender f_θ : ℝ^{H×W×3} → Δ^{K−1} (distribución sobre las clases) y predecir ŷ = argmin_j Σ_k p_θ(k|x)·C[k, j],
   donde C es la matriz de costos (`configs/config.yaml`). Si C es la matriz 0-1, esto se reduce a ŷ = argmax_k p_θ(k|x).
2. Tamaño: segmentar la máscara M(x) ⊂ Ω, calcular el diámetro equivalente y discretizarlo (sección 3.4).

**Objetivo.** θ* = argmin_θ E_{(x,y)}[ℓ(f_θ(x), y)], estimado con riesgo empírico regularizado sobre train
y validado sobre grupos que el modelo no vio (split agrupado por g).

**Por qué es un problema complejo de ingeniería** (argumentar con datos del EDA):
- Entrada de alta dimensión (128·128·3 ≈ 49 000 variables) con pocos ejemplos propios.
- Gran variabilidad dentro de cada clase (madurez, variedad, iluminación) y fronteras difusas entre clases ordinales.
- Etiquetas subjetivas, que se cuantifican con kappa de Cohen.
- Cambio de dominio: India (Kaggle) frente a Colombia (fotos propias); desbalanceo de clases.
- Costos de error asimétricos y dos salidas con naturaleza distinta (clasificación y medición).

## 2. Estrategia de solución
| Enfoque | Justificación | Alternativas descartadas y motivo |
|---|---|---|
| Características manuales φ(x) + SVM / Random Forest | Con pocos datos, un sesgo inductivo fuerte (color y textura codifican los defectos) generaliza mejor; además es interpretable. | KNN: sensible a la escala y a la dimensión. Árbol único: alta varianza. |
| CNN pequeña desde cero | Aprende las características de forma jerárquica; la convolución explota la invarianza local a traslación. | CNN profunda desde cero: sobreajusta con este volumen de datos. Transfer learning: permitido, pero el enunciado prioriza modelos desde cero. |
| Tamaño por segmentación + geometría | Es medible y explicable, y no requiere etiquetas de tamaño para entrenar. | Regresión del tamaño con una CNN: no hay suficientes diámetros medidos. |

## 3. Fundamentos aplicados (justificar cada paso)
### 3.0 Grupos de casi-duplicados (implementado en `src/data/dedup.py`)
Cada imagen se resume en un pHash h(x) ∈ {0,1}⁶⁴: la DCT de la imagen reducida a 32×32 en grises, comparando los 8×8 coeficientes
de baja frecuencia con su mediana. Dos imágenes son "la misma fruta" si la distancia de Hamming d_H(h(x), h(x′)) ≤ τ, con τ = 6.
Los grupos son las componentes conexas de esa relación (union-find), y g = grupo. El split agrupado garantiza que un mismo g no
quede en train y test a la vez. Efecto medido: Apple_Bad pasa de 1.141 imágenes a 378 grupos.

### 3.1 Segmentación
Implementado en `src/data/segment.py`. Se estima el color del fondo b como la mediana, en Lab, del marco exterior de la imagen
(5 % del lado menor; el protocolo deja margen alrededor de la fruta). Para cada píxel se calcula
D(u) = √( (w_L·ΔL)² + Δa² + Δb² ), con w_L = 0,5: se le resta peso a la luminosidad para que las sombras, que oscurecen
el fondo sin cambiarle el color, no cuenten como fruta. Sobre D se aplica el umbral de Otsu, que elige el t que maximiza
la varianza entre clases σ_B²(t) = ω₀(t)·ω₁(t)·[μ₀(t) − μ₁(t)]². Si ese t es menor que 12 unidades Lab, no hay nada
distinto del fondo. Después vienen apertura y cierre morfológicos, el relleno de huecos (brillos de la piel) y las
componentes conexas con área ≥ 1 % de la imagen. Se priorizan las que no tocan el borde.
Antes se detecta el cuadrado de referencia (contorno de 4 vértices, convexo, aspecto 0,85–1,15, que llena ≥ 90 % de su
rectángulo mínimo, que no toca el borde, que ocupa ≤ 20 % de la imagen y con interior oscuro y uniforme: gris medio
≤ 90 y desviación ≤ 25) y se borra para que no salga como fruta. Lado = promedio de los lados de su rectángulo mínimo + 1 px.
Validación: con una imagen sintética (disco de 300 px y cuadrado de 200 px = 50 mm, fondo con sombra y brillo en la fruta)
el diámetro estimado fue 75,0 mm frente a 75 mm. Pruebas automáticas en `tests/test_segment.py` y `tests/test_size_estimator.py`. En fotos de Kaggle funciona con fondos uniformes y falla con fondos texturados (madera, césped)
o cuando hay una mano. Las frutas que se tocan quedan unidas: se probó separar con watershed sobre la transformada de
distancia y no fue confiable, así que se descartó.

### 3.2 Características y normalización
Implementado en `src/data/features.py`: 54 valores calculados solo dentro de la componente mayor de la máscara. Color y textura
se miden con el recorte llevado a 128 px de lado mayor (LBP compara vecinos a 1 px, así que solo es comparable a la misma
escala), y la textura excluye el borde de la máscara, donde los vecinos son fondo. La forma se mide en la máscara original.
- Histograma HSV normalizado (H 16 bins, S 8, V 8): h_b = n_b / Σ n_b. Se usa HSV porque separa el tono de la iluminación (V).
- Momentos de color en Lab por canal: media μ, desviación σ y asimetría E[(x − μ)³]/σ³. La asimetría capta colas, como pocas manchas oscuras sobre una piel clara.
- Textura LBP uniforme (P = 8, R = 1): cada píxel se codifica comparando sus 8 vecinos con él; el histograma de 10 patrones resume la textura. Los golpes y el moho rompen la textura lisa.
- Forma: circularidad 4πA/P², solidez A/A_casco y relación de aspecto. El área no se usa: no es calidad y en Kaggle depende de la distancia.
- Estandarización z = (φ − μ_train) / σ_train, con μ y σ calculados **solo con train** (si no, hay fuga).
  Es necesaria para el SVM con kernel RBF, que depende de distancias euclidianas.

### 3.3 Modelos
- **SVM con margen suave:** min ½‖w‖² + C·Σ ξᵢ, sujeto a yᵢ(wᵀϕ(xᵢ) + b) ≥ 1 − ξᵢ, con kernel k(x, x′) = exp(−γ‖x − x′‖²).
  Para K clases se usa un esquema uno contra uno. Hiperparámetros: C y γ.
- **Random Forest:** divisiones que minimizan la impureza de Gini, G = 1 − Σ_k p_k². El bagging reduce la varianza.
- **CNN:** entropía cruzada ponderada L = −(1/N) Σᵢ w_{yᵢ} log p_θ(yᵢ|xᵢ) + λ‖θ‖², con w_k = N / (K·N_k) para compensar el desbalanceo.
  Se optimiza con Adam; dropout y early stopping (sobre la pérdida de validación) controlan el sobreajuste.

### 3.4 Tamaño
- Área de la máscara A (píxeles) → diámetro equivalente d_px = 2·√(A/π).
- Calibración con un objeto de referencia de diámetro D_ref (mm) que mide d_ref píxeles: s₀ = D_ref / d_ref (mm/px en el plano de la mesa).
- Corrección de perspectiva (modelo de cámara estenopeica, vista cenital a altura H). Un objeto a altura h sobre la mesa se ve con escala
  s(h) = s₀·(H − h)/H. El contorno más ancho de la manzana está a h ≈ d/2, así que d = d_px·s₀·(H − d/2)/H; despejando,
  d̂_mm = d_px·s₀ / (1 + d_px·s₀/(2H)). Sin corregir, el error relativo es ≈ d/(2H) (≈ 12 % con d = 75 mm y H = 30 cm).
- Error frente al diámetro medido: MAE = (1/n) Σ |d̂ᵢ − dᵢ| y error relativo medio.
- Clases de tamaño por especie: umbrales en los terciles q₁/₃ y q₂/₃ de d̂ calculados en train.

### 3.5 Métricas y validación
- F1 por clase F1_k = 2·P_k·R_k / (P_k + R_k); **F1 macro** = (1/K) Σ F1_k. Es la métrica principal porque, a diferencia de la accuracy, no la domina la clase mayoritaria.
- Recall de "mala" y costo esperado (1/N) Σᵢ C[yᵢ, ŷᵢ].
- Validación cruzada estratificada y agrupada (StratifiedGroupKFold) para ajustar hiperparámetros; el test se usa **una sola vez**.
- Acuerdo entre anotadores: κ = (p_o − p_e) / (1 − p_e).
- Comparación de modelos sobre el mismo test: prueba de McNemar χ² = (|b − c| − 1)² / (b + c), o intervalos de confianza bootstrap del 95 % para el F1 macro.

## 4. Análisis de resultados
Brecha de generalización (F1_train − F1_val), curvas de aprendizaje y de pérdida, desempeño por fuente (Kaggle frente a propias),
matriz de confusión: ¿los errores ocurren entre clases vecinas (buena↔regular) o extremas (buena↔mala)?

## 5. Conclusiones técnicas
Conectar cada resultado con la formulación: qué supuesto se cumplió o no (p. ej. separabilidad en el espacio de color)
y qué mejora sugiere la teoría (más datos de la clase regular, una pérdida ordinal, calibración de probabilidades).
