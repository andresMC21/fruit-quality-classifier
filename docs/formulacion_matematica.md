# Formulación y fundamentos matemáticos (PI3)
> **Estado:** la estructura sigue el pipeline implementado (`src/`). Valores reales: 59 características (`features.py`), SVM C = 10, γ = 0,03; CNN de 3 convoluciones con entrada 64x64 y dos salidas (especie y calidad). El modelo de calidad recibe además la especie en one-hot. Las secciones 4 y 5 se completan con el test.

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
- Cambio de dominio: entre fuentes (India/Kaggle, Fruit grade, Bangladesh/Sultana); medido con el test externo (F1 0,889 → 0,600 en el SVM); desbalanceo de clases.
- Costos de error asimétricos y dos salidas con naturaleza distinta (clasificación y medición).

## 2. Estrategia de solución
| Enfoque | Justificación | Alternativas descartadas y motivo |
|---|---|---|
| Características manuales φ(x) + SVM / Random Forest | Con pocos datos, un sesgo inductivo fuerte (color y textura codifican los defectos) generaliza mejor; además es interpretable. | KNN: sensible a la escala y a la dimensión. Árbol único: alta varianza. |
| CNN pequeña desde cero | Aprende las características de forma jerárquica; la convolución explota la invarianza local a traslación. | CNN profunda desde cero: sobreajusta con este volumen de datos. Transfer learning: permitido, pero el enunciado prioriza modelos desde cero. |
| Tamaño por segmentación + geometría | Es medible y explicable, y no requiere etiquetas de tamaño para entrenar. | Regresión del tamaño con una CNN: no hay suficientes diámetros medidos. |

## 3. Fundamentos aplicados (justificar cada paso)
### 3.1 Segmentación
Umbral de Otsu sobre un canal (p. ej. S de HSV o a\* de Lab). Se elige el t que maximiza la varianza entre clases:
σ_B²(t) = ω₀(t)·ω₁(t)·[μ₀(t) − μ₁(t)]².
Después se aplican operaciones morfológicas (apertura y cierre) y se conserva la componente conexa de mayor área.

### 3.2 Características y normalización
- Histograma HSV normalizado: h_b = n_b / Σ n_b. Se usa HSV porque separa el tono de la iluminación (V).
- Textura LBP o GLCM (contraste, homogeneidad); forma (circularidad 4πA/P², solidez).
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
- Calibración con un objeto de referencia de diámetro D_ref (mm) que mide d_ref píxeles: d̂_mm = d_px · D_ref / d_ref.
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
