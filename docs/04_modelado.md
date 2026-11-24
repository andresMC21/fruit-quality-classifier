# Fase 4 - Modelado
Todo se ajusta con validación cruzada agrupada (StratifiedGroupKFold, 4 pliegues, solo sobre train); el test no se toca. Registro: `experiments/results/tables/experiments_log.csv`.

## Diseño
- **Dos clasificadores:** especie (solo características visuales) y calidad (características + especie en one-hot). En la app, la especie predicha alimenta a la de calidad.
  Si la confianza de especie < 0,60 la app responde "fruta no reconocida".
- **Modelos de ML tradicional (2) + línea base (2):** mayoritaria, regresión logística, SVM (RBF), Random Forest. `class_weight` balanceado.
- **Deep learning (1):** CNN pequeña desde cero, 3 convoluciones (32/64/128) + BN + pooling global + densa, entrada 64x64, dos salidas (especie y calidad; 8 especies).
  Aumento leve (giros, rotación 5°, zoom 5 %, brillo 10 %), Adam con coseno 2e-3 → 4e-5, parada temprana en la pérdida de validación, 30 épocas, pesos por clase.
- No se usó transfer learning ni XGBoost/MLP (recortados en el plan).

## Rejillas y mejores hiperparámetros (CV agrupada, F1 macro)
| Modelo | Rejilla | Mejor | F1 macro CV |
|---|---|---|---|
| Regresión logística | C ∈ {0,1; 1; 10} | C = 1 | 0,776 |
| **SVM RBF** | C ∈ {1; 10; 100}, γ ∈ {scale; 0,003; 0,03} | C = 10, γ = 0,03 | **0,904** |
| Random Forest | árboles {200; 400}, profundidad {None; 20}, hoja mín. {1; 3} | 400, 20, 3 | 0,878 |
| SVM de especie | misma rejilla | C = 100, γ = scale | 0,955 |

## Tiempos de ajuste (CPU de 2 núcleos)
Logreg 2 s, SVM 14 s, Random Forest 92 s, CNN 596 s (30 épocas). Es el dato disponible para el análisis ambiental (`impactos.md`).

## Selección
El SVM tiene el mejor F1 macro en validación (0,903) y en test (0,889); **pero el Random Forest es más robusto en otra fuente** (externo: 0,764 frente a 0,600; ver `05_evaluacion.md`). Los tres modelos clásicos quedan en el selector de la app.
