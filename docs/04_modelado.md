# Fase 4 - Modelado
Todo se ajusta con validación cruzada agrupada (StratifiedGroupKFold, 4 pliegues, solo sobre train); el test no se toca. Registro: `experiments/results/tables/experiments_log.csv`.

## Diseño
- **Dos clasificadores:** especie y calidad (características + especie en one-hot). En la app, la especie predicha alimenta a la de calidad.
  Si la especie es `other` o su confianza < 0,60, la app responde "fruta no reconocida". La calidad se entrena solo con imágenes que tienen calidad (sin Fruits-360).
- **Especie (v2.1): transfer learning** permitido por el enunciado: MobileNetV2 preentrenada en ImageNet (entrada 128x128) **congelada** como extractor (vector de 1280) + **capas propias** (Dropout 0,3, densa 256 ReLU, Dropout 0,3, softmax de 9 clases), Adam 1e-3, parada temprana en la pérdida de validación (11 épocas). `src/models/transfer.py`, `src/training/train_species_tl.py`. Se eligió por validación dejando una fuente fuera (`05_evaluacion.md`). v2.2: + racimos de WGISD, más mango de Fruits-360 y pesos por clase (13 épocas; validación 93,8 %).
- **Modelos de ML tradicional (2) + línea base (2):** mayoritaria, regresión logística, SVM (RBF), Random Forest. `class_weight` balanceado.
- **Deep learning (1):** CNN pequeña desde cero, 3 convoluciones (32/64/128) + BN + pooling global + densa, entrada 64x64, dos salidas (especie y calidad; 8 especies).
  Aumento leve (giros, rotación 5°, zoom 5 %, brillo 10 %), Adam con coseno 2e-3 → 4e-5, parada temprana en la pérdida de validación, 30 épocas, pesos por clase.
- La CNN se entrena solo con imágenes con calidad. No se usaron XGBoost ni MLP.

## Rejillas y mejores hiperparámetros (CV agrupada, F1 macro)
| Modelo | Rejilla | Mejor | F1 macro CV |
|---|---|---|---|
| Regresión logística | C ∈ {0,1; 1; 10} | C = 10 | 0,729 |
| **SVM RBF** | C ∈ {1; 10; 100}, γ ∈ {scale; 0,003; 0,03} | C = 100, γ = scale | **0,880** |
| Random Forest | árboles {200; 400}, profundidad {None; 20}, hoja mín. {1; 3} | 400, None, 3 | 0,849 |
| SVM de especie | misma rejilla | C = 10, γ = scale | 0,811 |
| RF de especie | misma rejilla | 400, 20, 3 | 0,769 |
| CNN | lr máximo {2e-3; 1e-3} (validación) | 2e-3 | F1 val 0,766 |
| Cabeza de especie | unidades {128; 256; 512} × dropout {0,3; 0,5} × lr {1e-3; 3e-4} (validación) | 256, 0,3, 1e-3 (a < 1 punto del mejor) | 93,0 % (v2.1) |

Cifras de v2 (con FruitVision; no comparables con v1, cuya validación no tenía FruitVision: SVM 0,902). La búsqueda es reanudable (`tune.py` guarda cada combinación).

## Tiempos de ajuste (CPU de 2 núcleos)
v2: logreg 13 s, SVM 43 s, Random Forest 100 s, CNN 1 281 s (30 épocas); especie con MobileNetV2: ~8 min para extraer los vectores una vez + 15 s de la cabeza. Es el dato disponible para el análisis ambiental (`impactos.md`).

## Selección
El SVM tiene el mejor F1 macro en validación (0,883) y en test (0,910). En la **validación dejando una fuente fuera** (criterio de v2.1, sin test) el Random Forest es el mejor en calidad (0,777 frente a 0,708 del SVM), y es el que la app usa por defecto porque las fotos del usuario son otra fuente; el externo lo confirma (0,765 frente a 0,657). Los tres modelos clásicos quedan en el selector de la app.
