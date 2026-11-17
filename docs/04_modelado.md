# Fase 4 - Modelado
- Línea base. >=2 modelos de ML clásico + 1 de deep learning.
- Espacio de hiperparámetros, validación cruzada, tabla de experimentos (`experiments/results/tables/experiments_log.csv`).
- Selección del modelo y justificación.

## Decisión: qué entrada recibe cada modelo (2026-11-17)
**Problema.** En Kaggle la segmentación parece correcta en pocas fotos: en una estimación preliminar, con circularidad ≥ 0,7
como indicador, fue el 14 % de las 778 imágenes de la cola. Falla sobre todo cuando hay una mano o el fondo es texturado.
En esas fotos las características de forma y color mezclan fruta con mano o fondo, y la mano es mucho más frecuente en
"mala" (hallazgo H4 en `02_comprension_datos.md`). Un modelo podría aprender "mano → mala" en lugar de la calidad.

**Decisión.**
- **ML clásico (SVM, Random Forest):** se entrena solo con imágenes cuya segmentación pasa un control de calidad. Hay menos
  datos, pero las características describen la fruta.
- **CNN:** recibe la imagen completa, porque puede aprender a ignorar el fondo si los datos lo permiten.
- **Comparación decisiva:** sobre las fotos propias de test, que siguen el protocolo de fondo uniforme y no tienen manos.
  Si la CNN rinde bien en Kaggle y mal en las fotos propias, es señal de que aprendió el atajo.

- **Experimento de contraste:** los mismos modelos clásicos con color y textura de la **imagen completa**
  (`--features full`, sin forma, porque sería la del rectángulo de la foto). Si en Kaggle rinde más que la versión
  segmentada pero menos en las fotos propias, la diferencia mide cuánto se apoya en el fondo y la mano.

**Consecuencia medida en la prueba de funcionamiento** (con pre-etiquetas, solo para dimensionar; no son resultados):
con el control de segmentación (circularidad ≥ 0,7, revisado a ojo en 20 fotos) quedan unas 63 imágenes de Kaggle para
entrenar los clásicos, frente a 472 con la imagen completa. Por eso las **fotos propias**, tomadas con el protocolo de fondo
uniforme, son la fuente principal de datos limpios: la meta sube a ~100 frutas, y se piden las fotos de otros grupos.

**Pendiente de confirmar** cuando estén las etiquetas verificadas: el criterio exacto del control de segmentación y las cifras
por fondo y mano.

## Cómo se ejecuta
```powershell
python -m src.main prepare                            # etiquetas -> annotations.csv, splits.csv, features.csv
python -m src.main train --model baseline             # y logreg, svm, rf (fruta segmentada)
python -m src.main train --model svm --features full  # contraste con imagen completa
python -m src.main train --model cnn                  # 8 combinaciones de cnn.grid con early stopping
```
Cada experimento queda en `experiments/results/tables/experiments_log.csv`; las curvas de la CNN, en `experiments/logs/`.
La val F1 de la CNN es optimista, porque val elige el epoch y los hiperparámetros. La medida limpia es test, una sola vez.
