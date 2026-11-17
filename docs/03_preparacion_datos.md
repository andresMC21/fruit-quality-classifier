# Fase 3 - Preparación de los datos
- Segmentación de frutas individuales; preprocesamiento (resize, normalización); estimación de tamaño.
- Manejo de desbalanceo (pesos de clase, aumento de datos).
- Split fijo train/val/test sin fuga de datos; test con imágenes propias.

## Hecho
- **Grupos de casi-duplicados** (`src/data/dedup.py`): `fruit_id` para Kaggle. Ver `02_comprension_datos.md`.
- **Cola de re-etiquetado** (`src/data/build_label_set.py`, `src/data/annotate_app.py`): ver `guia_anotacion.md`, sección 5.
- **Segmentación** (`src/data/segment.py`) y **tamaño** (`src/models/size_estimator.py`): método y validación en
  `formulacion_matematica.md`, secciones 3.1 y 3.4.

- **Características** (`src/data/features.py`, 54 valores) y **preprocesamiento para la CNN** (`src/data/preprocess.py`).
- **Split** (`src/data/split.py`): por `fruit_id`, estratificado por la clase más grave del grupo, propias y públicas por
  separado (test 40 % de frutas propias y 15 % de públicas; val 15 %). Se niega a sobrescribir `splits.csv` sin `--force`.
  Se ejecuta cuando `annotations.csv` tenga las etiquetas.

## Limitaciones de la segmentación (observadas en Kaggle)
| Situación | Resultado | Consecuencia |
|---|---|---|
| Fondo uniforme (blanco, gris, cian, oscuro) | Contorno ajustado a la fruta | Es el caso del protocolo de captura |
| Fondo texturado (madera, piso con vetas, césped) | Entra parte del fondo | No usar la máscara para características de forma en esas fotos |
| Mano sosteniendo la fruta | La mano entra en la máscara | Otra razón para no depender de la máscara en Kaggle (ver H4) |
| Varias frutas que se tocan (`mixed_quality`) | Quedan unidas en una componente | La separación automática se descartó: recortar a mano o no usar `mixed_quality` |

## Pendiente
- Consolidar `kaggle_labels/*.csv` y las fotos propias en `annotations.csv`, y ejecutar `split.py`.
- Validar el tamaño con fotos propias: MAE en mm del diámetro equivalente frente al de Feret, con y sin corrección de perspectiva.
