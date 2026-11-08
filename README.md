# Clasificación automática de calidad de frutas (Algoritmos y Programación III, 2026-2)

Sistema de visión por computadora que, a partir de una foto de **una fruta/verdura individual sobre fondo uniforme**,
predice (1) su **clase de calidad** (≥3 categorías) y (2) su **tamaño** (pequeño/mediano/grande o diámetro normalizado).
Metodología: **CRISP-DM** adaptada. Despliegue: interfaz **Streamlit** con carga de imagen y cámara.

## Integrantes
| Nombre | Rol principal |
|---|---|
| _Integrante A_ | Datos y anotación (recolección, segmentación, EDA) |
| _Integrante B_ | Modelos de ML clásico y evaluación |
| _Integrante C_ | CNN, despliegue y video |

## Mapa CRISP-DM -> repositorio
| Fase | Documentación | Código | Notebook | Resultados |
|---|---|---|---|---|
| 1. Comprensión del negocio | `docs/01_comprension_negocio.md` | - | `notebooks/01_negocio.ipynb` | - |
| 2. Comprensión de los datos | `docs/02_comprension_datos.md` | `src/data/` | `notebooks/02_eda.ipynb` | `experiments/results/eda/` |
| 3. Preparación de los datos | `docs/03_preparacion_datos.md` | `src/data/` | `notebooks/03_preparacion.ipynb` | `data/processed/` |
| 4. Modelado | `docs/04_modelado.md` | `src/models/`, `src/training/` | `notebooks/04_modelado.ipynb` | `experiments/` |
| 5. Evaluación | `docs/05_evaluacion.md` | `src/evaluation/` | `notebooks/05_evaluacion.ipynb` | `experiments/results/` |
| 6. Despliegue | `docs/06_despliegue.md` | `src/app/` | - | - |

## Competencias evaluadas (ver `docs/`)
- **PI1 Ética:** `docs/etica.md`
- **PI2 Impactos (social, económico, ambiental, global):** `docs/impactos.md`
- **PI3 Matemática:** `docs/formulacion_matematica.md`

## Ejecución rápida
```bash
pip install -r requirements.txt          # o: conda env create -f environment.yml
python -m src.main prepare               # segmentación + preprocesamiento + split
python -m src.main train --model svm     # svm | rf | xgboost | cnn | mlp
python -m src.main evaluate
streamlit run src/app/streamlit_app.py
```
Más detalle en `docs/instalacion.md`.

## Datos y fuentes de terceros
- Fruit Quality Classification (Kaggle, ryandpark): https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
- Imágenes propias (>= 30-50) y anotaciones compartidas entre grupos: ver `docs/guia_anotacion.md`.
- Todo código o dato de terceros se cita en `docs/referencias.md` (omitirlo se considera fraude).
