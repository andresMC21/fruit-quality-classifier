# Clasificación automática de calidad de frutas (Algoritmos y Programación III, 2026-2)

Sistema de visión por computadora que, a partir de una foto de **una fruta individual sobre fondo uniforme**,
predice (1) su **clase de calidad** (≥ 3 categorías ordinales) y (2) su **tamaño** (pequeño/mediano/grande y diámetro estimado en mm).
Especies: manzana, banano, mango, guayaba, naranja y lima. Metodología: **CRISP-DM** adaptada. Despliegue: interfaz **Streamlit** con carga de imagen y cámara.

Plan de trabajo, roles y riesgos: [`docs/plan_trabajo.md`](docs/plan_trabajo.md).

## Integrantes
| Nombre | Rol principal (semanas 1-2 → 3-4) |
|---|---|
| Camilo A. Martinez Moreno | Datos y anotación → ética, impactos, edición del informe |
| Santiago Carlosama Ortiz | Segmentación, tamaño y línea base → ML clásico, evaluación, formulación matemática |
| Javier Andrés Paz Luna | Pipeline de inferencia y esqueleto de la app → CNN, integración, video |
| Paula Andrea Piedrahita | _por definir_ |

## Mapa CRISP-DM → repositorio
| Fase | Documentación | Código | Notebook | Resultados |
|---|---|---|---|---|
| 1. Comprensión del negocio | `docs/01_comprension_negocio.md` | - | `notebooks/01_negocio.ipynb` | - |
| 2. Comprensión de los datos | `docs/02_comprension_datos.md`, `docs/guia_anotacion.md` | `src/data/dedup.py` | `notebooks/02_eda.ipynb` | `experiments/results/eda/` |
| 3. Preparación de los datos | `docs/03_preparacion_datos.md` | `src/data/` | `notebooks/03_preparacion.ipynb` | `data/processed/splits.csv` |
| 4. Modelado | `docs/04_modelado.md` | `src/models/`, `src/training/` | `notebooks/04_modelado.ipynb` | `experiments/results/tables/experiments_log.csv` |
| 5. Evaluación | `docs/05_evaluacion.md` | `src/evaluation/` | `notebooks/05_evaluacion.ipynb` | `experiments/results/` |
| 6. Despliegue | `docs/06_despliegue.md` | `src/inference/`, `src/app/` | - | `models/` |

Diagrama CRISP-DM personalizado: `docs/diagramas/`.

## Competencias evaluadas
- **PI1 Ética:** `docs/etica.md`
- **PI2 Impactos (social, económico, ambiental, global):** `docs/impactos.md`
- **PI3 Matemática:** `docs/formulacion_matematica.md`

## Ejecución rápida
```bash
pip install -r requirements.txt          # o: conda env create -f environment.yml
python -m src.main prepare --budget 150  # manifiesto + deduplicación + segmentación (reanudable: repetir hasta que no diga PENDIENTE)
python -m src.main split                 # data/processed/splits.csv (agrupado, estratificado, se versiona)
python -m src.main train --model svm     # majority | logreg | svm | rf | cnn (calidad y especie; registra en experiments_log.csv)
python -m src.main export --model svm    # copia los modelos a models/ con meta.json
python -m src.data.eda                   # figuras y tablas del EDA
pytest -q                                # pruebas unitarias
streamlit run src/app/streamlit_app.py
# Evaluación en test: UNA sola vez, ver src/evaluation/evaluate.py (deja candado en experiments/results/test_done.json)
```
Más detalle en `docs/instalacion.md`.

## Datos y fuentes de terceros
- Fruit grade (AFruitDB / Asian Fruits Dataset): carpeta `Fruit grade/` (no versionada); referencia completa pendiente en `docs/referencias.md`.
- Fruit Quality Classification (Kaggle, ryandpark): https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
- FruitNet (K. Patil, V. Meshram), Mendeley Data, doi: 10.17632/b6fftwbr2v.1, licencia CC BY 4.0.
- Sultana et al., *Fresh and rotten fruits* (Mendeley Data, doi 10.17632/bdd69gyhv8.1): carpeta `Sultana/` (no versionada); uva y fresa para entrenar; manzana, banano, naranja y guayaba como test externo.
- Evaluadas y descartadas: FruQ-DB/FruQ-multi (Zenodo) y FruityHub (motivos en `docs/02_comprension_datos.md`).
- No hay fotos propias: ver `data/README.md`.
- Todo código o dato de terceros se cita en `docs/referencias.md` (omitirlo se considera fraude).
