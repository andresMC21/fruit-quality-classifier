# Clasificación automática de calidad de frutas (Algoritmos y Programación III, 2026-2)

Sistema de visión por computadora que, a partir de una foto de **una fruta individual sobre fondo uniforme**,
predice (1) su **clase de calidad** (≥ 3 categorías ordinales) y (2) su **tamaño** (pequeño/mediano/grande y diámetro estimado en mm).
Especies: manzana, banano, mango, guayaba, naranja, lima, uva (grano o racimo) y fresa (otras frutas: "no reconocida"). Metodología: **CRISP-DM** adaptada. Despliegue: interfaz **Streamlit** con carga de imagen y cámara.

Informe: `report/Informe_IEEE_frutas.pdf`. Video (10 min): https://youtu.be/avQUO4oVqyg. Repositorio: https://github.com/andresMC21/fruit-quality-classifier.git

Plan de trabajo, roles y riesgos: [`docs/plan_trabajo.md`](docs/plan_trabajo.md). Arquitectura: [`docs/arquitectura.md`](docs/arquitectura.md). Los notebooks de `notebooks/` están ejecutados y muestran las tablas y figuras de cada fase.

## Integrantes
| Nombre | GitHub | Aporte en el repositorio (etapas subidas) |
|---|---|---|
| Camilo A. Martinez Moreno | [andresMC21](https://github.com/andresMC21) | Configuración, fuentes de datos y comprensión del negocio y de los datos → anotaciones, split y análisis exploratorio |
| Santiago Carlosama Ortiz | [CarlosamaSantiago](https://github.com/CarlosamaSantiago) | Preparación de datos (manifiesto, deduplicación, segmentación, características, split) → evaluación (test único, test externo, tablas) |
| Javier Andrés Paz Luna | [Javi-19](https://github.com/Javi-19) | Modelado (modelos clásicos, CNN, hiperparámetros) → inferencia, app Streamlit, modelos exportados y pruebas |
| Paula Andrea Piedrahita | [Pauli55](https://github.com/Pauli55) | Ética, impactos, formulación matemática y referencias → informe y diagramas CRISP-DM |

Roles según las etapas que cada integrante subió al repositorio (historial de commits).

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
pip install -r requirements-lock.txt     # versiones probadas (o requirements.txt; o conda env create -f environment.yml)
python -m src.main prepare --budget 150  # manifiesto + deduplicación + segmentación (reanudable: repetir hasta que no diga PENDIENTE)
python -m src.main split                 # data/processed/splits.csv (agrupado, estratificado, se versiona)
python -m src.main train --model svm     # majority | logreg | svm | rf | cnn (calidad y especie; registra en experiments_log.csv)
python -m src.training.train_species_tl  # especie: MobileNetV2 congelada + cabeza propia (models/species_tl.keras)
python -m src.training.tune_species_tl   # rejilla de la cabeza de especie (validación)
python -m src.training.run_cnn 30 125    # CNN reanudable; python -m src.training.tune_cnn resume la búsqueda
python -m src.evaluation.overfitting     # brecha train / CV / validación
python -m src.evaluation.cross_source especie emb_head   # validación dejando una fuente fuera (criterio de selección)
python -m src.main export --model rf --species tl   # modelos a models/ con meta.json (RF por defecto)
python -m src.evaluation.error_analysis  # figuras del informe (report/figures/)
python -m src.data.eda                   # figuras y tablas del EDA
pytest -q                                # pruebas unitarias
streamlit run src/app/streamlit_app.py
# Evaluación en test: UNA sola vez, ver src/evaluation/evaluate.py (candados en experiments/results/test_done_v2*.json; v1 en experiments/results/v1/)
```
Más detalle en `docs/instalacion.md`.

## Resultados (v2.2)
| | Test (2 195) | Externo, otra fuente (1 495) |
|---|---|---|
| Calidad, Random Forest (por defecto) | F1 macro 0,891 | **0,765**, recall de "mala" 0,75 |
| Calidad, SVM | **0,910**, recall de "mala" 0,96 | 0,657 |
| Especie, MobileNetV2 congelada + cabeza | 96,8 % (racimos de uva: 97,9 %) | 71,2 % (v1: 60,5 %) |

Limitaciones por fruta en la sección siguiente.

Detalle, versiones y candados en `docs/05_evaluacion.md`.

## Alcance y limitaciones por fruta
Exactitud de **especie** de la app (v2.2: acierto y no rechazada), por fuente, en test y en el test externo (`experiments/results/tables/especie_por_fuente_v22.csv`). **No hay fotos propias de Colombia**: ninguna fruta está validada con fotos locales.

Origen de las fuentes: Kaggle/FruitNet (India), Fruit grade, FruitVision y Sultana (Bangladesh), WGISD (viñedos de Brasil), Fruits-360 (fotos de estudio, solo entrenamiento).

| Fruta | Se reconoce bien | Falla o no está validado |
|---|---|---|
| Manzana | Manzanas rojas y verdes de India y Bangladesh: 88-97 % en test y **87 % en otra fuente** (Sultana) | Absorbe frutas rojas que no conoce: los **mangos rojos** se leen como manzana |
| Banano | Banano tipo Cavendish de India y Bangladesh: 97-100 % en test y **96 % en otra fuente** | Plátano hartón, banano bocadillo y plátano maduro negro: no validados |
| Mango | **Mango asiático verde o amarillo** (Fruit grade, FruitVision): 98-100 % en test | **Mango rojo-amarillo tipo Tommy, el común en Colombia: no se reconoce** (sale manzana); sin prueba en otra fuente |
| Guayaba | Guayaba verde de India (Kaggle): 94 % en test | **En otra fuente (Sultana) solo 14 %**: no es fiable fuera de Kaggle; guayaba colombiana no validada |
| Naranja | Naranja anaranjada (FruitVision) 99,7 % y cítrico verde-amarillo de India (Kaggle) 93 % en test; **73 % en otra fuente** | La "naranja" de Kaggle es verde-amarilla: puede confundirse con manzana o lima; mandarina, toronja y clementina están fuera del alcance |
| Lima | Lima verde de India (Kaggle): 91 % en test | Una sola fuente, sin prueba en otra; el limón amarillo se trata como "otra fruta" |
| Uva | **Grano suelto** (FruitVision, Sultana) 98-99 % y **racimos de uva de vino** (WGISD) 96 % en test | Racimos de uva de mesa en otro contexto (plato, hojas): 2 de 4 fotos de internet; sin prueba en otra fuente |
| Fresa | Fresa de Bangladesh (Sultana): 100 % en test | Una sola fuente con calidad: el 100 % no prueba generalización |

Otras limitaciones:
- **Calidad:** "regular" solo existe en manzana, banano y mango (Fruit grade). Fuera de dominio el modelo por defecto (RF) detecta 75 % de la fruta mala; la naranja (48 %) y la manzana (61 %) malas son las más difíciles.
- **Tamaño:** solo relativo al encuadre (pequeño, mediano, grande); sin objeto de referencia no hay milímetros ni error medido.
- **Frutas fuera de la lista:** se rechazan como "no reconocida" el 72 % de las veces, así que no es un filtro seguro.
- **Uso:** una fruta (o un racimo) por foto, sobre fondo liso y con buena luz. Es un prototipo de apoyo, no sustituye la inspección humana.

## Datos y fuentes de terceros
- Fruit grade (AFruitDB / Asian Fruits Dataset): carpeta `Fruit grade/` (no versionada); referencia en `docs/referencias.md`.
- Fruit Quality Classification (Kaggle, ryandpark): https://www.kaggle.com/datasets/ryandpark/fruit-quality-classification
- FruitNet (K. Patil, V. Meshram), Mendeley Data, doi: 10.17632/b6fftwbr2v.1, licencia CC BY 4.0.
- Sultana et al., *Fresh and rotten fruits* (Mendeley Data, doi 10.17632/bdd69gyhv8.1): carpeta `Sultana/` (no versionada); uva y fresa para entrenar; manzana, banano, naranja y guayaba como test externo.
- FruitVision (Bijoy et al., Mendeley Data, doi 10.17632/xkbjx8959c.2; Fresh/Rotten de manzana, banano, uva, mango, naranja): `data/raw/FruitVision/` (no versionada); entrenamiento, validación y test desde v2.
- Fruits-360 (H. Mureșan y M. Oltean; repositorio `fruits-360/fruits-360-100x100` en GitHub): selección fija en `data/raw/Fruits360/_seleccion.json`; solo para la especie y la clase `other`.
- WGISD (Santos et al., Embrapa; github.com/thsant/wgisd): racimos de uva recortados con su máscara (`src/data/wgisd_crops.py`, `data/raw/WGISD/`, no versionada); solo especie.
- MobileNetV2 preentrenada en ImageNet (pesos de Keras, `models/pretrained/`): extractor congelado del clasificador de especie.
- Evaluadas y descartadas: FruitQuality (copia de la base de Kaggle), FruQ-DB/FruQ-multi (Zenodo) y FruityHub (motivos en `docs/02_comprension_datos.md`).
- No hay fotos propias: ver `data/README.md`.
- Todo código o dato de terceros se cita en `docs/referencias.md` (omitirlo se considera fraude).
