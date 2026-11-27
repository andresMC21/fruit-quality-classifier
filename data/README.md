# Datos

| Carpeta | Contenido | ¿Se versiona? |
|---|---|---|
| `raw/kaggle/` | Dataset de referencia (Kaggle, ryandpark). Solo trae calidad **buena / mala**. | No (descargar) |
| `Fruit grade/` (raíz del repo, configurable en `paths.raw_afruitdb`) | AFruitDB / Asian Fruits: manzana, banano, mango con grados 1.º/2.º/3.º. | No (descargar) |
| `raw/Sultana/` | Sultana et al.: fresca/podrida. Uva y fresa entrenan; manzana, banano, naranja y guayaba son el test externo. | No (descargar) |
| `raw/FruitVision/Fruits Original/` | FruitVision: Fresh/Rotten (Formalin-mixed no se usa) de manzana, banano, uva, mango y naranja. | No (descargar) |
| `raw/Fruits360/` | Selección fija de Fruits-360 (100x100; `_seleccion.json`), solo especie y clase `other`. | No (se reconstruye con el JSON) |
| `raw/WGISD/` | Racimos de uva recortados de WGISD con su máscara (`_recortes.json`, `_seleccion.txt`), solo especie. | No (se reconstruye con `src/data/wgisd_crops.py`) |
| `raw/mixed_quality/` | Fotos con varias frutas de calidad mixta, **sin etiqueta por fruta**: hay que segmentarlas y anotarlas a mano. | No |
| `raw/own_images/` | Imágenes propias (mín. 30-50; objetivo ≥ 100), según el protocolo de `docs/guia_anotacion.md`. | No (carpeta en la nube) |
| `raw/other_groups/` | Imágenes que compartan los demás grupos del curso. | No (carpeta en la nube) |
| `interim/segmented/` | Recortes individuales + máscaras generados por `src/data/segment.py`. | No (se regeneran) |
| `processed/splits.csv` | Asignación fija `image_id → train/val/test/external` (v2). | **Sí** |
| `processed/splits_v1.csv`, `splits_v2.csv` | Particiones congeladas de v1 y v2: sus imágenes conservan el split. | **Sí** |
| `annotations/annotations.csv` | Una fila por imagen o recorte (esquema abajo). | **Sí** |

## Compartir con el curso
Las anotaciones sin las imágenes no sirven a otros grupos. Se comparten **ambas cosas**:
- Imágenes: el grupo **no recolectó fotos propias**, así que no hay imágenes propias que compartir; las de terceros no se redistribuyen (licencias). Si el curso habilita una carpeta común, el enlace va aquí.
- Anotaciones: este CSV, con el mismo esquema acordado entre todos los grupos.

## Esquema de `annotations.csv`
| Columna | Descripción |
|---|---|
| `image_id` | Identificador único (p. ej. `G05_own_0001`). |
| `filename` | Ruta relativa a `data/raw/` o `data/interim/`. |
| `source` | `kaggle` \| `afruitdb` \| `sultana` \| `fruitvision` \| `fruits360` \| `wgisd` \| `mixed_quality` \| `own` \| `other_group`. |
| `dup_group`, `keep`, `orig_label`, `timestamp` | Grupo de duplicados, si entra al modelado (máx. 3 por grupo), etiqueta original de la carpeta y marca de tiempo del nombre del archivo. |
| `group_id` | Grupo que tomó y anotó la imagen. |
| `fruit_species` | `apple`, `banana`, ... |
| `fruit_id` | Unidad de partición: grupo de duplicados, tramo de fotos consecutivas (FruitVision) o carpeta (Fruits-360). La **misma fruta física** fotografiada varias veces comparte `fruit_id`. Así se evita la fuga entre train y test. |
| `parent_image` | Para recortes de `mixed_quality`: foto original de la que salen. |
| `quality_class` | Según los criterios escritos en la guía de anotación. |
| `size_class` | Pequeño / mediano / grande (umbrales por especie). |
| `diameter_mm` | Diámetro medido con regla o calibrador (solo frutas propias). Es la verdad de referencia del estimador de tamaño. |
| `annotator`, `annotator2_quality` | Anotador principal y segunda anotación independiente (en un subconjunto ≥ 20 %) para calcular kappa de Cohen. |
| `capture_device`, `lighting`, `background` | Metadatos de captura, útiles para el EDA y el análisis de sesgos. |
| `date`, `notes` | Fecha y observaciones. |

## Licencias
FruitNet / Kaggle se distribuye bajo **CC BY 4.0**: hay que atribuirlo en el README, el informe y `docs/referencias.md`.
