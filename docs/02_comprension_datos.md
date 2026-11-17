# Fase 2 - Comprensión de los datos (EDA)
- Fuentes: Kaggle, mixed_quality, imágenes propias.
- Distribución de clases y desbalanceo; resolución; iluminación; fondos; variabilidad.
- Calidad de etiquetas y coherencia entre anotadores.
- Figuras en vectorial (`experiments/results/eda/`).

Estado: inventario y hallazgos iniciales de Kaggle (2026-11-17). Pendiente: figuras, fotos propias y kappa.

## Obtención de Kaggle
- Kaggle exige cuenta para descargar. Se usó el origen que Kaggle cita, **FruitNet** (Mendeley Data, DOI 10.17632/b6fftwbr2v.3, CC BY 4.0), que se descarga sin cuenta: `FruitNetDataset.zip`, 3.253.463.240 bytes, CRC verificado.
- El zip tiene **19.526 imágenes**, la misma cifra que declara la ficha de Kaggle, con los mismos nombres de carpeta (`Bad Quality_Fruits/Apple_Bad/...`). No se compararon archivo por archivo.
- Inventario: `python -m src.data.fruitnet_inventory data/raw/kaggle/FruitNetDataset.zip --extract-apple` → `data/annotations/kaggle_inventory.csv`.

## Manzana
| Carpeta | Imágenes | Grupos de casi-duplicados* | Grupos con >1 imagen | Grupo más grande |
|---|---|---|---|---|
| `Apple_Good` | 1.149 | 1.056 | 44 | 28 |
| `Apple_Bad` | 1.141 | **378** | 178 | **36** |
| Mixed `Apple` | 113 | 88 | 17 | 4 |

\* pHash, distancia de Hamming ≤ 6, unión transitiva (`src/data/dedup.py`). Ningún grupo mezcla carpetas. Los 3 grupos más grandes de `Apple_Bad` se revisaron a ojo: son ráfagas de la misma fruta en la misma pose (`IMG_20190910_1727xx`).

Resolución: el lado mayor siempre mide 256 px (256×256, 192×256 o 256×192).

## Hallazgos
| # | Hallazgo | Evidencia | Consecuencia |
|---|---|---|---|
| H1 | `Apple_Bad` tiene ~378 frutas/poses distintas, no 1.141 | Tabla | Split por `fruit_id` (grupo); con split por imagen las métricas se inflan |
| H2 | La carpeta no siempre es confiable | En una muestra al azar de 8 imágenes de `Apple_Good` apareció una foto de bananos | Re-etiquetado con la guía (`guia_anotacion.md`, sección 5) |
| H3 | Varias frutas por foto también en buena y mala | Revisión visual | Excluir o segmentar; la entrada del sistema es una fruta |
| H4 | Posible atajo: en mala hay muchas manos y fondos claros; en buena, fondos variados (piso, periódico, madera) | Revisión visual de muestras pequeñas, **sin cuantificar** | Se anotan `hand` y `background` para medirlo |
| H5 | Hay manzanas verdes y amarillas en buena | Revisión visual | El color no puede ser criterio de calidad |
| H6 | Baja resolución y sin escala | Tamaños de imagen | Defectos pequeños pueden no verse; el tamaño físico solo se evalúa con fotos propias |
