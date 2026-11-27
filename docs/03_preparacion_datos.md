# Fase 3 - Preparación de los datos
Comando: `python -m src.main prepare` (reanudable, `--budget` segundos por ejecución) y `python -m src.main split`.

| Paso | Código | Decisión y justificación |
|---|---|---|
| Manifiesto | `src/data/manifest.py` | IDs estables por fuente+especie; etiqueta original conservada en `orig_label`. Las fuentes de v2 (FruitVision, Fruits-360) van al final: no cambian IDs ni grupos de v1. |
| Deduplicación | `src/data/dedup.py` | pHash (Hamming ≤ 6) + ráfagas (≤ 3 s y Hamming ≤ 18 en la misma carpeta). Agrupa de más a propósito: se prefiere perder variedad en train a filtrar fuga. |
| Unidad de partición | `dedup.partition_units` | Grupo de duplicados (v1); **tramo de ~30 fotos consecutivas** con corte en el mayor salto de pHash (±5) y unión por archivos idénticos (FruitVision); carpeta/variedad (Fruits-360); foto de viñedo de origen (WGISD). |
| Duplicados entre fuentes | `dedup.cross_source_dups` | Se excluyen 83 fotos de FruitVision con pHash ≤ 6 respecto a otra fuente de la misma especie (posible redistribución; protege el test externo). |
| Limitar redundancia | `main.prepare` | Máx. 3 imágenes por grupo, al azar con semilla 42, sorteado por bloques de fuentes (`dedup.sample_blocks`) para que v1 se reproduzca idéntica (verificado: mismas 15 309 filas, grupos y selección). FruitVision y Fruits-360 sin tope (`uncapped_sources`). |
| Segmentación | `src/data/segment.py` | Distancia en Lab al fondo + Otsu; si falla, GrabCut con semilla central; si falla, se excluye (`ok=False`). L pesa 0,45 para reducir sombras. Control de calidad por especie (banano curvo: solidez ≥ 0,70; lima pequeña: área ≥ 1,2 %; uva en racimo ≥ 0,60; fresa ≥ 0,80). |
| Recorte | `src/data/preprocess.py` | Cuadrado de 128x128 con 8 % de margen; **fuera de la máscara = gris 128** para quitar fondo, manos y papel. La máscara va en el canal alfa. |
| Características | `src/data/features.py` | 59 variables: HSV (32), momentos Lab (6), fracciones oscuro/saturado/marrón (3), LBP (10), GLCM (4), forma (4); solo sobre la máscara. |
| Split | `src/data/split.py` | StratifiedGroupKFold por `fruit_id`, estratificado por especie x calidad. **v1 congelada** (`splits_v1.csv`): sus imágenes conservan el split. Imágenes nuevas: FruitVision 15 % test y 15 % val por tramos; Fruits-360 de las 8 especies solo train; `other` repartida por carpeta. **v1 y v2 congeladas** (`splits_v2.csv`); en v2.2 solo se reparten WGISD (por foto) y los fotogramas nuevos de mango (train). Total v2.2: train 13 234, val 2 363, test 2 363 y **1 495 de test externo** (Sultana: manzana, banano, naranja, guayaba), que nunca entran a entrenar ni a elegir modelo. Con etiqueta de calidad: 10 596 / 2 153 / 2 195. Verificado: ningún grupo en dos splits. |

## Limitaciones conocidas
- En Fruit grade el hash no detecta la misma manzana en vistas distintas: puede haber fuga residual entre train y test de esa fuente. Las fotos propias llevan `fruit_id` explícito para evitarlo.
- **Reproducibilidad:** GrabCut usa un k-means aleatorio; se fija con `GRABCUT_SEED = 42` (`segment.py`). Una primera corrida sin semilla daba máscaras distintas al repetirla (236 imágenes cambiaban de resultado) y se descartó: todo lo reportado sale de la corrida con semilla.
- v2.2: 19 455 de 20 573 imágenes conservadas pasan la segmentación (95 %; WGISD 99,7 %; v2: 18 669 de 19 786; v1: 9 556 de 10 647); las 1 117 que fallan no entran al modelado (sesgo hacia fotos "fáciles": se reporta; fresa de Sultana pierde 29 %).
- No hay fotos propias: el tamaño es relativo y el conjunto externo (Sultana) cumple su papel de prueba en otra fuente.
- FruitVision: los tramos evitan que vistas de la misma fruta queden en dos splits; en un corte de tramo puede quedar, como máximo, una fruta repartida entre tramos vecinos (fuga residual pequeña). Los tramos grandes desbalancean un poco algunos conjuntos (p. ej. uva mala: 275 en val).
- Fruits-360 son fotos de 100x100 de una fruta girando: el tope por grupo no aplica, pero la carpeta completa queda en un solo split.
