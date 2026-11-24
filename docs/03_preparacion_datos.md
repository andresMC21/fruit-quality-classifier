# Fase 3 - Preparación de los datos
Comando: `python -m src.main prepare` (reanudable, `--budget` segundos por ejecución) y `python -m src.main split`.

| Paso | Código | Decisión y justificación |
|---|---|---|
| Manifiesto | `src/data/manifest.py` | IDs estables por fuente+especie; etiqueta original conservada en `orig_label`. |
| Deduplicación | `src/data/dedup.py` | pHash (Hamming ≤ 6) + ráfagas (≤ 3 s y Hamming ≤ 18 en la misma carpeta). Agrupa de más a propósito: se prefiere perder variedad en train a filtrar fuga. |
| Limitar redundancia | `main.prepare` | Máx. 3 imágenes por grupo, al azar con semilla 42. |
| Segmentación | `src/data/segment.py` | Distancia en Lab al fondo + Otsu; si falla, GrabCut con semilla central; si falla, se excluye (`ok=False`). L pesa 0,45 para reducir sombras. Control de calidad por especie (banano curvo: solidez ≥ 0,70; lima pequeña: área ≥ 1,2 %; uva en racimo ≥ 0,60; fresa ≥ 0,80). |
| Recorte | `src/data/preprocess.py` | Cuadrado de 128x128 con 8 % de margen; **fuera de la máscara = gris 128** para quitar fondo, manos y papel. La máscara va en el canal alfa. |
| Características | `src/data/features.py` | 59 variables: HSV (32), momentos Lab (6), fracciones oscuro/saturado/marrón (3), LBP (10), GLCM (4), forma (4); solo sobre la máscara. |
| Split | `src/data/split.py` | StratifiedGroupKFold por `fruit_id` (grupo de duplicados), estratificado por especie x calidad. Train 5 784, val 1 139, test 1 162 imágenes (71,5 / 14,1 / 14,4 %) y **1 493 de test externo** (Sultana: manzana, banano, naranja, guayaba; `split.external` en `config.yaml`), que nunca entran a entrenar ni a elegir modelo. Verificado: ningún grupo en dos splits (prueba en `tests/`). |

## Limitaciones conocidas
- En Fruit grade el hash no detecta la misma manzana en vistas distintas: puede haber fuga residual entre train y test de esa fuente. Las fotos propias llevan `fruit_id` explícito para evitarlo.
- 9 578 de 10 647 imágenes conservadas pasan la segmentación (90 %); las 1 069 que fallan no entran al modelado (sesgo hacia fotos "fáciles": se reporta; fresa de Sultana pierde 29 %).
- No hay fotos propias: el tamaño es relativo y el conjunto externo (Sultana) cumple su papel de prueba en otra fuente.
