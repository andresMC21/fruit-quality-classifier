# Fase 1 - Comprensión del negocio

## Problema
En mercados y agroindustria la clasificación de fruta fresca por calidad y tamaño es manual: lenta, subjetiva y propensa a error.
Clasificar mal tiene dos costos distintos: **descartar fruta buena** (desperdicio y pérdida para el productor) y **vender fruta en mal estado**
(riesgo para el consumidor y reclamos). Un sistema de visión con un celular o una webcam estandariza el criterio y apoya la decisión humana.

## Alcance acordado
- **Especies (8):** manzana, banano, mango, guayaba, naranja, lima, uva y fresa. Son las de la lista de frutas comunes del curso que tienen datos públicos
  utilizables. Quedan **fuera de alcance** pera, piña, sandía, mandarina, mora y durazno, y el limón propiamente dicho (las bases traen
  *lima*, que es otra fruta); las bases FruQ y FruityHub se evaluaron y descartaron (ver `02_comprension_datos.md`). La app avisa "fruta no reconocida" en vez de inventar una calidad.
- **Salidas:** especie, calidad (buena / regular / mala) y tamaño (pequeño / mediano / grande; diámetro en mm si hay objeto de referencia).
- **Entrada:** foto de una fruta individual sobre fondo liso.
- **Despliegue:** app Streamlit (cámara o archivo) con selector de modelo.

## Objetivos y criterios de éxito
| Objetivo de negocio | Objetivo analítico | Métrica |
|---|---|---|
| No vender fruta mala como buena | Alto recall de "mala" | recall(mala) |
| Clasificación equilibrada entre clases | Buen desempeño en las 3 clases | **F1 macro** (principal) |
| Penalizar más los errores graves | Minimizar el costo de error | costo esperado con la matriz de `configs/config.yaml` |
| Funcionar con fotos de celular, no solo con las bases | Generalizar a fotos propias | métricas por fuente (kaggle / afruitdb / propias) |
| Tamaño útil | Estimar el diámetro | MAE en mm (solo fotos propias con referencia) |

Línea base obligatoria: clase mayoritaria y regresión logística. Un modelo "sirve" si supera claramente a la regresión logística con intervalo de confianza que no se solape.

## Decisiones de diseño tomadas
- La clase **regular** no existe en Kaggle (solo buena/mala); sale de los grados de Fruit grade (1.º = buena, 2.º = regular, 3.º = mala) y de las fotos propias.
- Kaggle: *Good* = buena, *Bad* = mala. Sultana: *Fresh* = buena, *Rotten* = mala.

## Riesgos principales
Atajo de fuente (la clase regular solo viene de una base), fuga por fotos casi idénticas, tamaño sin escala en las bases públicas, poca fruta propia.
Mitigaciones y evidencia en `02`, `03` y `05`.

Antecedentes y literatura: `referencias.md`. Cronograma y roles: `plan_trabajo.md`.
