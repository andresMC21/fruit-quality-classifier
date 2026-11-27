# Plan de trabajo, roles y riesgos

## Roles (según las etapas subidas al repositorio)
| Integrante | Etapas |
|---|---|
| Camilo A. Martinez Moreno | 1. Configuración, fuentes de datos, comprensión del negocio y de los datos · 3. Anotaciones, split y EDA |
| Santiago Carlosama Ortiz | 2. Preparación de datos: manifiesto, deduplicación, segmentación, características y split · 5. Evaluación: métricas, test único, test externo y tablas |
| Javier Andrés Paz Luna | 4. Modelado: modelos clásicos, CNN y ajuste de hiperparámetros · 6. Inferencia, app Streamlit, modelos exportados y pruebas |
| Paula Andrea Piedrahita | 7. Ética, impactos, formulación matemática y referencias · 8. Informe y diagramas CRISP-DM |

## Iteraciones (CRISP-DM)
| Versión | Qué se hizo | Por qué |
|---|---|---|
| v1 (17-18 nov.) | Kaggle + Fruit grade + Sultana (externo); deduplicación; SVM, RF, logística, CNN | Línea base y primer test |
| v1 con semilla (26 nov.) | GrabCut con semilla fija y todo regenerado | 236 máscaras cambiaban al repetir |
| v2 (26 nov.) | + FruitVision (calidad y especie) y Fruits-360 (especie, `other`) | Mango leído como naranja/manzana: especie de una sola fuente |
| v2.1 (26 nov.) | Especie con MobileNetV2 congelada + cabeza; RF por defecto | Validación dejando una fuente fuera |
| v2.2 (26 nov.) | + WGISD (racimos), más mango, pesos por clase | Racimos de uva y mangos rojos mal reconocidos |

## Riesgos y cómo se trataron
| Riesgo | Medida |
|---|---|
| Fuga entre train y test (ráfagas, vistas giradas) | Grupos de pHash y ráfagas, tramos de secuencia, unidad = foto o carpeta; verificación automática |
| Atajo de la fuente | Fondo neutro, métricas por fuente, test externo, validación dejando una fuente fuera |
| Sobreajustar al test | Un candado por versión; decisiones con validación; las post hoc se declaran |
| Sin fotos propias | Cuatro fuentes con calidad + test externo; limitación declarada |
| Cómputo limitado (CPU de 2 núcleos) | Pasos reanudables, extractor congelado, CNN pequeña |
| Variedades ausentes (mango rojo) | Declarado en la app y en el informe; aumento probado y descartado |
