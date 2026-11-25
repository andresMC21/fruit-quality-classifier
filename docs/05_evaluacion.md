# Fase 5 - Evaluación

El conjunto de **test se evaluó una sola vez** con los modelos congelados (`experiments/results/test_done.json`). Un segundo ajuste del código fue
solo de reporte: el IC bootstrap del F1 macro del conjunto externo contaba una clase de más; se corrigió y se repitió la evaluación con los mismos modelos
(sin reentrenar ni elegir nada con ese resultado). Tablas completas en `experiments/results/tables/`.

Tres conjuntos distintos:
- **Validación** (1 139 imágenes): para elegir hiperparámetros y modelo.
- **Test público** (1 162): grupos no vistos de las mismas bases del entrenamiento (Kaggle, Fruit grade, Sultana en uva y fresa).
- **Externo** (1 493): fotos de **Sultana** de manzana, banano, naranja y guayaba, una fuente (tiendas de Bangladesh, cámara réflex) que no entró al entrenamiento. Es el sustituto de las fotos propias.

## Resultados en test público
| Modelo | F1 macro [IC 95 %] | Exactitud | Recall "mala" | Costo esperado |
|---|---|---|---|---|
| Mayoritaria | 0,235 [0,227; 0,242] | 0,544 | 0,000 | 1,584 |
| Regresión logística | 0,774 [0,741; 0,802] | 0,813 | 0,847 | 0,396 |
| Random Forest | 0,867 [0,838; 0,892] | 0,905 | 0,911 | 0,219 |
| **SVM (RBF)** | **0,889 [0,863; 0,912]** | **0,935** | **0,959** | **0,118** |

McNemar (SVM contra cada modelo, aciertos pareados): mayoritaria p < 1e-80; regresión logística p = 5e-23; Random Forest p = 4e-4 (el SVM acierta en 64 imágenes donde RF falla, y RF en 29 donde falla el SVM). La diferencia SVM-RF es estadísticamente significativa pero pequeña.

La CNN (64x64, 3 convoluciones) solo se midió en validación (F1 0,806; exactitud 0,850; recall "mala" 0,928) y no pasó al test porque el SVM la supera en validación.

Matriz de confusión del SVM en test (filas = real): buena→(589 buena, 34 regular, 9 mala); regular→(15, 78, 0); mala→(17, 1, 419). Los errores graves (mala predicha como buena: 17 de 437; buena predicha como mala: 9) son pocos; la mayor parte del error ocurre con "regular".

## SVM en test por especie y fuente
| Especie | F1 macro | Recall mala | | Fuente | F1 macro | Recall mala |
|---|---|---|---|---|---|---|
| Manzana | 0,890 | 0,943 | | Kaggle | 0,972 | 0,961 |
| Banano | 0,883 | 1,000 | | Fruit grade | 0,824 | 0,970 |
| Mango | 0,837 | 1,000 | | Sultana (uva, fresa) | 0,969 | 0,939 |
| Guayaba | 0,963 | 0,932 | | | | |
| Naranja | 0,956 | 0,910 | | | | |
| Lima | 0,977 | 0,988 | | | | |
| Uva | 0,947 | 0,893 | | | | |
| Fresa | 1,000 (39 imágenes) | 1,000 | | | | |

## Conjunto externo (otra fuente): el resultado más importante
| Modelo | F1 macro [IC 95 %] | Exactitud | Recall "mala" | Costo esperado |
|---|---|---|---|---|
| Mayoritaria | 0,322 [0,309; 0,333] | 0,474 | 0,000 | 2,103 |
| Regresión logística | 0,632 [0,608; 0,655] | 0,584 | 0,622 | 1,049 |
| SVM (RBF) | 0,600 [0,576; 0,625] | 0,595 | **0,450** | 1,348 |
| **Random Forest** | **0,764 [0,742; 0,785]** | 0,753 | 0,778 | 0,694 |

- **El SVM, el mejor en test, cae de 0,889 a 0,600** y detecta menos de la mitad de las frutas malas (en manzana 18 % y en naranja 8 %). Los dos errores van en la dirección más costosa: fruta podrida clasificada como buena.
- El Random Forest es más robusto (0,764), aunque también baja mucho frente a su test público (0,867).
- La **especie** también falla fuera de las bases de entrenamiento: el clasificador de especie acierta 54,6 % en el externo (guayaba → naranja/manzana, manzana → fresa), frente a 96,2 % en validación.
- Lectura: las bases públicas se parecen demasiado entre sí por fuente (fondo, cámara, variedad, iluminación), y el color de la piel que usan las características no se transfiere entre ellas. El F1 del test público sobreestima el desempeño con fotos nuevas. Esto era el riesgo que habíamos medido (la fuente se adivina con 98,8 %).
- Para uva y fresa **no hay medida externa** (solo existen en Sultana), así que sus métricas buenas en test (F1 0,947 y 1,000) probablemente están inflados por la fuente.

## Lectura de los resultados
- Características manuales + SVM superan a la CNN de 64x64 en validación con este volumen de datos y CPU.
- Las especies con la clase "regular" (manzana, banano, mango) son las más difíciles; Fruit grade rinde peor (0,824) que Kaggle (0,972).
- **Pendiente para el informe:** análisis de errores con ejemplos visuales, curvas de la CNN, comparación con el artículo de FruitNet y con la literatura de Sultana et al.
