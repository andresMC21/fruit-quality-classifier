# Fase 5 - Evaluación

## Versiones y candados
| Versión | Qué cambia | Test evaluado | Resultados |
|---|---|---|---|
| v1 | Kaggle + Fruit grade (+ uva y fresa de Sultana); GrabCut con semilla | una vez (`experiments/results/v1/test_done.json`) | `experiments/results/v1/` |
| **v2** | + **FruitVision** (calidad y especie) y **Fruits-360** (solo especie, con la clase `other`) | una vez (`test_done_v2.json`) | `experiments/results/tables/` |
| v2.1 | Solo el clasificador de **especie**: MobileNetV2 congelada + cabeza propia | una vez (`test_done_v2_1.json`) | tablas con sufijo `_v21` |
| **v2.2** | Especie: + **WGISD** (racimos de uva), más mango de Fruits-360 y pesos por clase | una vez (`test_done_v22.json`) | tablas con sufijo `_v22` |

Las imágenes de v1 y v2 **conservan su partición** (`data/processed/splits_v1.csv`, `splits_v2.csv`); el test de v2 = test de v1 + tramos de FruitVision no vistos. Los modelos e hiperparámetros se eligieron solo con validación cruzada agrupada y validación. La v2.1 se motivó por el resultado de especie de v2 en el externo, pero el candidato se **eligió con validación dejando una fuente fuera** (sin test), y se declara así.

Conjuntos (imágenes con calidad):
- **Validación** (2 153): elegir hiperparámetros y modelo.
- **Test** (2 195): grupos/tramos no vistos de Kaggle, Fruit grade, FruitVision y Sultana (uva y fresa). Para la especie se suman 120 imágenes de `other` (carpetas de Fruits-360 no vistas).
- **Externo** (1 495): Sultana (manzana, banano, naranja, guayaba), fuente que nunca entra al entrenamiento.

## Validación (v2)
| Modelo | F1 macro | Recall "mala" | Costo esperado |
|---|---|---|---|
| Mayoritaria | 0,224 | 0,000 | 1,847 |
| Regresión logística | 0,733 | 0,837 | 0,450 |
| CNN (64x64, 3 conv.) | 0,766 | 0,937 | 0,323 |
| Random Forest | 0,867 | 0,917 | 0,220 |
| **SVM (RBF)** | **0,883** | 0,941 | 0,156 |

Hiperparámetros (CV agrupada de 4 pliegues): regresión logística C = 10; SVM C = 100, γ = scale; Random Forest 400 árboles, hoja mínima 3, sin profundidad máxima. Especie: SVM C = 10, γ = scale (exactitud en validación 91,2 %); MobileNetV2 + cabeza 93,0 %.

## Validación dejando una fuente fuera (criterio de v2.1)
`src/evaluation/cross_source.py`, tabla `validacion_entre_fuentes.csv`. Se entrena sin una fuente (train+val) y se mide en ella; las clases que solo existen en la fuente retirada no se cuentan ("regular" al retirar Fruit grade).

| Tarea | Candidato | Kaggle | Fruit grade | FruitVision | **Media** |
|---|---|---|---|---|---|
| Especie (exactitud) | SVM, 59 características | 0,180 | 0,585 | 0,542 | 0,436 |
| | Random Forest | 0,192 | 0,666 | 0,675 | 0,511 |
| | MobileNetV2 congelada + reg. logística | 0,354 | 0,845 | 0,800 | 0,666 |
| | **MobileNetV2 congelada + cabeza propia** | 0,366 | 0,862 | 0,849 | **0,692** |
| Calidad (F1 macro) | SVM | 0,656 | 0,731 | 0,736 | 0,708 |
| | MobileNetV2 congelada + cabeza | 0,621 | 0,877 | 0,762 | 0,753 |
| | **Random Forest** | 0,719 | 0,809 | 0,802 | **0,777** |

- Con características manuales la especie **no se transfiere entre cámaras**. Al retirar Kaggle, sus naranjas se predicen como manzana: las "naranjas" de FruitNet/Kaggle (India) son cítricos verde-amarillos, distintas de las de FruitVision y Sultana. Parte del error es de los datos, no del modelo.
- Decisión v2.1: especie = **MobileNetV2 congelada + cabeza propia** (transfer learning permitido por el enunciado: capas congeladas y capas propias). Calidad por defecto = **Random Forest**, el más robusto entre fuentes; el SVM sigue en el selector de la app.

## Calidad en test (v2)
| Modelo | F1 macro [IC 95 %] | Exactitud | Recall "mala" | Costo esperado |
|---|---|---|---|---|
| Mayoritaria | 0,233 [0,227; 0,239] | 0,538 | 0,000 | 1,719 |
| Regresión logística | 0,748 [0,722; 0,772] | 0,824 | 0,874 | 0,380 |
| Random Forest | 0,891 [0,867; 0,912] | 0,935 | 0,941 | 0,159 |
| **SVM (RBF)** | **0,910 [0,885; 0,930]** | **0,955** | **0,964** | **0,100** |

McNemar (SVM contra cada modelo): mayoritaria p ≈ 7e-181; logística p ≈ 1e-51; Random Forest p = 8,3e-5 (85 imágenes solo las acierta el SVM, 40 solo el RF). La CNN no pasó al test (peor en validación).

Matrices de confusión en test (filas = real; buena, regular, mala):
- SVM: buena (1 130, 22, 29); regular (13, 80, 1); mala (30, 3, 887).
- RF (por defecto en la app): buena (1 109, 19, 53); regular (12, 77, 5); mala (49, 5, 866).

| Por fuente (F1) | n | SVM | RF | | Por especie (F1) | n | SVM | RF |
|---|---|---|---|---|---|---|---|---|
| Kaggle | 831 | 0,968 | 0,927 | | Manzana | 553 | 0,876 | 0,812 |
| Fruit grade | 223 | 0,846 | 0,837 | | Banano | 340 | 0,902 | 0,897 |
| FruitVision | 1 039 | 0,973 | 0,962 | | Mango | 246 | 0,931 | 0,941 |
| Sultana (uva, fresa) | 102 | 0,971 | 0,951 | | Uva | 221 | 0,954 | 0,932 |
| | | | | | Guayaba | 192 | 0,993 | 0,946 |
| | | | | | Naranja | 494 | 0,967 | 0,952 |
| | | | | | Lima | 108 | 0,977 | 0,956 |
| | | | | | Fresa | 41 | 0,976 | 0,976 |

Mango y uva ya tienen dos fuentes (Fruit grade/FruitVision y Sultana/FruitVision), así que su F1 ya no se apoya en una sola fuente como en v1.

## Calidad en el externo (Sultana)
| Modelo | F1 macro [IC 95 %] | Exactitud | Recall "mala" | Costo esperado | v1 (F1) |
|---|---|---|---|---|---|
| Mayoritaria | 0,322 [0,310; 0,334] | 0,476 | 0,000 | 2,098 | 0,322 |
| Regresión logística | 0,711 [0,690; 0,732] | 0,686 | 0,709 | 0,854 | 0,620 |
| SVM (RBF) | 0,657 [0,633; 0,679] | 0,653 | 0,554 | 1,147 | 0,597 |
| **Random Forest** | **0,765 [0,743; 0,786]** | 0,765 | **0,751** | **0,729** | 0,776 |

Recall de "mala" por especie (SVM / RF): manzana 0,257 / 0,607; banano 0,965 / 1,000; guayaba 0,808 / 0,917; naranja 0,180 / 0,480. RF en el externo: buena (554 buena, 4 regular, 153 mala); mala (195, 0, 589).

- FruitVision mejora la regresión logística (+0,09) y el SVM (+0,06) fuera de dominio; el RF queda igual (0,765 frente a 0,776, dentro del IC).
- La brecha test → externo sigue: el SVM pasa de 0,910 a 0,657. La fuente se adivina desde las características con 97,5 % de exactitud entre las cuatro fuentes con calidad.
- Naranja y manzana malas siguen siendo las más difíciles fuera de dominio.

## Especie
| | SVM (v1) | SVM (v2) | **MobileNetV2 + cabeza (v2.1)** |
|---|---|---|---|
| Test | — | 97,5 % | 97,0 % |
| Externo | 60,5 % | 55,7 % | **69,3 %** |
| Rechazo falso en externo (umbral 60 % o `other`) | — | 24,9 % | **8,0 %** |
| Rechazo correcto de `other` (test) | sin clase | 61,7 % | 58,3 % |

v2.1 en test por fuente: Fruit grade 99,6 %, FruitVision 97,9 %, Kaggle 94,7 %, Sultana (uva, fresa) 100 %; mango 246/246. Externo por especie: manzana 330/355, banano 385/397, naranja 259/389 (101 → manzana), **guayaba 33/354** (127 → manzana, 79 rechazadas): en train solo hay guayabas de Kaggle.

### Pipeline completo (como la app: calidad con la especie predicha por v2.1)
| | SVM | RF | Reg. logística |
|---|---|---|---|
| Test (F1 macro) | 0,909 | 0,890 | 0,748 |
| Externo (F1 macro) | 0,690 | 0,752 | 0,676 |

Con la especie predicha, el SVM en el externo sube a 0,690 (con la especie verdadera, 0,657), y el RF queda en 0,752.

## Versión 2.2: racimos de uva y mango rojo (especie)
Motivo: una prueba con fotos de internet mostró que los **racimos de uva** se leían como fresa o lima y los **mangos rojo-amarillos** (tipo Tommy) como manzana: en las fuentes la uva es un grano suelto (también en Fruits-360) y el mango es verde o amarillo.
- **WGISD** (racimos de viñedo con máscara de instancia): 369 racimos recortados de 40 fotos, sobre fondo blanco; partición por foto (train 274 / val 46 / test 48).
- **Fruits-360:** mango hasta 240 fotogramas por carpeta (Mango 1 y Mango Red 1).
- **Pesos por clase** en la cabeza (la manzana es la clase más numerosa). Validación dejando una fuente fuera: 0,681 con pesos frente a 0,679 sin pesos.
- **Aumento de mango rojo-amarillo** (tono desplazado solo en la fruta, `src/data/augment_mango.py`): **descartado**, baja la validación de 93,8 % a 92,2 % y el rechazo de `other` de 60 % a 40 % (`especie_v22_aumento.csv`).

| Especie | v2 (SVM) | v2.1 | **v2.2** |
|---|---|---|---|
| Validación | 91,2 % | 93,0 % | **93,8 %** |
| Test | 97,5 % | 97,0 % | 96,8 % |
| Racimos de WGISD (test, 48) | — | — | **97,9 %** |
| Externo (Sultana) | 55,7 % | 69,3 % | **71,2 %** |
| Rechazo correcto de `other` | 61,7 % | 58,3 % | **71,7 %** |
| Rechazo falso en externo | 24,9 % | 8,0 % | 8,7 % |

Externo por especie (v2.2): manzana 309/355, banano 380/397, naranja 285/389, guayaba 50/354. Pipeline completo con la especie predicha (v2.2): RF 0,891 en test y **0,758** en el externo; SVM 0,908 / 0,682.

**Prueba informal con 7 fotos de internet** (no se usaron para elegir nada): racimos de uva 2/4 reconocidos (antes 0/4; uno sale fresa y otro queda bajo el umbral del 60 %); mangos rojo-amarillos 0/3 (salen manzana). Sin una fuente con mangos rojos, este caso sigue fuera del alcance y se declara.

Exactitud de especie por fruta y fuente (v2.2): `especie_por_fuente_v22.csv`; resumen del alcance validado por fruta en el README (sección "Alcance y limitaciones por fruta").

## Búsqueda de hiperparámetros del deep learning y sobreajuste
- **CNN** (`tuning_cnn.csv`, solo validación): lr máximo 2e-3 → F1 0,766; lr 1e-3 → 0,748. Se mantiene 2e-3.
- **Cabeza de especie** (`tuning_cabeza_especie.csv`): 12 combinaciones (unidades 128/256/512 × dropout 0,3/0,5 × lr 1e-3/3e-4) entre 92,8 % y 94,0 % en validación (v2.1). La configuración usada (256, 0,3, 1e-3) queda a menos de 1 punto, dentro del ruido; la búsqueda se hizo después del test de v2.1 y no se cambió el modelo para no volver a consultar el test.
- **Sobreajuste** (`sobreajuste.csv`, F1 macro): logística train 0,751 / CV 0,729 / val 0,733 (sin brecha); SVM 0,980 / 0,880 / 0,883 (brecha 0,10); RF 0,986 / 0,849 / 0,867 (brecha 0,12). La CNN no sobreajusta (exactitud train 0,847 ≈ val 0,843). La brecha que importa es otra: test → externo.

## Lectura de los resultados
- Más fuentes ayudan: FruitVision da a mango y uva una segunda fuente y sube la calidad fuera de dominio de los modelos clásicos.
- La especie era el eslabón débil (reporte del usuario: un mango leído como naranja o manzana). Las características manuales no la transfieren entre cámaras; un extractor preentrenado congelado sí mejora (+24 puntos en validación entre fuentes, +14 en el externo).
- El test sobreestima el desempeño con fotos nuevas; la cifra útil para el usuario es la del externo: RF 0,765 con recall de "mala" 0,75. El sistema es apoyo, no sustituto de la inspección.
- Pendiente: fuentes de guayaba, "naranja" y mango rojo-amarillo; fotos de Colombia; rechazo de frutas fuera del alcance (72 % en v2.2).
