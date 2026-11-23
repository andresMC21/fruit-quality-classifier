# Fase 2 - Comprensión de los datos (EDA)
Figuras vectoriales y tablas en `experiments/results/eda/` (`python -m src.data.eda`).

## Fuentes usadas
| Fuente | Contenido | Resolución | Fondo | Papel |
|---|---|---|---|---|
| Kaggle *Fruit Quality Classification* (FruitNet) | Buena/mala de manzana, banano, guayaba, lima, naranja (11 325 imágenes) | 256 px | muy variado (pasto, baldosas, manos) | entrenamiento |
| Fruit grade (AFruitDB / Asian Fruits) | Grados 1.º/2.º/3.º de manzana, banano, mango (1 584) | 4000x3000 | papel blanco o gris | entrenamiento |
| **Sultana et al.** (Mendeley, doi 10.17632/bdd69gyhv8.1) | Fresca/podrida; 200 por clase; tiendas y campos, cámara réflex | ~4160x3120 | variado | uva y fresa: entrenamiento; manzana, banano, naranja, guayaba: **test externo** |
| `mixed_quality` (Kaggle) | Varias frutas por foto, sin etiqueta | 256 px | pasto | no usada |

Total en el manifiesto: **15 309 imágenes** de **8 especies** (manzana, banano, mango, guayaba, naranja, lima, uva, fresa).

## Fuentes evaluadas y descartadas
- **FruQ-DB y FruQ-multi** (Zenodo 7224690): 5 647 y 9 370 fotogramas de videos en lapso de tiempo. Cada carpeta es casi una sola fruta física fotografiada mientras se pudre;
  las clases buena/mild/podrida son momentos de la misma pieza. Los nombres de carpeta no coinciden con la fruta ("PearQ" muestra un aguacate; el artículo original habla de aguacate,
  y "GrapeQ" muestra una toronja partida). FruQ-DB no trae la especie. Un test por fotogramas mediría memoria y no generalización. Fresa no tiene clase buena.
- **FruityHub** (Mendeley ch5vgt8v9k): 1 557 fotogramas (`frame_1000.jpg`…) de videos en lapso de tiempo, una fruta por carpeta, con un logo "TEMPONAUT" y un
  **contador de días superpuesto** en la imagen ("140 DAYS"), que delata la clase. Se descartó por riesgo de que el modelo aprenda a leer el contador.
- Fruta fuera de la lista del curso (granada, papaya, tomate, uva birmana, jujube): no se usa.

## Hallazgos
1. **"Regular" solo existe en Fruit grade** (659 imágenes tras limitar por grupo; 0 en Kaggle y en Sultana). Guayaba, naranja, lima, uva y fresa son casi binarias.
2. **Fuente adivinable:** un SVM predice la fuente (Fruit grade/Kaggle) con 98,8 % de exactitud en validación con las características, incluso con el fondo neutralizado.
   Es un atajo posible, y la caída en el conjunto externo (`05_evaluacion.md`) lo confirma.
3. **Uva y fresa solo existen en una fuente (Sultana)**: especie y fuente quedan confundidas, así que sus buenas métricas no prueban generalización.
4. **Duplicados:** 15 309 imágenes forman 8 340 grupos. Kaggle es muy redundante (ráfagas): 4,6 imágenes por grupo en lima, 3,3 en banano. Sultana y Fruit grade casi no repiten (1,0-1,1), aunque Fruit grade puede tener vistas distintas de una misma fruta que el hash no enlaza. Grupo más grande: 162 imágenes. Se conservan máx. **3 por grupo** (10 647).
5. **Tamaño:** el diámetro relativo (d/lado corto) no es comparable entre fuentes; los umbrales se calculan por especie y fuente (`04_tamano_relativo`). **No hay fotos propias**, así que no hay mm ni MAE.
6. **Segmentación:** Otsu funciona con fondos lisos; Kaggle exige GrabCut en 23-50 % de los casos. Tasa de fallo: 0 % (mango, uva, banano en Fruit grade y Sultana) a 29 % (fresa en Sultana, por cajas de varias fresas). Las manos entran en la máscara en parte de las fotos de Kaggle "mala".

## Calidad de las etiquetas
Las etiquetas públicas son **débiles** (vienen de la carpeta o del grado de la base). Sin fotos propias no se calculó kappa de Cohen ni se revisó una muestra con los criterios de `guia_anotacion.md`; queda como limitación.
