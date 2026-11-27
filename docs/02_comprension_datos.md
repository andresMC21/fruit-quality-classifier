# Fase 2 - Comprensión de los datos (EDA)
Figuras vectoriales y tablas en `experiments/results/eda/` (`python -m src.data.eda`).

## Fuentes usadas
| Fuente | Contenido | Resolución | Fondo | Papel |
|---|---|---|---|---|
| Kaggle *Fruit Quality Classification* (FruitNet) | Buena/mala de manzana, banano, guayaba, lima, naranja (11 325 imágenes) | 256 px | muy variado (pasto, baldosas, manos) | entrenamiento |
| Fruit grade (AFruitDB / Asian Fruits) | Grados 1.º/2.º/3.º de manzana, banano, mango (1 584) | 4000x3000 | papel blanco o gris | entrenamiento |
| **Sultana et al.** (Mendeley, doi 10.17632/bdd69gyhv8.1) | Fresca/podrida; 200 por clase; tiendas y campos, cámara réflex | ~4160x3120 | variado | uva y fresa: entrenamiento; manzana, banano, naranja, guayaba: **test externo** |
| **FruitVision** (v2) | Fresh/Rotten/Formalin-mixed de manzana, banano, uva, mango, naranja; se usan Fresh → buena y Rotten → mala (6 978) | ~960x1280 | gris liso, una fruta | entrenamiento, validación y test |
| **Fruits-360** (v2, GitHub fruits-360-100x100) | Variedades de frutas girando en una plataforma; selección fija de 2 244 imágenes (8 especies + 864 de 36 frutas/verduras ajenas = `other`) | 100x100 | blanco | **solo especie** (train; `other` también en val/test) |
| **WGISD** (v2.2, Embrapa; GitHub thsant/wgisd) | Racimos de uva de viñedo con máscara de instancia; 369 racimos recortados de 40 fotos (5 variedades) | 2048x1365 | viñedo → fondo blanco tras recortar con la máscara | **solo especie** (train, val y test por foto) |
| `mixed_quality` (Kaggle) | Varias frutas por foto, sin etiqueta | 256 px | pasto | no usada |

Total en el manifiesto: **25 318 imágenes** (15 309 de v1 + 6 978 de FruitVision + 2 662 de Fruits-360 + 369 de WGISD) de **8 especies** (manzana, banano, mango, guayaba, naranja, lima, uva, fresa) más la clase `other`.

## Fuentes evaluadas y descartadas
- **FruQ-DB y FruQ-multi** (Zenodo 7224690): 5 647 y 9 370 fotogramas de videos en lapso de tiempo. Cada carpeta es casi una sola fruta física fotografiada mientras se pudre;
  las clases buena/mild/podrida son momentos de la misma pieza. Los nombres de carpeta no coinciden con la fruta ("PearQ" muestra un aguacate; el artículo original habla de aguacate,
  y "GrapeQ" muestra una toronja partida). FruQ-DB no trae la especie. Un test por fotogramas mediría memoria y no generalización. Fresa no tiene clase buena.
- **FruityHub** (Mendeley ch5vgt8v9k): 1 557 fotogramas (`frame_1000.jpg`…) de videos en lapso de tiempo, una fruta por carpeta, con un logo "TEMPONAUT" y un
  **contador de días superpuesto** en la imagen ("140 DAYS"), que delata la clase. Se descartó por riesgo de que el modelo aprenda a leer el contador.
- **FruitQuality** (carpeta local): copia de la base de Kaggle (mismos nombres y bytes en una muestra de 40 de 40). No aporta.
- **FruitNet / AFruitDB en Hugging Face (Project-AgML):** espejos de FruitNet y Fruit grade, ya incluidas.
- **FruitVision `Formalin-mixed`:** fruta tratada con formol, criterio químico no visible en la foto; se excluye.
- Fruta fuera de la lista del curso (granada, papaya, tomate, uva birmana, jujube): no se usa.

## Hallazgos
1. **"Regular" solo existe en Fruit grade** (659 imágenes tras limitar por grupo; 0 en Kaggle y en Sultana). Guayaba, naranja, lima, uva y fresa son casi binarias.
2. **Fuente adivinable:** con las características, la fuente se predice con 98,9 % de exactitud entre Fruit grade y Kaggle (v1) y con **97,5 %** entre las cuatro fuentes con calidad (v2, Random Forest, CV agrupada sobre train), incluso con el fondo neutralizado.
   Es un atajo posible, y la caída en el conjunto externo (`05_evaluacion.md`) lo confirma.
3. **Especies de una sola fuente.** En v1, mango (solo Fruit grade), uva y fresa (solo Sultana) y lima (solo Kaggle) dependían de una fuente; el clasificador de especie aprendía el estilo de foto (caso reportado: un mango leído como naranja o manzana). En v2, FruitVision da una segunda fuente a mango y uva; fresa, lima y guayaba siguen con una fuente con calidad (Fruits-360 aporta otra solo para la especie).
4. **Formas y variedades ausentes:** en todas las fuentes la uva es un grano suelto (también en Fruits-360) y el mango es verde o amarillo; los racimos y los mangos rojos (tipo Tommy) no estaban. WGISD aporta racimos; no se encontró una fuente accesible de mangos rojos (las bases de variedades están en Mendeley/Hugging Face, fuera de la red permitida, y son de Bangladesh).
5. **Una misma especie no se ve igual en todas las fuentes:** las "naranjas" de Kaggle/FruitNet (India) son cítricos verde-amarillos; las de FruitVision y Sultana, anaranjadas.
6. **Duplicados:** 15 309 imágenes forman 8 340 grupos. Kaggle es muy redundante (ráfagas): 4,6 imágenes por grupo en lima, 3,3 en banano. Sultana y Fruit grade casi no repiten (1,0-1,1), aunque Fruit grade puede tener vistas distintas de una misma fruta que el hash no enlaza. Grupo más grande: 162 imágenes. Se conservan máx. **3 por grupo** (10 647 en v1). FruitVision no trae hora en el nombre y repite la misma fruta girada en fotos seguidas; el pHash no une vistas rotadas y, con el fondo gris uniforme, encadena frutas distintas, así que ahí la unidad de partición es un **tramo de ~30 fotos consecutivas** (186 tramos) y los duplicados son archivos idénticos (MD5). Se excluyen 83 fotos de FruitVision casi idénticas (pHash ≤ 6) a otra fuente de la misma especie. Total v2: 16 651 grupos, 8 600 unidades de partición, **19 786 imágenes conservadas**.
7. **Tamaño:** el diámetro relativo (d/lado corto) no es comparable entre fuentes; los umbrales se calculan por especie y fuente (`04_tamano_relativo`). **No hay fotos propias**, así que no hay mm ni MAE.
8. **Segmentación:** Otsu funciona con fondos lisos; Kaggle exige GrabCut en 23-50 % de los casos. Tasa de fallo: 0 % (mango, uva, banano en Fruit grade y Sultana) a 29 % (fresa en Sultana, por cajas de varias fresas). FruitVision falla en 0,2 % y Fruits-360 en 0,6 %. Las manos entran en la máscara en parte de las fotos de Kaggle "mala".

## Calidad de las etiquetas
Las etiquetas públicas son **débiles** (vienen de la carpeta o del grado de la base). Sin fotos propias no se calculó kappa de Cohen ni se revisó una muestra con los criterios de `guia_anotacion.md`; queda como limitación.
