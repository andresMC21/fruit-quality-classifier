# Guía de anotación y protocolo de captura

Especie de trabajo: **manzana**, de varias variedades (las más comunes que se consigan, indicación del profesor).

## 1. Definición de clases de calidad
Las clases son **ordinales** (buena > regular > mala). Cada criterio debe poder verificarse mirando la foto.
Se etiqueta la **calidad visual de la cara visible**, no la madurez, el sabor ni la inocuidad: una foto no demuestra que la fruta sea segura para comer.
El **color propio de la variedad** (verde, amarilla, roja, bicolor) no es un defecto.

| Clase | Criterios visuales | Foto de ejemplo |
|---|---|---|
| Buena | Piel íntegra. Sin golpes, cortes, podredumbre ni moho. Se permiten lenticelas (puntos naturales), *russeting* de la variedad y mezcla de colores. | |
| Regular | Solo defectos superficiales: golpe sin piel rota, raspón, mancha localizada, arrugamiento o deformidad leve. **Área total con defecto < 10 %** del área visible. Apta para consumo o proceso. | |
| Mala | Cualquiera de: podredumbre (zona marrón blanda o acuosa), moho, herida abierta con pulpa expuesta, golpes extensos, arrugamiento severo, o **área con defecto ≥ 10 %**. No apta para la venta en fresco. | |

Reglas:
1. Manda el criterio más grave que se cumpla.
2. Si hay duda, se marca y se decide en grupo mirando la imagen; no se resuelve "por defecto" hacia una clase.
3. Se excluye la imagen (con motivo) si tiene varias frutas, no es manzana, está borrosa o la fruta está cortada.
4. El umbral del 10 % es provisional: se ajusta en la sesión de calibración (sección 4) y luego no se cambia.

> El dataset de Kaggle solo trae **buena** y **mala**, y su carpeta no siempre es confiable (`docs/02_comprension_datos.md`).
> Por eso un subconjunto de Kaggle se **re-etiqueta** con estos criterios (sección 5). Las imágenes de "regular" saldrán de
> ese re-etiquetado, de los recortes de `mixed_quality`, de las fotos propias y de las de otros grupos.

## 2. Tamaño
- Se mide el **diámetro ecuatorial máximo real** (regla o calibrador) de cada fruta propia: dos medidas perpendiculares y su promedio, en `diameter_mm`.
- Cada foto incluye un **objeto de referencia** sobre la mesa, junto a la fruta sin tocarla y sin tocar el borde de la foto: un **cuadrado negro impreso de 50 mm** (medir el lado real con regla después de imprimir y anotarlo). El código detecta ese cuadrado; una moneda o una tapa no se detectan. No usar cédulas ni tarjetas bancarias.
  Sin referencia, el área en píxeles depende de la distancia a la cámara y no mide tamaño.
- **Perspectiva:** el contorno más ancho de la manzana está a unos d/2 sobre la mesa, más cerca de la cámara que la referencia, así que se ve más grande: error ≈ (d/2)/H. Con d = 75 mm, a H = 30 cm el error es ≈ 12 % y a 50 cm ≈ 7,5 %. Se corrige con la altura H medida (`docs/formulacion_matematica.md`), pero conviene H ≥ 45 cm.
- Los umbrales pequeño/mediano/grande son **por especie** y se calculan con los terciles de `diameter_mm` del conjunto de entrenamiento.
- Las fotos de Kaggle no tienen escala: en ellas solo hay tamaño relativo y no se evalúa tamaño físico.

## 3. Protocolo de captura
- Una fruta por foto, fondo uniforme y mate (cartulina blanca o gris), sin sombras duras.
- Celular en un soporte, **paralelo a la mesa (vista cenital)**, a una altura H fija y medida (≥ 45 cm); sin zoom digital, modo retrato ni filtros. Anotar el dispositivo y H.
- Fruta centrada, con margen alrededor mayor que su diámetro.
- Varias vistas de la misma fruta (2-4) con el mismo `fruit_id`; la primera, con el defecto más visible hacia la cámara.
- Variar a propósito la iluminación (natural/artificial), la madurez, la variedad y los defectos, para que el modelo no aprenda la luz en lugar de la calidad.
- Cubrir **todas las clases con todas las condiciones**. Si todas las "regulares" son fotos propias y todas las "buenas" son de Kaggle, el modelo aprende la fuente y no la calidad.
- Para conseguir defectos sin desperdiciar: comprar fruta de oferta o "de segunda", y dejar algunas unos días a temperatura ambiente y volver a fotografiarlas (mismo `fruit_id`, otra fecha).
- En mercados, fotografiar solo la fruta, sin rostros ni puestos identificables, y con permiso del vendedor (ver `etica.md`). Antes de compartir las fotos, quitar la ubicación GPS de los metadatos.
- Nombre de archivo: `<fruit_id>_<vista>.jpg`, por ejemplo `own_M001_a.jpg`.

## 4. Calidad de las etiquetas
- **Calibración:** antes de etiquetar, los cuatro revisan juntos unas 20 imágenes, ajustan el umbral del 10 % y guardan 2-3 ejemplos por clase en la columna "Foto de ejemplo".
- Un subconjunto (≥ 20 %) lo anota un segundo integrante sin ver la primera etiqueta (`annotator2_quality`).
- Se calcula la **kappa de Cohen**, κ = (p_o − p_e) / (1 − p_e). Si κ < 0,6, se revisan los criterios antes de seguir anotando.
- Los desacuerdos se resuelven en grupo y se documenta la regla que se adoptó.

## 5. Re-etiquetado de Kaggle
1. `python -m src.data.build_label_set` agrupa casi-duplicados (pHash) y deja una cola con **una imagen por grupo**: los 378 grupos de "mala" y 400 de "buena" al azar (semilla del config). Son 778 imágenes: 100 de muestra común para kappa (la etiquetan dos personas por separado) y unas 170 por persona. La cola está versionada; no se regenera.
2. `streamlit run src/data/annotate_app.py`: cada quien elige su usuario y "Mi lote". Solo las dos personas acordadas para la kappa abren además "Muestra comun (kappa)". La app no muestra la carpeta original para no sesgar.
3. Además de la clase se marca si aparece una mano y si el fondo es simple. Sirven para medir si el modelo podría aprender "mano o fondo → calidad" en lugar de la calidad.
4. Cada quien versiona solo su archivo `data/annotations/kaggle_labels/<usuario>.csv`.
