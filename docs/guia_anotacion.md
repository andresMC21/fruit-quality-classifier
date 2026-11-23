# Guía de anotación y protocolo de captura

## 0. Alcance de especies
Manzana, banano, mango, guayaba, naranja y lima. Las bases públicas se mapean así: Kaggle *Good* = buena, *Bad* = mala; Fruit grade 1.º = buena, 2.º = regular, 3.º = mala. Esas etiquetas son débiles y se revisan con los criterios de abajo.

## 1. Definición de clases de calidad (completar y acordar con el curso)
Las clases son **ordinales** (buena > regular > mala). Cada criterio debe poder verificarse mirando la foto.

| Clase | Criterios visuales (ejemplo para manzana; adaptar) | Foto de ejemplo |
|---|---|---|
| Buena | Piel íntegra, color uniforme propio de la variedad, sin golpes ni manchas > X mm. | |
| Regular | Defectos leves: 1-2 golpes o manchas pequeñas, sobremaduración incipiente, deformidad leve. Apta para consumo o proceso. | |
| Mala | Podredumbre, moho, cortes, golpes extensos o arrugamiento. No apta para la venta en fresco. | |

> El dataset de Kaggle solo trae **buena** y **mala**. Las imágenes de "regular" saldrán de los recortes de `mixed_quality`,
> de las fotos propias y de las de otros grupos. Además, hay que **revisar una muestra de Kaggle** con estos criterios:
> algunas "buenas" o "malas" pueden ser "regulares" según su definición.

## 2. Tamaño
- Se mide el **diámetro real** (regla o calibrador) de cada fruta propia y se guarda en `diameter_mm`.
- Cada foto incluye un **objeto de referencia** de tamaño conocido (moneda o tapa) en el mismo plano que la fruta.
  Sin referencia, el área en píxeles depende de la distancia a la cámara y no mide tamaño.
- Los umbrales pequeño/mediano/grande son **por especie**: una lima grande es más pequeña que una manzana pequeña.

## 3. Protocolo de captura
- Una fruta por foto, fondo uniforme y mate (cartulina blanca o gris), sin sombras duras.
- Distancia fija (p. ej. 30 cm) con el celular paralelo al fondo; anotar el dispositivo.
- Varias vistas de la misma fruta (2-4) con el mismo `fruit_id`.
- Variar a propósito la iluminación (natural/artificial), la madurez y los defectos, para que el modelo no aprenda la luz en lugar de la calidad.
- Cubrir **todas las clases con todas las condiciones**. Si todas las "regulares" son fotos propias y todas las "buenas" son de Kaggle, el modelo aprenderá a distinguir la fuente (fondo, cámara) y no la calidad.
- En mercados, fotografiar solo la fruta, sin rostros ni puestos identificables, y con permiso del vendedor (ver `etica.md`).

## 4. Calidad de las etiquetas
- Un subconjunto (≥ 20 %) lo anota un segundo integrante sin ver la primera etiqueta (`annotator2_quality`).
- Se calcula la **kappa de Cohen**, κ = (p_o − p_e) / (1 − p_e). Si κ < 0,6, se revisan los criterios antes de seguir anotando.
- Los desacuerdos se resuelven en grupo y se documenta la regla que se adoptó.
