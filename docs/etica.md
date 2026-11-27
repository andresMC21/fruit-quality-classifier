# Ética (PI1)

**Marcos de referencia** (citar en `referencias.md`):
- *ACM Code of Ethics and Professional Conduct* (2018): principios 1.2 Evitar daño, 1.3 Honestidad, 1.4 Equidad / no discriminación, 1.5 Respetar el trabajo ajeno, 1.6 Privacidad y 2.5 Evaluar a fondo los sistemas y sus riesgos.
- *IEEE Code of Ethics*.
- **Ley 842 de 2003** (Colombia), que incluye el código de ética para el ejercicio de la ingeniería.
- **Ley 1581 de 2012** (Colombia), de protección de datos personales: aplica si una foto capta rostros o datos de personas.

> El documento de competencias se redactó para un caso de video (vigilancia, privacidad). Aquí hay que adaptarlo:
> en fotos de frutas la privacidad pesa poco, y los dilemas centrales son el **sesgo**, el **costo de los errores** y la **honestidad de las métricas**.

## Tabla de decisiones éticas (obligatoria en el informe)
Las medidas propuestas son un punto de partida: hay que reemplazarlas por lo que el grupo haga realmente y enlazar la evidencia.

| # | Situación / decisión del proyecto | Dilema o responsabilidad | Principio | Medida adoptada (y evidencia) |
|---|---|---|---|---|
| 1 | Privacidad en las imágenes (el grupo decidió **no tomar fotos propias**) | Algunas bases públicas pueden traer personas o manos en el encuadre | ACM 1.6; Ley 1581 de 2012 | No se recolectan datos personales; las imágenes de terceros se usan solo como están publicadas, sin redistribuirlas (`.gitignore`). Si se añaden fotos propias: permiso del vendedor y encuadre solo de la fruta. |
| 2 | Datos públicos de frutas de Asia (Kaggle/FruitNet y AFruitDB) con otras variedades y luz | Sesgo: el modelo puede fallar con variedades locales | ACM 1.4, 2.5 | Sin fotos propias, se reservó una fuente entera (Sultana: manzana, banano, naranja, guayaba) como **test externo**. Resultado: F1 en test 0,910 (SVM) y en la fuente externa 0,765 (RF, por defecto) / 0,657 (SVM); reportamos ambos. En v2 se añadió FruitVision para que mango y uva tengan dos fuentes. Métricas por fuente y alcance de 8 especies declarado. |
| 3 | La clase "regular" solo existe en Fruit grade | Atajo: el modelo aprende la fuente y no la calidad, y sus métricas engañan | ACM 1.3, 2.5 | Se midió: la fuente es adivinable con 97,5 % de exactitud entre cuatro fuentes; se neutraliza el fondo y se reporta por fuente. El test externo confirma el riesgo (en otra fuente el SVM detecta 55 % de las frutas malas y el RF 75 %; por eso la app usa el RF por defecto). Riesgo residual declarado. |
| 4 | Falsos "mala" (descartar fruta buena) frente a falsos "buena" (vender fruta en mal estado) | Desperdicio y pérdida para el productor frente a riesgo para el consumidor | ACM 1.2; Ley 842 de 2003 (bienestar público) | Matriz de costos explícita (`configs/config.yaml`); reportar el recall de "mala"; mostrar la confianza y recomendar revisión humana si es baja. |
| 5 | Reportar métricas | Inflar resultados con fuga de datos o con varias consultas al test | ACM 1.3 | Split por `fruit_id` y por duplicados; tocar el test una sola vez; publicar el split. |
| 6 | Uso de datos y código de terceros | Plagio o fraude | ACM 1.5 | Citar en el README, el informe y `referencias.md`; respetar la licencia CC BY 4.0. |
| 7 | Uso del sistema por terceros (comerciantes) | Delegar decisiones económicas en un prototipo | ACM 2.5 | Aviso en la interfaz (`streamlit_app.py`): prototipo académico; rechazo de frutas no reconocidas y revisión humana con confianza baja. |
| 8 | Frutas fuera del alcance (pera, piña, sandía, etc.) | Dar una calidad confiada para una fruta que el modelo nunca vio | ACM 1.2, 2.5 | Clase `other` (36 frutas y verduras de Fruits-360) + umbral de confianza de especie (60 %); la app dice "fruta no reconocida". Medido en test (v2.2): rechaza 72 % de las frutas ajenas, así que no es un filtro seguro; los mangos rojos se leen como manzana (declarado en la app). |
| 9 | Descartar fotos en las que la segmentación falla | Sesgo de selección: las fotos "difíciles" (manos, fondos) no se evalúan | ACM 1.3 | Tasa de fallo publicada por especie y fuente (0-18 %); la app pide repetir la foto en vez de predecir. |
