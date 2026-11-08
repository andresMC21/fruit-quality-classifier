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
| 1 | Tomar fotos en plazas de mercado | Captar personas o puestos sin consentimiento | ACM 1.6; Ley 1581 de 2012 | Pedir permiso al vendedor; encuadrar solo la fruta; borrar o desenfocar rostros antes de compartir. |
| 2 | Datos públicos de frutas de India (FruitNet) con otras variedades y luz | Sesgo: el modelo puede fallar con variedades locales | ACM 1.4, 2.5 | Test con imágenes propias; métricas por fuente y por condición de luz; declarar el alcance del modelo. |
| 3 | La clase "regular" sale casi solo de fotos propias | Atajo: el modelo aprende la fuente y no la calidad, y sus métricas engañan | ACM 1.3, 2.5 | Cubrir cada clase con cada fuente; análisis de errores por fuente. |
| 4 | Falsos "mala" (descartar fruta buena) frente a falsos "buena" (vender fruta en mal estado) | Desperdicio y pérdida para el productor frente a riesgo para el consumidor | ACM 1.2; Ley 842 de 2003 (bienestar público) | Matriz de costos explícita (`configs/config.yaml`); reportar el recall de "mala"; mostrar la confianza y recomendar revisión humana si es baja. |
| 5 | Reportar métricas | Inflar resultados con fuga de datos o con varias consultas al test | ACM 1.3 | Split por `fruit_id` y por duplicados; tocar el test una sola vez; publicar el split. |
| 6 | Uso de datos y código de terceros | Plagio o fraude | ACM 1.5 | Citar en el README, el informe y `referencias.md`; respetar la licencia CC BY 4.0. |
| 7 | Uso del sistema por terceros (comerciantes) | Delegar decisiones económicas en un prototipo | ACM 2.5 | Aviso en la interfaz: prototipo académico, no apto para certificar calidad. |
