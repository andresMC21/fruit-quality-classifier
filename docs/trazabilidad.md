# Matriz de trazabilidad de requisitos

Fuentes: **PF** = `0_Proyecto Final 26-2.pdf`; **CC** = `1_competencias_consigna_Actualizada.pdf`; **R** = `Rubrica_proyecto.xlsx - Rúbrica.pdf` (criterios 1–8, sin pesos publicados).
Tipo: **E** = requisito explícito, **Rec** = recomendación del documento, **D** = decisión propuesta por el grupo.
Estado al 2026-11-17: ninguna evidencia producida todavía, salvo la revisión de la plantilla.

| # | Requisito | Tipo | Fuente | Evidencia esperada | Archivo / entregable | Rúbrica | Estado |
|---|---|---|---|---|---|---|---|
| 1 | Grupo de 3 estudiantes (grupo de 4 aprobado por el profesor, confirmado por el equipo el 2026-11-17) | E | PF p.1 | Nombres reales en informe y README | `README.md`, informe | 8 | Hecho en README |
| 2 | Aplicar CRISP-DM adaptado, con evidencia en las 6 fases | E | PF p.1, §4 p.3–4 | Un documento por fase con evidencia real | `docs/0X_*.md`, notebooks | 4, 8 | Pendiente |
| 3 | Diagrama de flujo CRISP-DM personalizado | E | PF §4 p.4; §7 p.5 | Diagrama vectorial con iteraciones reales | `docs/diagramas/` | 8 | Pendiente |
| 4 | Entrada: imagen de una fruta/verdura individual, fondo simple | E | PF §1.2 p.2 | Protocolo de captura; validación de entrada | `docs/guia_anotacion.md` | 3 | Pendiente |
| 5 | Salida: ≥3 clases de calidad definidas por el grupo | E | PF §1.3 p.2 | Guía de etiquetado con criterios observables | `docs/guia_anotacion.md` | 3, 4 | Pendiente (ver inconsistencia I2) |
| 6 | Salida: tamaño (P/M/G) o diámetro en píxeles normalizados | E | PF §1.3 p.2 | Definición de la medida, referencia, error | `src/models/size_estimator.py`, `docs/05_evaluacion.md` | 3, 5, 6 | Pendiente |
| 7 | Interfaz: cargar imagen o capturar con cámara y mostrar predicción | E | PF §1.4 p.2 | App funcional, capturas, video | `src/app/` | 8 | Pendiente |
| 8 | Despliegue: ver "en tiempo real" la calidad ante la cámara | E | PF §4.6 p.4 | Modo cámara continuo con latencia medida | `src/app/`, informe | 6 | Pendiente (ver I3) |
| 9 | Raspberry Pi / línea de empaque | Opcional | PF §1.5 p.2 | — | — | — | Fuera de alcance inicial |
| 10 | Usar dataset Kaggle *Fruit Quality Classification* | E | PF §2.1 p.2 | Inventario, EDA, licencia, cita al origen FruitNet | `data/README.md`, notebook EDA | 6, 8 | Descargado e inventariado (`docs/02_comprension_datos.md`) |
| 11 | Segmentar individualmente frutas de `mixed_quality` si se usan | E (condicional) | PF §2.1 p.2 | Recortes con ID de imagen origen | `data/annotations/` | 5 | Pendiente |
| 12 | Recolectar ≥30–50 imágenes propias con variedad de madurez, tamaño y defectos | E | PF §2.2 p.2 | 50 imágenes (decisión conservadora D), metadatos | `data/raw/own_images/`, `data/annotations/annotations.csv` | — (pregunta de PF §5) | Pendiente |
| 13 | Etiquetar imágenes propias con calidad y tamaño, coherente con definiciones | E | PF §2.3 p.3 | CSV de anotaciones + acuerdo entre anotadores | `data/annotations/` | 5, 6 | Pendiente |
| 14 | Mecanismo para compartir anotaciones entre grupos | E (definido "entre todos") | PF §2.3 p.3 | Formato de intercambio + enlace | `data/README.md` | — | Depende de la clase |
| 15 | ≥2 modelos ML tradicionales con ajuste por CV o rejilla | E | PF §3 p.3 | Configuraciones, resultados de CV | `configs/`, `experiments/` | 4, 5 | Pendiente |
| 16 | ≥1 modelo de deep learning (CNN pequeña; TL permitido con capas congeladas) | E / Rec (priorizar desde cero) | PF §3 p.3 | Arquitectura, curvas, procedimiento de ajuste | `src/models/cnn.py`, `experiments/` | 4, 5, 6 | Pendiente |
| 17 | Comparación con línea base | E | PF §4.5 p.3 | Tabla con baseline | informe §Resultados | 6 | Pendiente |
| 18 | EDA: distribución de clases, desbalance, calidad, variabilidad | E | PF §4.2 p.3 | Notebook + figuras | `notebooks/02_eda.ipynb` | 6 | Pendiente |
| 19 | Manejo de desbalance (si aplica) | E (condicional) | PF §4.3 p.3 | Justificación + método | `docs/03_preparacion_datos.md` | 5 | Pendiente |
| 20 | Métricas en prueba, análisis de errores, sobreajuste, generalización | E | PF §4.5 p.3; §7.6–7 p.5; CC p.3 | Tablas, matrices, curvas, galería de errores | `experiments/`, informe | 6 | Pendiente |
| 21 | Análisis ético con código profesional (tabla) — PI1 | E | CC p.1–2; PF §6 p.4 | Tabla situación→riesgo→principio→medida→evidencia | `docs/etica.md` + versión en informe | 1 | Pendiente |
| 22 | Impactos social, económico, ambiental, global, inicial y final — PI2 | E | CC p.2; PF §6 p.4 | Matriz con mitigaciones e indicadores | `docs/impactos.md` + informe | 2 | Pendiente |
| 23 | Formulación matemática, estrategia, fundamentos, análisis, conclusiones — PI3 | E | CC p.2–3 | Ecuaciones conectadas con código | `docs/formulacion_matematica.md` + informe | 3, 4, 5, 6, 7 | Pendiente |
| 24 | Informe máx. 7 páginas con 9 secciones; IEEE | E | PF §7 p.5 | Informe en plantilla | `report/` | 8 | Plantilla revisada |
| 25 | Figuras vectoriales; diagramas de metodología y arquitectura | E | PF §6 p.5 | SVG/PDF/EMF | `report/figures/` | 8 | Pendiente |
| 26 | Código documentado; referenciar datos y código de terceros (si no, fraude) | E | PF §6 p.5 | Docstrings, sección de atribuciones | `README.md`, código | 8 | Pendiente |
| 27 | Repositorio GitHub organizado según fases de la metodología | E | PF §5 p.4 | Estructura + README | repo | 8 | Inicio |
| 28 | Integrantes + enlace del repo por documento que compartirá el profesor | E | PF §5 p.4 | — | Documento del profesor | — | Pendiente (mecanismo no publicado) |
| 29 | Video ≤10 min | E | PF §6 p.4 | Video | enlace en README | — | Pendiente |
| 30 | Revisar artículos de NeurIPS, ICML, ICLR, etc. antes del informe | Rec | PF Observaciones p.5 | Referencias leídas | informe | 8 | Pendiente |
