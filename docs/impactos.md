# Impactos (PI2)
El enunciado pide un **análisis inicial** (antes de modelar, con lo que se sabía del problema y del EDA) y un **análisis final** (con los resultados). Cada celda se apoya en un dato o un argumento técnico.

## 1. Análisis inicial (antes de modelar)
Basado en el contexto del problema y en el EDA (`02_comprension_datos.md`); son hipótesis por comprobar.

| Dimensión | Impactos positivos esperados | Riesgos previstos | Mitigación planeada | Base de la hipótesis |
|---|---|---|---|---|
| **Social** | Estandarizar la calidad; consumidor mejor informado. | Desplazar a clasificadores manuales; desconfianza de pequeños productores; fotos en mercados (privacidad). | Herramienta de apoyo, no reemplazo; criterios transparentes; protocolo de captura sin personas. | La clasificación manual es lenta y subjetiva (enunciado). |
| **Económica** | Menos pérdida por mala clasificación; hardware barato (celular o webcam). | Errores con costo asimétrico: vender fruta mala cuesta más que descartar fruta buena. | Matriz de costos explícita (`configs/config.yaml`); reportar el recall de "mala". | Costo esperado de la línea base mayoritaria en validación: 1,85 (todo se declara "buena"). |
| **Ambiental** | Reducir el desperdicio de alimentos. | Energía de entrenamiento e inferencia; residuos si se usa hardware dedicado. | Preferir modelos livianos en CPU; inferencia en el celular o el PC. | Equipo disponible: CPU de 2 núcleos, sin GPU. |
| **Global** | Útil donde hay celulares y no hay equipos industriales. | Datos de India y Bangladesh: otras variedades, luz y cámaras; normas distintas por país (NTC de Icontec). | Medir en una fuente externa; documentar el alcance por fruta. | EDA: la fuente se adivina desde las características con 97,5 % de exactitud. |

## 2. Análisis final (con los resultados, v2.2)
| Dimensión | Lo que mostraron los resultados | Riesgo confirmado | Mitigación adoptada | Evidencia |
|---|---|---|---|---|
| **Social** | El sistema sirve como apoyo: muestra confianza, recorte y probabilidades. | En otra fuente se equivoca en 1 de cada 4 frutas malas; ninguna fruta validada con fotos de Colombia. | Aviso de alcance en la app; revisión humana si la confianza es baja; rechazo de frutas ajenas. | RF en el externo: F1 0,765, recall de "mala" 0,75 (`05_evaluacion.md`). |
| **Económica** | En las fuentes de entrenamiento el error es bajo. | Fuera de dominio el costo se multiplica por ~4,6. | Modelo por defecto elegido por robustez entre fuentes (RF), no por el mejor test (SVM). | Costo esperado: RF 0,159 en test y 0,729 en externo; SVM 0,100 y 1,147; mayoritaria 1,719 y 2,098. |
| **Ambiental** | Todo se entrena en CPU en minutos. | La CNN cuesta ~30 veces más que el SVM y rinde menos. | Modelos clásicos para la calidad; extractor preentrenado congelado para la especie (se reutiliza, no se entrena). | Ajuste: logística 13 s, SVM 43 s, RF 100 s, CNN 1 281 s; MobileNetV2 ~8 min una vez + 15 s la cabeza. No se midió energía. |
| **Global** | La especie mejora con un extractor general (71,2 % en otra fuente). | Variedades ausentes: mango rojo-amarillo (tipo Tommy) y guayaba de otra fuente no se reconocen. | Alcance por fruta documentado (README, informe, app); validar con datos de cada región antes de usarlo. | Por fuente en test (SVM): Kaggle 0,968, FruitVision 0,973, Fruit grade 0,846; guayaba en otra fuente 14 % (`especie_por_fuente_v22.csv`). |

**Cifra clave para las cuatro dimensiones:** en una fuente de fotos nueva, el modelo por defecto detecta 75 % de la fruta mala. El sistema es un apoyo y no sustituye la inspección humana.
