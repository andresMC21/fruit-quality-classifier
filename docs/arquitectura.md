# Arquitectura

```
foto (cámara o archivo)
   │
   ▼
segmentación (src/data/segment.py): distancia en Lab + Otsu → GrabCut con semilla → control por especie
   │  máscara + caja
   ▼
recorte 128x128, fondo gris 128, máscara en alfa (src/data/preprocess.py)
   │
   ├──► especie (src/inference/predict.py → models/species_tl.keras)
   │       MobileNetV2 congelada (ImageNet) → normalización → cabeza propia → 9 clases (8 frutas + other)
   │       other o confianza < 60 % → "fruta no reconocida"
   │
   ├──► 59 características (src/data/features.py): HSV, Lab, fracciones de color, LBP, GLCM, forma
   │       + especie predicha en one-hot → calidad (RF por defecto; SVM y logística en el selector)
   │       confianza < 50 % → "revisión humana"
   │
   └──► tamaño (src/models/size_estimator.py): d = 2√(A/π) → terciles por especie (meta.json); mm con objeto de referencia
   │
   ▼
interfaz Streamlit (src/app/streamlit_app.py)
```

## Módulos
| Carpeta | Contenido |
|---|---|
| `src/data/` | manifiesto de fuentes, deduplicación y unidades de partición, segmentación, recortes, características, split, EDA, recortes de WGISD, aumento de mango (probado y descartado) |
| `src/models/` | modelos clásicos y rejillas, CNN pequeña, extractor MobileNetV2 + cabeza, estimador de tamaño |
| `src/training/` | entrenamiento reanudable (CPU de 2 núcleos), búsquedas de hiperparámetros, exportación a `models/` |
| `src/evaluation/` | métricas y costo, evaluación con candado por versión, validación dejando una fuente fuera, sobreajuste, figuras |
| `src/inference/predict.py` | pipeline único que usan la app y la evaluación |
| `src/app/` | interfaz Streamlit |

## Decisiones de diseño
- **Un solo pipeline de inferencia** para la app y la evaluación: lo que se mide es lo que usa el usuario.
- **Fondo neutralizado** antes de extraer características: reduce el atajo de la fuente (fondo, manos, papel).
- **Especie y calidad separadas:** la especie usa un extractor general (generaliza entre cámaras); la calidad usa características interpretables de color y textura (el RF es el más robusto entre fuentes).
- **Todo reanudable y con semilla:** cada paso pesado guarda su avance (máximo ~3 min por ejecución en el equipo de trabajo) y GrabCut usa semilla fija.
- **Respaldo sin TensorFlow:** si falta, la app usa el SVM de especie y lo avisa.
