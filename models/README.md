# Modelos exportados (versionados)
`python -m src.main export --model rf --species tl` deja aquí lo que usa la app, sin reentrenar:
- `species_tl.keras` + `species_tl.classes.json`: especie (9 clases, con `other`) = MobileNetV2 congelada + cabeza propia.
- `pretrained/` (no versionado, `*.h5`): pesos de MobileNetV2 (ImageNet, 128x128, sin capa superior) para reentrenar la cabeza; se descargan de github.com/JonathanCMitchell/mobilenet_v2_keras (release v1.1). `species_tl.keras` ya los incluye, así que la app no los necesita.
- `quality_rf.joblib` (por defecto), `quality_svm.joblib`, `quality_logreg.joblib`: calidad (59 características + especie en one-hot).
- `species_svm.joblib`: especie de la v2 (59 características), como referencia.
- `meta.json`: clases, especies, tamaño de recorte, umbrales de tamaño por especie y modelo por defecto.
Los checkpoints intermedios van en `experiments/checkpoints/` (no versionados).
