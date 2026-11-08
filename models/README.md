# Modelo final (versionado)
Aquí va **solo** el modelo seleccionado para la app (p. ej. `svm_final.joblib` o `cnn_final.keras`, de pocos MB)
y un `metadata.json` con: clases, tamaño de imagen, umbrales de tamaño, fecha, `exp_id` de `experiments_log.csv` y métricas de test.
Así la app funciona desde un clon nuevo sin reentrenar. Los checkpoints intermedios van en `experiments/checkpoints/` (no versionados).
