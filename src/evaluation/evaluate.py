"""Metricas en TEST (se ejecuta UNA vez sobre el modelo congelado).

- accuracy, F1 macro, recall por clase (en especial "mala"), matriz de confusion
- costo esperado con cfg["metrics"]["cost_matrix"]
- resultados desglosados por fuente (kaggle / mixed_quality / own): detecta atajos y cambio de dominio
- comparacion con la linea base y entre modelos: McNemar (statsmodels) o IC bootstrap del 95 % del F1 macro
- tamano: MAE en mm frente a diameter_mm
Debe usar src/inference/predict.py para preprocesar igual que la app.
"""


def evaluate(model_path: str, cfg: dict):
    raise NotImplementedError
