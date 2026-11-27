"""Ejecuta/reanuda la CNN: python -m src.training.run_cnn [épocas] [segundos] [lr] [dropout] [tag]

Búsqueda de hiperparámetros (solo validación): cnn_v4 (lr 2e-3, dropout 0,3) y cnn_lr1e3 (1e-3, 0,3); el
argumento dropout permite ampliar la rejilla. Resultado en experiments/results/tables/tuning_cnn.csv (src.training.tune_cnn).
"""
import sys

from src.training.train_cnn import train_cnn
from src.utils.config import load_config

a = sys.argv[1:]
epochs = int(a[0]) if len(a) > 0 else 30
budget = int(a[1]) if len(a) > 1 else 125
lr = float(a[2]) if len(a) > 2 else 2e-3
dropout = float(a[3]) if len(a) > 3 else 0.3
tag = a[4] if len(a) > 4 else "cnn_v4"
m, h = train_cnn(load_config(), epochs=epochs, budget_s=budget, resume=True, lr=lr, dropout=dropout, tag=tag)
print(tag, "epochs:", len(h["loss"]), "best val_loss", round(h["best"], 3), "done", h.get("done"))
