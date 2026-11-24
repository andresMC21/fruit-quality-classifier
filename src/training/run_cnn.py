"""Ejecuta/reanuda el entrenamiento de la CNN: python -m src.training.run_cnn [épocas] [segundos]"""
import sys

from src.training.train_cnn import train_cnn
from src.utils.config import load_config

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 30
budget = int(sys.argv[2]) if len(sys.argv) > 2 else 125
m, h = train_cnn(load_config(), epochs=epochs, budget_s=budget, resume=True)
print("epochs:", len(h["loss"]), "best val_loss", round(h["best"], 3), "done", h.get("done"))
