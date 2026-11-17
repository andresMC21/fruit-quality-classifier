"""Entrenamiento y registro de experimentos en experiments/results/tables/experiments_log.csv.

- Clasicos (baseline, logreg, svm, rf): features.csv, solo filas con seg_ok. Hiperparametros con
  src/training/tune.py (CV agrupada en train); val se usa para comparar modelos, nunca test.
- CNN: imagenes completas de splits.csv. Cada combinacion de cnn.grid es un entrenamiento con early
  stopping sobre la perdida de val; se guardan las curvas en experiments/logs/<exp_id>.csv.
Cada experimento escribe una fila en el registro y su modelo en experiments/checkpoints/.
"""
import csv
import datetime
import itertools
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from src.data.build_features import full_feature_names
from src.data.features import feature_names
from src.data.preprocess import preprocess
from src.models.classical import MODELS
from src.models.cnn import augment, build_cnn, count_parameters
from src.training.tune import tune
from src.utils.config import set_seed

RAW_DIR = Path("data/raw")
LOG_COLUMNS = ["exp_id", "date", "model", "hyperparams", "cv_f1_macro", "val_f1_macro", "test_f1_macro", "notes"]


def new_exp_id(model_name):
    return f"{model_name}_{datetime.datetime.now():%Y%m%d-%H%M%S-%f}"


def log_experiment(cfg, row):
    path = Path(cfg["paths"]["experiments_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def macro_f1(y_true, y_pred, classes):
    return float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))


def train(model_name: str, cfg: dict, features: str = "masked"):
    """features (solo clasicos): "masked" = fruta segmentada (solo seg_ok); "full" = imagen completa,
    todas las filas. "full" es el experimento de contraste para medir el atajo de fondo y mano."""
    set_seed(cfg["seed"])
    if model_name in MODELS:
        return train_classical(model_name, cfg, features)
    if model_name == "cnn":
        return train_cnn(cfg)
    raise ValueError(f"Modelo desconocido: {model_name}")


# ---------------------------------------------------------------- clasicos
def train_classical(model_name, cfg, features="masked"):
    table = pd.read_csv(cfg["paths"]["features"])
    if features == "masked":
        names = feature_names()
        table = table[table["seg_ok"]]
    elif features == "full":
        names = full_feature_names()
    else:
        raise ValueError(f"features debe ser 'masked' o 'full', no {features}")
    tr, va = table[table["split"] == "train"], table[table["split"] == "val"]
    model, params, cv_f1 = tune(model_name, cfg, tr[names].to_numpy(), tr["quality_class"], tr["fruit_id"])
    val_f1 = macro_f1(va["quality_class"], model.predict(va[names].to_numpy()), cfg["classes"]) if len(va) else float("nan")

    exp_id = new_exp_id(f"{model_name}-{features}")
    ckpt = Path(cfg["paths"]["checkpoints"]) / f"{exp_id}.joblib"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": names, "variant": features, "classes": cfg["classes"]}, ckpt)
    scope = "fruta segmentada, solo seg_ok" if features == "masked" else "imagen completa, todas las filas"
    log_experiment(cfg, dict(
        exp_id=exp_id, date=datetime.date.today().isoformat(), model=f"{model_name}-{features}",
        hyperparams=json.dumps(params), cv_f1_macro=round(cv_f1, 4), val_f1_macro=round(val_f1, 4),
        notes=f"{scope}; n_train={len(tr)}, n_val={len(va)}"))
    print(f"{exp_id}: CV F1 macro = {cv_f1:.3f}, val F1 macro = {val_f1:.3f}, {params}")
    return exp_id, val_f1


# ---------------------------------------------------------------- CNN
def load_images(filenames, size):
    """Imagenes completas -> tensor N x 3 x H x W en [0, 1] (mismo preprocess que la app)."""
    arr = []
    for f in filenames:
        img = cv2.imread(str(RAW_DIR / f))
        if img is None:
            raise FileNotFoundError(RAW_DIR / f)
        arr.append(preprocess(img, tuple(size)))
    return torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2).contiguous()


def class_weights(y, n_classes):
    """w_k = N / (K * N_k): cada clase pesa lo mismo en la perdida aunque tenga menos ejemplos."""
    counts = np.bincount(y, minlength=n_classes)
    return torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, X, batch_size):
    model.eval()
    return torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])


def fit_one(params, data, cfg, device):
    """Entrena una combinacion de hiperparametros. Devuelve (estado del mejor epoch, historial)."""
    X_tr, y_tr, X_va, y_va, weights = data
    c = cfg["cnn"]
    torch.manual_seed(cfg["seed"])
    model = build_cnn(n_classes=len(cfg["classes"]), dropout=params["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))
    gen = torch.Generator().manual_seed(cfg["seed"])

    history, best_loss, best_state, wait = [], float("inf"), None, 0
    for epoch in range(1, c["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(X_tr), generator=gen)
        total = 0.0
        for i in range(0, len(perm), c["batch_size"]):
            idx = perm[i:i + c["batch_size"]].to(device)
            opt.zero_grad()
            loss = loss_fn(model(augment(X_tr[idx])), y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        logits_tr = predict_logits(model, X_tr, 256)
        logits_va = predict_logits(model, X_va, 256)
        row = dict(
            epoch=epoch,
            train_loss=total / len(X_tr),
            val_loss=loss_fn(logits_va, y_va).item(),
            train_f1=macro_f1(y_tr.cpu(), logits_tr.argmax(1).cpu(), list(range(len(cfg["classes"])))),
            val_f1=macro_f1(y_va.cpu(), logits_va.argmax(1).cpu(), list(range(len(cfg["classes"])))),
        )
        history.append(row)
        if row["val_loss"] < best_loss:
            best_loss, wait = row["val_loss"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= c["patience"]:
                break
    return best_state, history


def train_cnn(cfg):
    torch.manual_seed(cfg["seed"])
    # Convoluciones deterministas en GPU (algo mas lentas); aun asi la GPU no garantiza igualdad bit a bit.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = pd.read_csv(cfg["split"]["output"])
    class_idx = {c: i for i, c in enumerate(cfg["classes"])}
    tr, va = splits[splits["split"] == "train"], splits[splits["split"] == "val"]
    y_tr_np = tr["quality_class"].map(class_idx).to_numpy()
    data = (
        load_images(tr["filename"], cfg["image_size"]).to(device),
        torch.tensor(y_tr_np, device=device),
        load_images(va["filename"], cfg["image_size"]).to(device),
        torch.tensor(va["quality_class"].map(class_idx).to_numpy(), device=device),
        class_weights(y_tr_np, len(cfg["classes"])),
    )

    grid = cfg["cnn"]["grid"]
    best = (None, -1.0)
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        state, history = fit_one(params, data, cfg, device)
        best_epoch = min(history, key=lambda h: h["val_loss"])

        exp_id = new_exp_id("cnn")
        for key in ("logs", "checkpoints"):
            Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(Path(cfg["paths"]["logs"]) / f"{exp_id}.csv", index=False)
        torch.save({"state_dict": state, "params": params, "classes": cfg["classes"],
                    "image_size": cfg["image_size"]}, Path(cfg["paths"]["checkpoints"]) / f"{exp_id}.pt")
        log_experiment(cfg, dict(
            exp_id=exp_id, date=datetime.date.today().isoformat(), model="cnn",
            hyperparams=json.dumps(params), val_f1_macro=round(best_epoch["val_f1"], 4),
            notes=(f"imagen completa; val elige epoch e hiperparametros (val F1 optimista, la medida limpia es test); "
                   f"mejor epoch {best_epoch['epoch']}/{len(history)}; "
                   f"train F1 {best_epoch['train_f1']:.3f}; n_train={len(tr)}, n_val={len(va)}; "
                   f"{count_parameters(build_cnn())} parametros; {device}")))
        print(f"{exp_id}: {params} -> val F1 macro = {best_epoch['val_f1']:.3f} (epoch {best_epoch['epoch']})")
        if best_epoch["val_f1"] > best[1]:
            best = (exp_id, best_epoch["val_f1"])
    return best
