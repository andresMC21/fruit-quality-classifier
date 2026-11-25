"""Métricas comunes (validación y test): F1 macro, recall de la clase crítica y costo esperado."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def expected_cost(y_true, y_pred, classes, cost_matrix) -> float:
    """Costo medio por imagen: C[real, predicho] (filas = real). Ver config.metrics.cost_matrix."""
    idx = {c: i for i, c in enumerate(classes)}
    C = np.asarray(cost_matrix, float)
    return float(np.mean([C[idx[t], idx[p]] for t, p in zip(y_true, y_pred)]))


def summarize(y_true, y_pred, cfg: dict) -> dict:
    classes = cfg["classes"]
    crit = cfg["metrics"]["critical_class"]
    present = [c for c in classes if c in set(y_true)]
    out = dict(n=len(y_true), accuracy=accuracy_score(y_true, y_pred),
               f1_macro=f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0),
               cost=expected_cost(y_true, y_pred, classes, cfg["metrics"]["cost_matrix"]))
    if crit in present:
        out["recall_" + crit] = recall_score(y_true, y_pred, labels=[crit], average="macro", zero_division=0)
    return out


def breakdown(df: pd.DataFrame, y_pred, cfg: dict, by: str) -> pd.DataFrame:
    rows = []
    for k, g in df.assign(pred=y_pred).groupby(by):
        r = summarize(g["quality_class"].to_numpy(), g["pred"].to_numpy(), cfg)
        r[by] = k
        rows.append(r)
    return pd.DataFrame(rows).set_index(by).round(3)


def confusion(y_true, y_pred, classes):
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes)
