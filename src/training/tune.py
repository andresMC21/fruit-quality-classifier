"""Búsqueda de hiperparámetros: GridSearchCV con StratifiedGroupKFold (los grupos de duplicados no se parten)."""
from __future__ import annotations

import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.models.classical import GRIDS, build_model


def tune(model_name: str, X, y, groups, strat, cfg: dict, n_splits: int = 4):
    """Devuelve (mejores parámetros, f1_macro CV, tabla de resultados) sobre train."""
    cv = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=cfg["seed"]).split(X, strat, groups))
    est = build_model(model_name, cfg["seed"], probability=False)       # sin probabilidades: más rápido
    gs = GridSearchCV(est, GRIDS[model_name], scoring="f1_macro", cv=cv, n_jobs=1, refit=False)
    t = time.time()
    gs.fit(X, y)
    res = pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    best = {k.replace("clf__", "clf__"): v for k, v in gs.best_params_.items()}
    return best, float(gs.best_score_), res.sort_values("rank_test_score"), time.time() - t
