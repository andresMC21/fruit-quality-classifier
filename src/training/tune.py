"""Búsqueda de hiperparámetros con StratifiedGroupKFold (los grupos de partición no se parten).

Reanudable: el F1 macro de CV de cada combinación se guarda en experiments/checkpoints/tune_<tarea>_<modelo>.json
apenas termina; si se pasa `deadline` y se agota, lanza TimeoutError y basta con repetir el comando.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold

from src.models.classical import GRIDS, build_model


def tune(model_name: str, X, y, groups, strat, cfg: dict, n_splits: int = 4, task: str = "quality",
         deadline: float | None = None):
    """Devuelve (mejores parámetros, f1_macro CV, tabla de resultados, segundos) sobre train."""
    cv = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=cfg["seed"]).split(X, strat, groups))
    est = build_model(model_name, cfg["seed"], probability=False)       # sin probabilidades: más rápido
    cache = Path("experiments/checkpoints") / f"tune_{task}_{model_name}.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    y = np.asarray(y)
    t = time.time()
    for params in ParameterGrid(GRIDS[model_name]):
        key = json.dumps(params, sort_keys=True)
        if key in done:
            continue
        if deadline is not None and time.time() > deadline:
            raise TimeoutError(f"tuning {task}/{model_name}: {len(done)} combinaciones listas; repetir")
        sc = []
        for tr, va in cv:
            m = clone(est).set_params(**params).fit(X[tr], y[tr])
            sc.append(f1_score(y[va], m.predict(X[va]), average="macro"))
        done[key] = [float(np.mean(sc)), float(np.std(sc))]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(done, indent=1))
    res = pd.DataFrame([dict(params=json.loads(k), mean_test_score=v[0], std_test_score=v[1]) for k, v in done.items()])
    res["rank_test_score"] = res["mean_test_score"].rank(ascending=False, method="min").astype(int)
    res = res.sort_values("rank_test_score")
    best = res.iloc[0]
    return best["params"], float(best["mean_test_score"]), res, time.time() - t
