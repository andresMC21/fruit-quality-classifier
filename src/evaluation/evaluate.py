"""Métricas en TEST. Se ejecuta UNA sola vez sobre los modelos congelados (candado en test_done.json).

- accuracy, F1 macro, recall de "mala", costo esperado (matriz de costos de config.yaml)
- desglose por especie y por fuente (detecta atajos y cambio de dominio)
- comparación frente a la línea base y entre modelos: prueba de McNemar + IC 95 % bootstrap del F1 macro
- tamaño: MAE en mm solo con fotos propias que tengan diameter_mm
Usa las mismas características que el entrenamiento (src/training/data.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from statsmodels.stats.contingency_tables import mcnemar

from src.evaluation.metrics import breakdown, confusion, summarize
from src.training.data import load_xy

OUT = Path("experiments/results")
LOCK = OUT / "test_done.json"


def bootstrap_f1(y, p, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    y, p = np.asarray(y), np.asarray(p)
    labels = sorted(set(y))                      # mismas clases que summarize(): solo las presentes en la verdad
    v = [f1_score(y[i], p[i], labels=labels, average="macro", zero_division=0) for i in (rng.randint(0, len(y), len(y)) for _ in range(n))]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def evaluate(cfg: dict, models=("majority", "logreg", "svm", "rf"), force: bool = False):
    if LOCK.exists() and not force:
        raise RuntimeError(f"El test ya se evaluó ({LOCK}). No se repite para no sobreajustar a él.")
    df, X, Xs = load_xy(cfg)
    te = (df["split"] == "test").to_numpy()
    y = df.loc[te, "quality_class"].to_numpy()
    ck = Path("experiments/checkpoints")
    preds, rows = {}, []
    for m in models:
        preds[m] = joblib.load(ck / f"quality_{m}.joblib").predict(Xs[te])
        r = summarize(y, preds[m], cfg); lo, hi = bootstrap_f1(y, preds[m]); r.update(model=m, f1_ic95=f"[{lo:.3f}, {hi:.3f}]")
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("model").round(4)
    OUT.joinpath("tables").mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / "tables/test_resultados.csv")
    out = {"resumen": tab}
    best = tab["f1_macro"].idxmax()
    out["mejor"] = best
    for m in models:
        breakdown(df[te], preds[m], cfg, "fruit_species").to_csv(OUT / f"tables/test_{m}_por_especie.csv")
        breakdown(df[te], preds[m], cfg, "source").to_csv(OUT / f"tables/test_{m}_por_fuente.csv")
    confusion(y, preds[best], cfg["classes"]).to_csv(OUT / "tables/test_matriz_confusion_mejor.csv")
    # McNemar: mejor modelo vs cada uno (aciertos pareados)
    mc = []
    for m in models:
        if m == best:
            continue
        a, b = preds[best] == y, preds[m] == y
        t = [[(a & b).sum(), (a & ~b).sum()], [(~a & b).sum(), (~a & ~b).sum()]]
        mc.append(dict(vs=m, only_best_correct=int(t[0][1]), only_other_correct=int(t[1][0]), p_value=float(mcnemar(t, exact=False, correction=True).pvalue)))
    pd.DataFrame(mc).to_csv(OUT / "tables/test_mcnemar.csv", index=False)
    out["mcnemar"] = pd.DataFrame(mc)
    # conjunto EXTERNO (otra fuente, nunca visto): ¿generaliza fuera de Kaggle/Fruit grade?
    ex = (df["split"] == "external").to_numpy()
    if ex.any():
        ye = df.loc[ex, "quality_class"].to_numpy(); erows = []
        for m in models:
            pe = joblib.load(ck / f"quality_{m}.joblib").predict(Xs[ex])
            r = summarize(ye, pe, cfg); lo, hi = bootstrap_f1(ye, pe); r.update(model=m, f1_ic95=f"[{lo:.3f}, {hi:.3f}]"); erows.append(r)
            breakdown(df[ex], pe, cfg, "fruit_species").to_csv(OUT / f"tables/externo_{m}_por_especie.csv")
            if m == best:
                confusion(ye, pe, [c for c in cfg["classes"] if c in set(ye)]).to_csv(OUT / "tables/externo_matriz_confusion_mejor.csv")
        ext_tab = pd.DataFrame(erows).set_index("model").round(4)
        ext_tab.to_csv(OUT / "tables/externo_resultados.csv"); out["externo"] = ext_tab
    # tamaño (si hay fotos propias con diámetro medido)
    own = df[te & df["source"].isin(["own", "other_group"]).to_numpy() & df["diameter_mm"].notna().to_numpy()]
    out["size_mae_mm"] = None if own.empty else float((own["d_eq_px"] * own["scale"] * own["mm_per_px"]).sub(own["diameter_mm"]).abs().mean()) if "mm_per_px" in own else None
    LOCK.write_text(json.dumps({"models": list(models), "best": best, "n_test": int(te.sum())}, indent=2))
    return out
