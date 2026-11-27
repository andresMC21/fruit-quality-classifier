"""Métricas en TEST. Se ejecuta UNA sola vez sobre los modelos congelados (candado en test_done.json).

- accuracy, F1 macro, recall de "mala", costo esperado (matriz de costos de config.yaml)
- desglose por especie y por fuente (detecta atajos y cambio de dominio)
- comparación frente a la línea base y entre modelos: prueba de McNemar + IC 95 % bootstrap del F1 macro
- tamaño: MAE en mm solo con fotos propias que tengan diameter_mm
- especie: exactitud por fuente, rechazo de frutas fuera del alcance (clase other y umbral de confianza) y calidad
  con la especie PREDICHA (como en la app), no solo con la verdadera
Usa las mismas características que el entrenamiento (src/training/data.py).
Versión 2 (+ FruitVision, + Fruits-360 para especie): candado propio; los resultados de la versión 1 están en
experiments/results/v1/.
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
from src.training.data import has_quality, load_xy

OUT = Path("experiments/results")
LOCK = OUT / "test_done_v2.json"
MIN_SPECIES_CONF = 0.60          # igual que src/inference/predict.py


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
    lab = has_quality(df)
    te = (df["split"] == "test").to_numpy() & lab
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
    ex = (df["split"] == "external").to_numpy() & lab
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
    out["especie"] = evaluate_species(cfg, df, X, Xs, models, ck)
    LOCK.write_text(json.dumps({"models": list(models), "best": best, "n_test": int(te.sum())}, indent=2))
    return out


def _onehot(species_pred, species):
    return np.stack([(np.asarray(species_pred) == s).astype(np.float32) for s in species], axis=1)


def species_tl_proba(cfg: dict, df: pd.DataFrame, rows: np.ndarray, path: str = "models/species_tl.keras"):
    """Probabilidades del clasificador de especie con transfer learning sobre los RECORTES (como la app)."""
    import cv2
    from src.inference.predict import KerasSpecies
    sm = KerasSpecies(Path(path))
    seg = Path(cfg["paths"]["interim"]) / "segmented"
    proba = np.zeros((len(df), len(sm.classes_)), np.float32)
    idx = np.where(rows)[0]
    for k in range(0, len(idx), 256):
        part = idx[k:k + 256]
        proba[part] = sm.predict_proba([cv2.imread(str(seg / f"{i}.png"), cv2.IMREAD_COLOR) for i in df["image_id"].to_numpy()[part]])
    return proba, np.array(sm.classes_)


def evaluate_species(cfg: dict, df: pd.DataFrame, X, Xs, models, ck: Path, species_model: str = "svm", suffix: str = ""):
    """Especie en test y externo + rechazo de `other` + calidad con la especie predicha (pipeline de la app)."""
    if species_model == "tl":
        proba, cls = species_tl_proba(cfg, df, df["split"].isin(["test", "external"]).to_numpy())
    else:
        sm = joblib.load(ck / f"species_{species_model}.joblib")
        proba, cls = sm.predict_proba(X), np.array(sm.classes_)
    pred, conf = cls[proba.argmax(1)], proba.max(1)
    rejected = (pred == "other") | (conf < MIN_SPECIES_CONF)
    rows = []
    for name in ("test", "external"):
        m = (df["split"] == name).to_numpy()
        tgt = m & (df["fruit_species"] != "other").to_numpy()
        oth = m & (df["fruit_species"] == "other").to_numpy()
        for src in ["todas", *sorted(df.loc[tgt, "source"].unique())]:
            k = tgt if src == "todas" else tgt & (df["source"] == src).to_numpy()
            rows.append(dict(conjunto=name, fuente=src, n=int(k.sum()),
                             exactitud=float((pred[k] == df["fruit_species"].to_numpy()[k]).mean()),
                             rechazo_falso=float(rejected[k].mean())))
        if oth.any():
            rows.append(dict(conjunto=name, fuente="fruits360 (other)", n=int(oth.sum()), exactitud=float("nan"),
                             rechazo_correcto=float(rejected[oth].mean())))
        tgt_all = m & (df["fruit_species"] != "-").to_numpy()            # incluye `other` como fila
        pd.crosstab(df["fruit_species"].to_numpy()[tgt_all], np.where(rejected, "rechazada", pred)[tgt_all],
                    rownames=["real"], colnames=["predicha"]).to_csv(OUT / f"tables/{name}_especie_confusion{suffix}.csv")
    tab = pd.DataFrame(rows).round(4)
    tab.to_csv(OUT / f"tables/especie_resultados{suffix}.csv", index=False)
    # calidad con la especie predicha (one-hot de la predicción; `other` -> ceros), solo filas con calidad
    lab = has_quality(df)
    nq = X.shape[1]
    Xp = np.hstack([X, _onehot(pred, cfg["fruit_species"])])
    prow = []
    for name in ("test", "external"):
        m = (df["split"] == name).to_numpy() & lab
        y = df.loc[m, "quality_class"].to_numpy()
        for mod in models:
            q = joblib.load(ck / f"quality_{mod}.joblib")
            r = summarize(y, q.predict(Xp[m]), cfg); r.update(conjunto=name, model=mod, rechazadas=float(rejected[m].mean()))
            prow.append(r)
    ptab = pd.DataFrame(prow).round(4)
    ptab.to_csv(OUT / f"tables/pipeline_especie_predicha{suffix}.csv", index=False)
    return dict(especie=tab, pipeline=ptab)


def evaluate_species_v21(cfg: dict, models=("majority", "logreg", "svm", "rf")):
    """Versión 2.1: solo cambia el clasificador de especie (MobileNetV2 congelada + cabeza propia), elegido con
    validación dejando una fuente fuera. Una sola evaluación (candado propio); los modelos de calidad son los de v2."""
    lock = OUT / "test_done_v2_1.json"
    if lock.exists():
        raise RuntimeError(f"La especie v2.1 ya se evaluó ({lock}).")
    df, X, Xs = load_xy(cfg)
    res = evaluate_species(cfg, df, X, Xs, models, Path("experiments/checkpoints"), species_model="tl", suffix="_v21")
    lock.write_text(json.dumps({"species_model": "models/species_tl.keras", "quality_models": list(models)}, indent=2))
    return res


def evaluate_species_version(cfg: dict, version: str = "v22", models=("majority", "logreg", "svm", "rf")):
    """Versión 2.2 en adelante: mismo procedimiento que v2.1 con el clasificador de especie vigente en models/.
    v2.2 = + WGISD (racimos) + más mango de Fruits-360 + pesos por clase. Una sola evaluación por versión."""
    lock = OUT / f"test_done_{version}.json"
    if lock.exists():
        raise RuntimeError(f"La especie {version} ya se evaluó ({lock}).")
    df, X, Xs = load_xy(cfg)
    res = evaluate_species(cfg, df, X, Xs, models, Path("experiments/checkpoints"), species_model="tl", suffix=f"_{version}")
    lock.write_text(json.dumps({"species_model": "models/species_tl.keras", "quality_models": list(models)}, indent=2))
    return res
