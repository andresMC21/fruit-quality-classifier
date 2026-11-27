"""Exporta el modelo elegido (por F1 macro en VALIDACIÓN) a models/ junto con sus metadatos.

meta.json: clases, especies, tamaño de recorte y umbrales de tamaño (terciles de train, por especie).
Los umbrales relativos salen de la fuente más parecida al protocolo de captura (afruitdb > kaggle > fruitvision > sultana) porque la
distancia a la cámara cambia entre fuentes; los umbrales en mm se calculan solo con fotos propias.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from src.models.size_estimator import terciles
from src.training.data import load_table

SOURCE_PREF = ["own", "afruitdb", "kaggle", "fruitvision", "sultana"]


def export(cfg: dict, quality_model: str = "rf", species_model: str = "tl", out: str = "models",
           extra_quality=("svm", "logreg")):
    """Exporta el modelo de calidad principal + alternativos (selector de la app) y el modelo de especie."""
    import joblib
    out_d = Path(out); out_d.mkdir(exist_ok=True)
    ck = Path("experiments/checkpoints")
    if species_model == "tl":                     # ya guardado en models/ por src/training/train_species_tl.py
        sf = "species_tl.keras"
    else:
        sf = f"species_{species_model}.joblib"
        joblib.dump(joblib.load(ck / sf), out_d / sf, compress=3)
    qmodels = {}
    for name in (quality_model, *extra_quality):
        f = f"quality_{name}.joblib"
        if (ck / f).exists() and name not in qmodels:
            joblib.dump(joblib.load(ck / f), out_d / f, compress=3)
            qmodels[name] = f
    qf = qmodels[quality_model]
    df = load_table(cfg)
    tr = df[df["split"] == "train"]
    th = {}
    for sp in cfg["fruit_species"]:
        d = tr[tr["fruit_species"] == sp]
        for src in SOURCE_PREF:
            g = d[d["source"] == src]
            if len(g) >= 30:
                th[sp] = {"relative": terciles(g["rel_diameter"]), "source": src, "n": int(len(g))}
                if src == "own" and g["diameter_mm"].notna().sum() >= 15:
                    th[sp]["mm"] = terciles(g["diameter_mm"].dropna())
                break
    meta = dict(species=cfg["fruit_species"], classes=cfg["classes"], image_size=cfg["image_size"],
                species_model=sf, species_fallback="species_svm.joblib", quality_model=qf, quality_models=qmodels, default_quality=quality_model, size_thresholds=th,
                note="Prototipo académico; umbrales relativos dependen de la fuente de datos. v2.2: especie con MobileNetV2 congelada + cabeza propia (racimos de WGISD, pesos por clase); calidad por defecto RF (mejor F1 en validación dejando una fuente fuera).")
    (out_d / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
