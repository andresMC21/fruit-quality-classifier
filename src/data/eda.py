"""Análisis exploratorio (fase 2): figuras vectoriales (SVG/PDF) y tablas en experiments/results/eda/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.helpers import save_vector_fig

OUT = Path("experiments/results/eda")


def run(cfg: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    ann = pd.read_csv(cfg["paths"]["annotations"])
    seg = pd.read_csv(Path(cfg["paths"]["interim"]) / "segmentation.csv")
    sp = pd.read_csv(cfg["split"]["output"])
    cls = cfg["classes"]

    # 1) conteo bruto: especie x calidad x fuente
    raw = ann.groupby(["fruit_species", "source", "quality_class"]).size().unstack(fill_value=0).reindex(columns=cls, fill_value=0)
    raw.to_csv(OUT / "conteos_brutos.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    piv = ann.groupby(["fruit_species", "quality_class"]).size().unstack(fill_value=0).reindex(columns=cls, fill_value=0)
    piv.plot.bar(stacked=True, ax=ax, color=["#4c9f70", "#e0a030", "#c0504d"], width=0.75)
    ax.set_ylabel("imágenes"); ax.set_xlabel(""); ax.set_title("Imágenes por especie y calidad (antes de deduplicar)")
    ax.legend(title="calidad", frameon=False); plt.xticks(rotation=0); save_vector_fig(fig, str(OUT / "01_conteos")); plt.close(fig)

    # 2) duplicados
    g = ann.groupby("dup_group").size()
    red = ann.groupby(["fruit_species", "source"]).agg(imagenes=("image_id", "size"), grupos=("dup_group", "nunique"))
    red["imagenes_por_grupo"] = (red["imagenes"] / red["grupos"]).round(2)
    red.to_csv(OUT / "duplicados.csv")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(g.clip(upper=30), bins=np.arange(0.5, 31.5, 1), color="#4472a8")
    ax.set_yscale("log"); ax.set_xlabel("imágenes por grupo de duplicados (recortado a 30)"); ax.set_ylabel("grupos (log)")
    ax.set_title(f"{len(g)} grupos para {len(ann)} imágenes; máximo {g.max()}")
    save_vector_fig(fig, str(OUT / "02_duplicados")); plt.close(fig)

    # 3) segmentación: tasa de fallo por especie y fuente
    d = ann[ann["keep"]].merge(seg, on="image_id")
    d["fallo"] = ~d["ok"]
    fr = d.groupby(["fruit_species", "source"]).agg(n=("ok", "size"), tasa_fallo=("fallo", "mean"),
                                                    otsu=("method", lambda s: (s == "otsu").mean()),
                                                    grabcut=("method", lambda s: (s == "grabcut").mean())).round(3)
    fr.to_csv(OUT / "segmentacion.csv")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    lab = [f"{a}\n{b}" for a, b in fr.index]
    ax.bar(lab, fr["otsu"], label="Otsu", color="#4c9f70"); ax.bar(lab, fr["grabcut"], bottom=fr["otsu"], label="GrabCut", color="#e0a030")
    ax.bar(lab, fr["tasa_fallo"], bottom=fr["otsu"] + fr["grabcut"], label="falla", color="#c0504d")
    ax.set_ylabel("fracción"); ax.set_title("Método de segmentación por especie y fuente"); ax.legend(frameon=False, ncol=3, loc="lower right")
    plt.xticks(fontsize=7); save_vector_fig(fig, str(OUT / "03_segmentacion")); plt.close(fig)

    # 4) tamaño relativo por fuente (por qué NO se mezclan fuentes para los umbrales)
    ok = d[d["ok"]]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    species = cfg["fruit_species"]; w = 0.38
    for j, srcn in enumerate(["afruitdb", "kaggle"]):
        data = [ok[(ok.fruit_species == s) & (ok.source == srcn)]["rel_diameter"].to_numpy() for s in species]
        pos = np.arange(len(species)) + (j - 0.5) * w
        bp = ax.boxplot([x if len(x) else [np.nan] for x in data], positions=pos, widths=w * 0.9, patch_artist=True, showfliers=False)
        for b in bp["boxes"]:
            b.set_facecolor(["#4472a8", "#e0a030"][j]); b.set_alpha(0.8)
    ax.set_xticks(range(len(species))); ax.set_xticklabels(species); ax.set_ylabel("diámetro equivalente / lado corto")
    ax.set_title("Tamaño relativo por fuente (azul: afruitdb; naranja: kaggle)")
    save_vector_fig(fig, str(OUT / "04_tamano_relativo")); plt.close(fig)

    # 5) tabla de splits
    sp.groupby(["split", "fruit_species", "quality_class"]).size().unstack(fill_value=0).to_csv(OUT / "splits.csv")
    # 6) atajo de fuente: clase vs fuente (¿se puede adivinar la calidad por la fuente?)
    src_q = ann[ann.keep].groupby(["source", "quality_class"]).size().unstack(fill_value=0).reindex(columns=cls, fill_value=0)
    src_q.to_csv(OUT / "fuente_vs_calidad.csv")
    return dict(raw=raw, dup=red, seg=fr, src_q=src_q)


if __name__ == "__main__":
    from src.utils.config import load_config
    r = run(load_config())
    for k, v in r.items():
        print("==", k); print(v.to_string())
