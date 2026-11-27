"""Figuras del informe y análisis de errores (fase 5).

Genera en `report/figures/` (SVG y PDF vectoriales; PNG solo para la galería de errores):
  fig1_f1_conjuntos      F1 macro por modelo en validación, test público y test externo
  fig2_recall_externo    recall de "mala" por especie en el externo (SVM frente a Random Forest)
  fig3_confusion         matrices de confusión del modelo por defecto (RF) en test (3 clases) y en el externo (2 clases)
  fig4_cnn_curvas        pérdida y exactitud por época de la CNN (sobreajuste: entrenamiento frente a validación)
  fig5_errores           ejemplos mal clasificados por el RF en test (recortes segmentados)
  fig6_especie           exactitud de especie: SVM (v2), MobileNetV2 congelada + cabeza (v2.1) y + racimos (v2.2)

No reentrena ni elige nada: lee las tablas de `evaluate.py` y los modelos congelados. Uso:
    python -m src.evaluation.error_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.training.data import has_quality, load_xy
from src.utils.config import load_config

TABLES = Path("experiments/results/tables")
CK = Path("experiments/checkpoints")
CLASSES = ["buena", "regular", "mala"]
# Paleta categórica validada (ranuras 1-3 de la paleta de referencia: azul, naranja, agua) + gris para la línea base.
C = {"azul": "#257de0", "naranja": "#f47230", "agua": "#00b894", "gris": "#8c8c8c", "texto": "#191919", "suave": "#666666"}
MODELOS = [("majority", "Mayoritaria"), ("logreg", "Reg. logística"), ("svm", "SVM"), ("rf", "Random Forest")]


def _estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.edgecolor": "#bbbbbb",
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.color": C["suave"],
        "ytick.color": C["suave"], "text.color": C["texto"], "axes.labelcolor": C["suave"],
        "svg.fonttype": "none", "pdf.fonttype": 42,        # texto editable en Word / incrustado en PDF
    })


def _guardar(fig, out: Path, nombre: str, raster=False):
    for ext in (["png"] if raster else ["svg", "pdf"]):
        fig.savefig(out / f"{nombre}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _f1_validacion() -> dict:
    """F1 macro en validación por modelo, de la corrida oficial v2 (experiments_log, sufijo _v2)."""
    log = pd.read_csv(TABLES / "experiments_log.csv")
    log = log[log["exp_id"].str.endswith("_v2")]
    return {r.exp_id.split("_")[1] if r.exp_id.startswith("quality") else "cnn": r.val_f1_macro
            for r in log.itertuples() if r.exp_id.startswith(("quality_", "cnn_"))}


def fig1(out):
    val = _f1_validacion()
    test = pd.read_csv(TABLES / "test_resultados.csv", index_col=0)["f1_macro"]
    ext = pd.read_csv(TABLES / "externo_resultados.csv", index_col=0)["f1_macro"]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(len(MODELOS)); w = 0.26
    series = [("Validación", [val[m] for m, _ in MODELOS], C["azul"]), ("Test público", [test[m] for m, _ in MODELOS], C["naranja"]),
              ("Test externo", [ext[m] for m, _ in MODELOS], C["agua"])]
    for k, (nombre, v, col) in enumerate(series):
        b = ax.bar(x + (k - 1) * w, v, w * 0.92, color=col, label=nombre)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 0.012, f"{vi:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=5.2, color=C["texto"])
    ax.set_xticks(x, [n for _, n in MODELOS]); ax.set_ylim(0, 1.05); ax.set_ylabel("F1 macro")
    ax.legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0, 1.13), handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="y", color="#e6e6e6", lw=0.5); ax.set_axisbelow(True)
    _guardar(fig, out, "fig1_f1_conjuntos")


def fig2(out):
    s = pd.read_csv(TABLES / "externo_svm_por_especie.csv").set_index("fruit_species")["recall_mala"]
    r = pd.read_csv(TABLES / "externo_rf_por_especie.csv").set_index("fruit_species")["recall_mala"]
    nombres = {"apple": "Manzana", "banana": "Banano", "guava": "Guayaba", "orange": "Naranja"}
    esp = list(s.index)
    fig, ax = plt.subplots(figsize=(3.5, 2.2)); x = np.arange(len(esp)); w = 0.36
    for k, (nombre, v, col) in enumerate([("SVM", s, C["azul"]), ("Random Forest", r, C["naranja"])]):
        ax.bar(x + (k - 0.5) * w, v.values, w * 0.92, color=col, label=nombre)
        for xi, vi in zip(x + (k - 0.5) * w, v.values):
            ax.text(xi, vi + 0.015, f"{vi:.2f}".replace(".", ","), ha="center", fontsize=5.8)
    ax.set_xticks(x, [nombres.get(e, e) for e in esp]); ax.set_ylim(0, 1.1); ax.set_ylabel('Recall de "mala"')
    ax.legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.12)); ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    _guardar(fig, out, "fig2_recall_externo")


def _matriz(ax, m: pd.DataFrame, titulo: str):
    v = m.values.astype(float); norm = v / np.maximum(v.sum(axis=1, keepdims=True), 1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, f"{int(v[i, j])}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.55 else C["texto"])
    ax.set_xticks(range(v.shape[1]), m.columns); ax.set_yticks(range(v.shape[0]), m.index)
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title(titulo)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(length=0)


DEFECTO = "rf"     # modelo de calidad por defecto (models/meta.json)


def _pred_defecto(cfg):
    df, X, Xs = load_xy(cfg)
    m = joblib.load(CK / f"quality_{DEFECTO}.joblib")
    lab = has_quality(df)
    return df, lab, m.predict(Xs)


def fig3(out, cfg):
    df, lab, p = _pred_defecto(cfg)
    mats = []
    for name in ("test", "external"):
        k = (df["split"] == name).to_numpy() & lab
        cl = [c for c in CLASSES if c in set(df.loc[k, "quality_class"])]      # filas: clases reales presentes
        mats.append(pd.crosstab(pd.Categorical(df.loc[k, "quality_class"], cl), pd.Categorical(p[k], CLASSES), dropna=False))
        mats[-1].index, mats[-1].columns = cl, CLASSES
    mats[0].to_csv(TABLES / "test_matriz_confusion_rf.csv"); mats[1].to_csv(TABLES / "externo_matriz_confusion_rf.csv")
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), gridspec_kw={"width_ratios": [3, 3]})
    _matriz(axs[0], mats[0], "RF, test"); _matriz(axs[1], mats[1], "RF, test externo")
    fig.tight_layout(w_pad=0.8)
    _guardar(fig, out, "fig3_confusion")


def fig4(out):
    h = json.load(open(CK / "cnn_v4_history.json")); ep = np.arange(1, len(h["loss"]) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.8))
    axs[0].plot(ep, h["loss"], color=C["azul"], lw=1.2, label="Entrenamiento")
    axs[0].plot(ep, h["val_loss"], color=C["naranja"], lw=1.2, label="Validación")
    axs[0].set_ylabel("Pérdida"); axs[0].set_xlabel("Época")
    axs[1].plot(ep, h["train_quality_acc"], color=C["azul"], lw=1.2); axs[1].plot(ep, h["val_quality_acc"], color=C["naranja"], lw=1.2)
    axs[1].set_ylabel("Exactitud (calidad)"); axs[1].set_xlabel("Época")
    for a in axs: a.grid(color="#e6e6e6", lw=0.5); a.set_axisbelow(True)
    axs[0].legend(frameon=False, loc="upper right")
    fig.tight_layout(w_pad=1.0)
    _guardar(fig, out, "fig4_cnn_curvas")


def fig5(out, cfg, n=8):
    from PIL import Image
    df, lab, p = _pred_defecto(cfg); te = (df["split"] == "test").to_numpy() & lab
    d = df[te].copy(); d["pred"] = p[te]
    err = d[d["pred"] != d["quality_class"]]
    # prioridad: la confusión más costosa (mala -> buena), después buena -> mala y las de "regular"
    grupos = [err[(err.quality_class == "mala") & (err.pred == "buena")], err[(err.quality_class == "buena") & (err.pred == "mala")],
              err[(err.quality_class == "regular") | (err.pred == "regular")]]
    elegidos = pd.concat([g.sample(min(len(g), 3 if i < 2 else 2), random_state=cfg["seed"]) for i, g in enumerate(grupos)]).head(n)
    fig, axs = plt.subplots(2, 4, figsize=(3.5, 2.0))
    for ax in axs.ravel(): ax.axis("off")
    for ax, r in zip(axs.ravel(), elegidos.itertuples()):
        im = Image.open(Path(cfg["paths"]["interim"]) / "segmented" / f"{r.image_id}.png").convert("RGBA")
        fondo = Image.new("RGBA", im.size, (128, 128, 128, 255)); fondo.alpha_composite(im)
        ax.imshow(fondo.convert("RGB")); ax.set_title(f"{r.quality_class} → {r.pred}", fontsize=5.5, pad=1.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.25)
    _guardar(fig, out, "fig5_errores", raster=True)
    elegidos[["image_id", "fruit_species", "quality_class", "pred"]].to_csv(TABLES / "errores_ejemplos.csv", index=False)
    print("errores del RF en test:", len(err), "de", len(d), "| mala->buena:", len(grupos[0]), "| buena->mala:", len(grupos[1]))


def fig6(out):
    """Exactitud de especie por versión: validación dejando una fuente fuera (media), test y externo."""
    v21 = pd.read_csv(TABLES / "validacion_entre_fuentes_v21.csv").query("tarea == 'especie'").set_index("candidato")["media"]
    v22 = pd.read_csv(TABLES / "validacion_entre_fuentes.csv").query("tarea == 'especie'").set_index("candidato")["media"]
    e = {k: pd.read_csv(TABLES / f"especie_resultados{k}.csv").set_index(["conjunto", "fuente"])["exactitud"] for k in ("", "_v21", "_v22")}
    grupos = ["Validación\nentre fuentes", "Test", "Externo\n(Sultana)"]
    series = [("SVM, 59 características (v2)", [v21["svm"], e[""][("test", "todas")], e[""][("external", "todas")]], C["azul"]),
              ("MobileNetV2 + cabeza (v2.1)", [v21["emb_head"], e["_v21"][("test", "todas")], e["_v21"][("external", "todas")]], C["naranja"]),
              ("+ racimos y pesos por clase (v2.2)", [v22["emb_head_cw"], e["_v22"][("test", "todas")], e["_v22"][("external", "todas")]], C["agua"])]
    fig, ax = plt.subplots(figsize=(3.5, 2.2)); x = np.arange(3); w = 0.27
    for k, (nombre, vals, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, vals, w * 0.92, color=col, label=nombre)
        for xi, vi in zip(x + (k - 1) * w, vals):
            ax.text(xi, vi + 0.015, f"{vi:.2f}".replace(".", ","), ha="center", fontsize=5.2)
    ax.set_xticks(x, grupos); ax.set_ylim(0, 1.12); ax.set_ylabel("Exactitud de especie")
    ax.legend(frameon=False, loc="upper left", ncol=1, bbox_to_anchor=(0, 1.3)); ax.grid(axis="y", color="#e6e6e6", lw=0.5)
    ax.set_axisbelow(True)
    _guardar(fig, out, "fig6_especie")


def make_figures(cfg: dict | None = None, out: str = "report/figures"):
    cfg = cfg or load_config(); _estilo(); o = Path(out); o.mkdir(parents=True, exist_ok=True)
    fig1(o); fig2(o); fig3(o, cfg); fig4(o); fig5(o, cfg); fig6(o)
    print("figuras en", o, sorted(p.name for p in o.iterdir()))


if __name__ == "__main__":
    make_figures()
