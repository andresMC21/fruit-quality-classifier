"""Consolida las etiquetas de Kaggle de cada anotador en data/annotations/annotations.csv.

- "Mi lote": la etiqueta del anotador. Las imagenes excluidas no pasan.
- Muestra kappa: dos anotadores. Si coinciden, esa es la etiqueta y la segunda va en annotator2_quality.
  Si no coinciden, se usa la decision de data/annotations/kappa_resolutions.csv (filename, quality_class);
  mientras no exista, la fila queda sin clase y split.py la deja fuera.
- Kappa de Cohen sobre la muestra comun, contando "excluir" como una categoria mas.
Las filas de otras fuentes (own, mixed_quality, other_group) que ya esten en annotations.csv se conservan.

Uso:
    python -m src.data.consolidate
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.utils.config import load_config

COLUMNS = ["image_id", "filename", "source", "group_id", "fruit_id", "fruit_species", "parent_image",
           "quality_class", "size_class", "diameter_mm", "annotator", "annotator2_quality",
           "capture_device", "lighting", "background", "hand", "notes"]


def read_labels(labels_dir):
    files = sorted(Path(labels_dir).glob("*.csv"))
    if not files:
        raise SystemExit(f"No hay etiquetas en {labels_dir}.")
    labels = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files])
    # Si un CSV se re-exporto o alguien etiqueto dos veces, vale la ultima etiqueta de cada anotador.
    labels = labels.drop_duplicates(["filename", "annotator", "batch"], keep="last")
    labels["label"] = labels.apply(lambda r: "excluir" if r["exclude"] == "1" else r["quality_class"], axis=1)
    return labels


def kappa_sample(labels):
    """Tabla filename -> (anotador A, etiqueta A, anotador B, etiqueta B) de la muestra comun.

    Solo entran las imagenes etiquetadas por exactamente dos anotadores distintos.
    """
    k = labels[labels["batch"] == "kappa"].sort_values(["filename", "annotator"])
    k = k.groupby("filename").filter(lambda g: g["annotator"].nunique() == 2)
    groups = k.groupby("filename")
    return groups.first().merge(groups.last(), left_index=True, right_index=True, suffixes=("_1", "_2"))


def consolidate(cfg, resolutions_csv="data/annotations/kappa_resolutions.csv"):
    labels = read_labels(cfg["paths"]["kaggle_labels"])
    queue = pd.read_csv("data/annotations/label_queue.csv", dtype=str)
    fruit_of = dict(zip(queue["filename"], queue["fruit_id"]))
    resolved = {}
    if Path(resolutions_csv).exists():
        r = pd.read_csv(resolutions_csv, dtype=str)
        resolved = dict(zip(r["filename"], r["quality_class"]))

    rows = []
    for _, r in labels[labels["batch"] == "own"].iterrows():
        if r["label"] != "excluir":
            rows.append(dict(filename=r["filename"], quality_class=r["label"], annotator=r["annotator"],
                             annotator2_quality="", background=r["background"], hand=r["hand"],
                             notes="duda" if r["doubt"] == "1" else ""))

    pairs = kappa_sample(labels)
    n_open = 0
    for filename, p in pairs.iterrows():
        if p["label_1"] == p["label_2"]:
            final = p["label_1"]
        else:
            final = resolved.get(filename, "")
            n_open += final == ""
        if final == "excluir":
            continue
        rows.append(dict(filename=filename, quality_class=final, annotator=p["annotator_1"],
                         annotator2_quality=p["label_2"], background=p["background_1"], hand=p["hand_1"],
                         notes="" if p["label_1"] == p["label_2"] else "desacuerdo kappa"))

    new = pd.DataFrame(rows, columns=["filename", "quality_class", "annotator", "annotator2_quality",
                                      "background", "hand", "notes"])
    if new["filename"].duplicated().any():
        raise ValueError(f"Imagenes etiquetadas en dos lotes: {new.loc[new['filename'].duplicated(), 'filename'].tolist()[:5]}")
    new["fruit_id"] = new["filename"].map(fruit_of)
    if new["fruit_id"].isna().any():
        raise ValueError(f"Imagenes que no estan en label_queue.csv: {new.loc[new['fruit_id'].isna(), 'filename'].tolist()[:5]}")
    new["image_id"] = new["fruit_id"] + "_" + new["filename"].map(lambda f: Path(f).stem)
    new["filename"] = "kaggle/" + new["filename"]       # data/README.md: rutas relativas a data/raw/
    new["source"], new["fruit_species"] = "kaggle", "apple"

    out = Path(cfg["paths"]["annotations"])
    old = pd.read_csv(out, dtype=str, keep_default_na=False) if out.exists() else pd.DataFrame(columns=COLUMNS)
    old = old[old["source"] != "kaggle"]
    merged = pd.concat([old, new], ignore_index=True).reindex(columns=COLUMNS).astype(object).fillna("")
    merged.to_csv(out, index=False)

    kappa = cohen_kappa_score(pairs["label_1"], pairs["label_2"]) if len(pairs) else float("nan")
    print(f"Kaggle: {len(new)} filas; muestra kappa: {len(pairs)} pares, kappa = {kappa:.3f}, "
          f"desacuerdos sin resolver: {n_open}")
    return merged, kappa


if __name__ == "__main__":
    consolidate(load_config())
