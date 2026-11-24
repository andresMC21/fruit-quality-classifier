"""Fase 2/3: construye data/annotations/annotations.csv para las bases públicas de la especie elegida.

Fuentes:
- kaggle   : Fruit Quality Classification (ryandpark). Good -> buena, Bad -> mala.
- afruitdb : Fruit grade (AFruitDB / Asian Fruits Dataset). 1st -> buena, 2nd -> regular, 3rd -> mala.
- mixed_quality (Kaggle): varias frutas por foto; se segmentan y se anotan a mano (no entra aquí).
- own / other_group: se agregan con `add_own_images` (fotos propias y de otros grupos).

La etiqueta de calidad de las bases públicas es una etiqueta DÉBIL: viene de la carpeta, no de los criterios
de docs/guia_anotacion.md. Por eso se conserva `orig_label` y se revisa una muestra (ver notebooks/02_eda.ipynb).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

COLUMNS = ["image_id", "filename", "source", "group_id", "fruit_species", "fruit_id", "parent_image",
           "quality_class", "size_class", "diameter_mm", "annotator", "annotator2_quality",
           "capture_device", "lighting", "background", "date", "notes", "orig_label", "timestamp"]

IMG_EXT = {".jpg", ".jpeg", ".png"}
_TS = re.compile(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})")

# (carpeta -> clase de calidad) por fuente; 'regular' solo existe en AFruitDB, mixed_quality y fotos propias.
KAGGLE_MAP = {"Good": "buena", "Bad": "mala"}
AFRUITDB_MAP = {"1st grade": "buena", "2nd grade": "regular", "3rd grade": "mala"}
# especie -> (carpeta en Kaggle, carpeta en AFruitDB); None = la base no la incluye.
# Pomegranate/Papayas/Tomatoes/Burmese_grape existen en las bases pero no están en la lista de frutas del curso.
SPECIES_DIR = {"apple": ("Apple", "Apple"), "banana": ("Banana", "Banana"), "mango": (None, "Mango"),
               "guava": ("Guava", None), "orange": ("Orange", None), "lime": ("Lime", None)}


def parse_timestamp(name: str):
    """Marca de tiempo embebida en el nombre del archivo (IMG20200728175856 / IMG_20230930_172415)."""
    m = _TS.search(name)
    if not m:
        return pd.NaT
    y, mo, d, h, mi, s = map(int, m.groups())
    try:
        return pd.Timestamp(y, mo, d, h, mi, s)
    except ValueError:
        return pd.NaT


def _files(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.is_dir() else []


# Sultana et al. (Mendeley, 2022): carpetas FreshX / RottenX; solo fresca/podrida (sin "regular").
SULTANA_SPECIES = {"Apple": "apple", "Banana": "banana", "Orange": "orange", "Guava": "guava",
                   "Grape": "grape", "Strawberry": "strawberry"}   # Jujube y Pomegranate fuera de la lista del curso
SULTANA_MAP = {"Fresh": "buena", "Rotten": "mala"}


def build_public_manifest(cfg: dict, root: str = ".") -> pd.DataFrame:
    root = Path(root)
    rows = []
    sroot = root / cfg["paths"].get("raw_sultana", "Sultana")
    for name, sp in SULTANA_SPECIES.items():
        if sp not in cfg["fruit_species"]:
            continue
        for prefix, qc in SULTANA_MAP.items():
            for p in _files(sroot / f"{prefix}{name}"):
                rows.append(dict(source="sultana", fruit_species=sp, quality_class=qc, orig_label=prefix, path=p))
    for sp in cfg["fruit_species"]:
        if sp not in SPECIES_DIR:
            continue
        kname, aname = SPECIES_DIR[sp]
        kroot = root / cfg["paths"]["raw_kaggle"]
        for qdir, label in (("Good Quality_Fruits", "Good"), ("Bad Quality_Fruits", "Bad")):
            for p in (_files(kroot / qdir / f"{kname}_{label}") if kname else []):
                rows.append(dict(source="kaggle", fruit_species=sp, quality_class=KAGGLE_MAP[label],
                                 orig_label=f"{label}", path=p))
        aroot = root / cfg["paths"]["raw_afruitdb"] / (aname or "_")
        for grade, qc in AFRUITDB_MAP.items():
            for p in (_files(aroot / grade) if aname else []):
                rows.append(dict(source="afruitdb", fruit_species=sp, quality_class=qc,
                                 orig_label=grade, path=p))
    df = pd.DataFrame(rows)
    df["filename"] = [p.relative_to(root).as_posix() for p in df.pop("path")]
    df["timestamp"] = [parse_timestamp(Path(f).name) for f in df["filename"]]
    # ID estable: no cambia al agregar especies (se numera dentro de cada fuente+especie)
    df["image_id"] = [f"{s[:3].upper()}_{sp[:3]}_{i:05d}" for s, sp, i in
                      zip(df["source"], df["fruit_species"], df.groupby(["source", "fruit_species"]).cumcount())]
    df["group_id"] = "public"
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df["fruit_id"] = ""            # se rellena con el grupo de duplicados (dedup.py)
    df["annotator"] = "folder_label"
    return df[COLUMNS]


def write_manifest(cfg: dict, root: str = ".") -> pd.DataFrame:
    df = build_public_manifest(cfg, root)
    out = Path(root) / cfg["paths"]["annotations"]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
