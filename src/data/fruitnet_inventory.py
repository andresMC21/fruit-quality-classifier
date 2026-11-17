"""Inventario del zip de FruitNet (mismo contenido que el dataset de Kaggle) y extraccion de manzanas.

Lee el indice del zip sin descomprimirlo y cuenta imagenes por carpeta. Con --extract-apple extrae
solo las manzanas: buena/mala a data/raw/kaggle/ (conservando la ruta del zip) y la carpeta de
calidad mixta a data/raw/mixed_quality/Apple/.

Uso:
    python -m src.data.fruitnet_inventory data/raw/kaggle/FruitNetDataset.zip
    python -m src.data.fruitnet_inventory data/raw/kaggle/FruitNetDataset.zip --extract-apple
"""
import argparse
import collections
import csv
import zipfile
from pathlib import Path, PurePosixPath

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(zip_path):
    """Entradas del zip que son imagenes, como rutas POSIX."""
    with zipfile.ZipFile(zip_path) as zf:
        return [
            PurePosixPath(info.filename)
            for info in zf.infolist()
            if not info.is_dir() and PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]


def count_by_folder(paths):
    return collections.Counter(str(p.parent) for p in paths)


def is_apple_folder(name):
    """True para 'Apple', 'Apple_Good' o 'Apple_Bad'; False para p. ej. 'Pineapple'."""
    name = name.lower()
    return name == "apple" or name.startswith("apple_")


def target_path(p, kaggle_dir, mixed_dir):
    """Destino de una entrada del zip: la carpeta Mixed va aparte, como en data/README.md."""
    if p.parts[0].lower().startswith("mixed"):
        return mixed_dir.joinpath(*p.parts[1:])
    return kaggle_dir / p


def extract_apple(zip_path, kaggle_dir, mixed_dir):
    """Extrae las imagenes de manzana y devuelve cuantas extrajo."""
    kaggle_dir, mixed_dir = Path(kaggle_dir).resolve(), Path(mixed_dir).resolve()
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            p = PurePosixPath(info.filename)
            if info.is_dir() or p.suffix.lower() not in IMAGE_EXTENSIONS or not is_apple_folder(p.parent.name):
                continue
            target = target_path(p, kaggle_dir, mixed_dir).resolve()
            # Un zip con rutas tipo "../x.jpg" podria escribir fuera de data/raw.
            if kaggle_dir not in target.parents and mixed_dir not in target.parents:
                raise ValueError(f"Ruta insegura en el zip: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description="Inventario del zip de FruitNet.")
    parser.add_argument("zip_path")
    parser.add_argument("--csv", default="data/annotations/kaggle_inventory.csv")
    parser.add_argument("--extract-apple", action="store_true")
    parser.add_argument("--kaggle-dir", default="data/raw/kaggle")
    parser.add_argument("--mixed-dir", default="data/raw/mixed_quality")
    args = parser.parse_args()

    paths = list_images(args.zip_path)
    counts = count_by_folder(paths)
    print(f"Imagenes totales: {len(paths)}")
    for folder, n in sorted(counts.items()):
        print(f"{n:6d}  {folder}")

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["folder", "n_images"])
        writer.writerows(sorted(counts.items()))
    print(f"Inventario guardado en {args.csv}")

    if args.extract_apple:
        n = extract_apple(args.zip_path, args.kaggle_dir, args.mixed_dir)
        print(f"Imagenes de manzana extraidas: {n}")


if __name__ == "__main__":
    main()
