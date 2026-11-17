"""Cola de re-etiquetado de las manzanas de Kaggle/FruitNet.

Kaggle solo trae buena/mala por carpeta y esa etiqueta no siempre es confiable (ver
docs/02_comprension_datos.md), asi que un subconjunto se re-etiqueta con la guia de anotacion.
Pasos:
1. Agrupa casi-duplicados por carpeta con dedup.compute_groups. El grupo hace de fruit_id.
2. Representante de cada grupo = primera imagen en orden alfabetico (solo se etiqueta esa).
3. Toma todos los grupos de Apple_Bad y una muestra de grupos de Apple_Good (semilla del config).
4. Separa una muestra comun (assigned_to="kappa") que etiquetan dos personas sin verse, para
   la kappa de Cohen, y reparte el resto entre los anotadores manteniendo la proporcion buena/mala.

Salidas:
    data/annotations/kaggle_groups.csv   todas las imagenes con su fruit_id
    data/annotations/label_queue.csv     representantes a etiquetar

Uso:
    python -m src.data.build_label_set
"""
import argparse
import csv
import random
from pathlib import Path

from src.data.dedup import compute_groups
from src.data.fruitnet_inventory import IMAGE_EXTENSIONS
from src.utils.config import load_config

FOLDERS = {"Apple_Good": "good", "Apple_Bad": "bad"}
ANNOTATORS = ["andresMC21", "CarlosamaSantiago", "Javi-19", "Pauli55"]


def collect_images(root):
    """Rutas relativas a root de las imagenes de cada carpeta de manzana, ordenadas."""
    root = Path(root)
    images = {}
    for folder in FOLDERS:
        images[folder] = sorted(
            p.relative_to(root).as_posix() for p in root.rglob("*")
            if p.parent.name.lower() == folder.lower() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images[folder]:
            raise SystemExit(f"No hay imagenes de {folder} en {root}. Ejecutar antes fruitnet_inventory --extract-apple.")
    return images


def build(root, n_good, n_kappa, threshold, seed):
    images = collect_images(root)
    rng = random.Random(seed)
    group_rows, queue = [], []

    for folder, rel_paths in images.items():
        groups_by_path = compute_groups([Path(root) / r for r in rel_paths], threshold)
        groups = {}
        for rel in rel_paths:
            fruit_id = f"kaggle_{FOLDERS[folder]}_{groups_by_path[Path(root) / rel]:04d}"
            groups.setdefault(fruit_id, []).append(rel)
            group_rows.append({"filename": rel, "folder": folder, "fruit_id": fruit_id})

        fruit_ids = sorted(groups)
        if folder == "Apple_Good" and n_good < len(fruit_ids):
            fruit_ids = sorted(rng.sample(fruit_ids, n_good))
        for fruit_id in fruit_ids:
            members = groups[fruit_id]
            queue.append({
                "filename": members[0],  # members ya viene ordenado
                "folder": folder,
                "fruit_id": fruit_id,
                "group_size": len(members),
            })

    return group_rows, assign(queue, n_kappa, rng)


def assign(queue, n_kappa, rng):
    """Reparte la cola manteniendo la proporcion buena/mala en la muestra kappa y en cada lote.

    Se baraja cada carpeta por separado y se intercalan segun su posicion relativa (i + 0.5) / n.
    Las primeras n_kappa filas van a la muestra comun y el resto se reparte en round-robin.
    """
    by_folder = {}
    for row in queue:
        by_folder.setdefault(row["folder"], []).append(row)
    for rows in by_folder.values():
        rng.shuffle(rows)
    positioned = [((i + 0.5) / len(rows), row["folder"], row)
                  for rows in by_folder.values() for i, row in enumerate(rows)]
    ordered = [row for _, _, row in sorted(positioned, key=lambda t: (t[0], t[1]))]
    for i, row in enumerate(ordered):
        row["assigned_to"] = "kappa" if i < n_kappa else ANNOTATORS[(i - n_kappa) % len(ANNOTATORS)]
    return ordered


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Construye la cola de re-etiquetado de Kaggle.")
    parser.add_argument("--root", default=cfg["paths"]["raw_kaggle"])
    parser.add_argument("--n-good", type=int, default=400)
    parser.add_argument("--n-kappa", type=int, default=100)
    args = parser.parse_args()

    group_rows, queue = build(args.root, args.n_good, args.n_kappa,
                              cfg["dedup"]["hamming_threshold"], cfg["seed"])
    write_csv("data/annotations/kaggle_groups.csv", group_rows)
    write_csv("data/annotations/label_queue.csv", queue)
    print(f"Imagenes agrupadas: {len(group_rows)}; representantes en cola: {len(queue)}")
    for name in ANNOTATORS + ["kappa"]:
        print(f"  {name}: {sum(r['assigned_to'] == name for r in queue)}")


if __name__ == "__main__":
    main()
