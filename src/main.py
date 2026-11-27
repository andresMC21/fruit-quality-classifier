"""Punto de entrada: python -m src.main {prepare|split|features|train|evaluate} [--model NOMBRE]"""
import argparse
import time

import pandas as pd

from src.utils.config import load_config, set_seed


def prepare(cfg, budget=150):
    """Manifiesto -> deduplicación -> segmentación/recortes (reanudable)."""
    from src.data.dedup import add_dup_groups, cross_source_dups, partition_units
    from src.data.manifest import write_manifest
    from src.data.preprocess import process_all
    t = time.time()
    deadline = t + budget
    df = write_manifest(cfg)
    df, hashes = add_dup_groups(df, cfg, deadline=deadline)
    # unidad de partición: grupo de duplicados; tramos de secuencia (FruitVision) o carpeta (Fruits-360)
    df["fruit_id"], df["dup_group"] = partition_units(df, hashes, cfg)
    # máx. N imágenes por grupo (al azar, con semilla): las ráfagas casi idénticas no aportan información nueva.
    # El sorteo se hace por bloques de fuentes (dedup.sample_blocks) para que añadir una fuente nueva no cambie
    # qué imágenes de las fuentes anteriores se conservan (la versión 1 se reproduce exactamente).
    cap = cfg["dedup"].get("max_per_group", 3)
    order = pd.Series(0, index=df.index)
    blocks = cfg["dedup"].get("sample_blocks") or [sorted(df["source"].unique())]
    for blk in blocks:
        b = df[df["source"].isin(blk)]
        order.loc[b.index] = b.sample(frac=1, random_state=cfg["seed"]).groupby("dup_group").cumcount()
    uncapped = ["own", "other_group", *cfg["dedup"].get("uncapped_sources", [])]
    df["keep"] = (order < cap) | df["source"].isin(uncapped)
    # fuentes nuevas: fuera las fotos casi idénticas a otra fuente (posible redistribución de la misma foto)
    nuevas = [s for blk in cfg["dedup"].get("sample_blocks", [[]])[1:] for s in blk]     # fuentes añadidas tras v1
    xdup = cross_source_dups(df, hashes, nuevas, cfg["dedup"]["hamming_threshold"])
    df.loc[xdup, "keep"] = False
    df.loc[xdup, "notes"] = "casi-duplicado de otra fuente"
    print("excluidas por duplicado entre fuentes:", int(xdup.sum()), flush=True)
    df.to_csv(cfg["paths"]["annotations"], index=False)
    print("imágenes:", len(df), "| grupos de duplicados:", df["dup_group"].nunique(), f"| {time.time()-t:.0f}s", flush=True)
    kept = df[df["keep"]]
    print(f"se conservan {len(kept)} de {len(df)} imágenes (máx. {cap} por grupo)", flush=True)
    res = process_all(kept, cfg, deadline=deadline)
    print(res.groupby("method").size().to_string())
    if res['image_id'].isin(kept['image_id']).sum() < len(kept):
        print("PENDIENTE: volver a ejecutar `prepare`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "split", "train", "export", "evaluate"])
    ap.add_argument("--model", default="rf", help="modelo de calidad por defecto")
    ap.add_argument("--species", default="tl", help="tl (MobileNetV2 congelada + cabeza) | svm")
    ap.add_argument("--budget", type=int, default=150, help="segundos máximos por ejecución de prepare")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    if args.command == "prepare":
        prepare(cfg, args.budget)
    elif args.command == "split":
        from src.data.split import make_splits, summarize
        print(summarize(make_splits(cfg)).to_string())
    elif args.command == "train":
        if args.model == "cnn":
            from src.training.train_cnn import train_cnn, finalize_cnn
            train_cnn(cfg, budget_s=args.budget); finalize_cnn(cfg)
        else:
            from src.training.train import train_classical
            train_classical(args.model, cfg)
            if args.model in ("svm", "rf", "logreg"):
                train_classical(args.model, cfg, task="species")
    elif args.command == "export":
        from src.training.export import export
        print(export(cfg, quality_model=args.model, species_model=args.species)["size_thresholds"])
    else:
        print("evaluate: ver src/evaluation/evaluate.py (se ejecuta una sola vez sobre el modelo congelado)")


if __name__ == "__main__":
    main()
