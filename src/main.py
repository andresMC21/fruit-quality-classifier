"""Punto de entrada: python -m src.main {prepare|split|features|train|evaluate} [--model NOMBRE]"""
import argparse
import time

import pandas as pd

from src.utils.config import load_config, set_seed


def prepare(cfg, budget=150):
    """Manifiesto -> deduplicación -> segmentación/recortes (reanudable)."""
    from src.data.dedup import add_dup_groups
    from src.data.manifest import write_manifest
    from src.data.preprocess import process_all
    t = time.time()
    deadline = t + budget
    df = write_manifest(cfg)
    df, _ = add_dup_groups(df, cfg, deadline=deadline)
    df["fruit_id"] = df["dup_group"]          # en bases públicas la unidad de partición es el grupo de duplicados
    # máx. N imágenes por grupo (al azar, con semilla): las ráfagas casi idénticas no aportan información nueva
    cap = cfg["dedup"].get("max_per_group", 3)
    order = df.sample(frac=1, random_state=cfg["seed"]).groupby("dup_group").cumcount()
    df["keep"] = (order.reindex(df.index) < cap) | df["source"].isin(["own", "other_group"])
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
    ap.add_argument("--model", default="svm")
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
        print(export(cfg, quality_model=args.model, species_model=args.model)["size_thresholds"])
    else:
        print("evaluate: ver src/evaluation/evaluate.py (se ejecuta una sola vez sobre el modelo congelado)")


if __name__ == "__main__":
    main()
