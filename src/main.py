"""Punto de entrada: python -m src.main {prepare|train|evaluate} [--model NOMBRE]

prepare   consolida las etiquetas, genera splits.csv (si no existe) y la tabla de caracteristicas
train     entrena un modelo: baseline | logreg | svm | rf | cnn  (clasicos: --features masked | full)
evaluate  pendiente: evaluacion UNICA en test (src/evaluation/evaluate.py)
"""
import argparse

from src.utils.config import load_config, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--model", default="svm")
    ap.add_argument("--features", choices=["masked", "full"], default="masked",
                    help="clasicos: fruta segmentada (masked) o imagen completa (full, experimento de contraste)")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--force-split", action="store_true", help="regenera splits.csv (solo antes de mirar el test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    if args.command == "prepare":
        from src.data.build_features import build_feature_table
        from src.data.consolidate import consolidate
        from src.data.split import write_splits
        from pathlib import Path
        consolidate(cfg)
        if Path(cfg["split"]["output"]).exists() and not args.force_split:
            # El split es fijo; las etiquetas nuevas no entran hasta regenerarlo (solo antes de mirar test).
            print(f"Aviso: {cfg['split']['output']} ya existe y no se regenera; se usa el actual.")
        else:
            write_splits(cfg, force=True)
        build_feature_table(cfg)
    elif args.command == "train":
        from src.training.train import train
        train(args.model, cfg, args.features)
    else:
        print("[evaluate] pendiente de implementar")


if __name__ == "__main__":
    main()
