"""Punto de entrada: python -m src.main {prepare|train|evaluate} [--model NOMBRE]"""
import argparse
from src.utils.config import load_config, set_seed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "train", "evaluate"])
    ap.add_argument("--model", default="svm")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(cfg["seed"])
    print(f"[{args.command}] model={args.model} (pendiente de implementar)")

if __name__ == "__main__":
    main()
