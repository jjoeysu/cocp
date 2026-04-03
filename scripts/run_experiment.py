# scripts/run_experiment.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
import sys
import argparse
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cocp.config import load_config
from cocp.experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()