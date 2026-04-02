# scripts/plot_sensitivity.py

from pathlib import Path
import sys
import argparse
import yaml
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cocp.plots import save_sensitivity_lineplots


def load_target_cov(exp_dir: Path, default=0.9):
    cfg_path = exp_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        return default
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    alpha = cfg.get("conformal", {}).get("alpha", 0.1)
    return 1.0 - float(alpha)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=str, required=True)
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    target_cov = load_target_cov(exp_dir)

    for ds_dir in sorted(exp_dir.iterdir()):
        if not ds_dir.is_dir():
            continue

        for study in ["beta", "K", "T"]:
            csv_path = ds_dir / f"summary_{study}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            out_dir = ds_dir / "plots"
            out_dir.mkdir(parents=True, exist_ok=True)

            save_sensitivity_lineplots(
                df_summary=df,
                study=study,
                out_path=out_dir / f"{study}_sensitivity.png",
                dataset_name=ds_dir.name,
                target_cov=target_cov,
            )
            print(f"[Done] {ds_dir.name} / {study}")


if __name__ == "__main__":
    main()