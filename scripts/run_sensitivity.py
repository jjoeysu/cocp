# scripts/run_sensitivity.py

from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cocp.config import load_config
from cocp.sensitivity import run_cocp_sensitivity


def _parse_float_list(text):
    if text is None or text.strip() == "":
        return None
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_int_list(text):
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["configs/synth1d.yaml", "configs/real.yaml"],
    )
    parser.add_argument(
        "--beta-values",
        type=str,
        default="0.002,0.005,0.01,0.02,0.05",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="2,3,4,5",
    )
    parser.add_argument(
        "--t-values",
        type=str,
        default="0,1,2,3,4,5",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="sensitivity",
    )
    args = parser.parse_args()

    beta_values = _parse_float_list(args.beta_values)
    k_values = _parse_int_list(args.k_values)
    t_values = _parse_int_list(args.t_values)

    for config_path in args.configs:
        cfg = load_config(config_path)
        run_cocp_sensitivity(
            cfg=cfg,
            config_path=config_path,
            beta_values=beta_values,
            k_values=k_values,
            t_values=t_values,
            out_subdir=args.out_subdir,
        )


if __name__ == "__main__":
    main()