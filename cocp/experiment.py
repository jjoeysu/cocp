# cocp/experiment.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import gc
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .config import Config, resolved_out_dir
from .data import make_dataset, prepare_data_for_run
from .methods import CoCP, FitContext
from .metrics import evaluate_synthetic_intervals, summarize_cov_len, compute_real_metrics
from .plots import save_summary_metrics_bar_chart, save_synth_1d_plot, save_real_centered_plot
from .utils import ensure_dir, save_json, torch_save, torch_load, make_logger, set_seed


def _summarize_records(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    group_cols = ["Dataset", "Method"]
    metric_cols = [c for c in df.columns if c not in group_cols + ["Run"]]

    summary = (
        df.groupby(group_cols)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        f"{a}_{b}" if b else a
        for a, b in summary.columns
    ]
    return summary


def run_experiment(cfg: Config):
    out_root = ensure_dir(resolved_out_dir(cfg))
    logger = make_logger("cocp.experiment", out_root / "experiment.log")
    target_cov = 1.0 - float(cfg.conformal.alpha)

    with open(out_root / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)

    all_records = []

    for ds_name in cfg.data.dataset_names:
        logger.info(f"{'=' * 20} DATASET: {ds_name} {'=' * 20}")
        ds_out_dir = ensure_dir(out_root / ds_name)

        dataset, is_synthetic = make_dataset(ds_name, cfg.data)
        ds_records = []

        for run_id in range(int(cfg.repro.n_runs)):
            seed = int(cfg.repro.base_seed + run_id)
            set_seed(seed, deterministic=bool(cfg.repro.deterministic))

            run_dir = ensure_dir(ds_out_dir / f"run_{run_id:02d}")
            model_dir = ensure_dir(run_dir / "model_cache")
            fig_dir = ensure_dir(run_dir / "figs")

            data = prepare_data_for_run(cfg, dataset, seed, is_synthetic)
            ctx = FitContext(
                device=cfg.project.device,
                run_dir=str(run_dir),
                seed=seed,
                alpha=float(cfg.conformal.alpha),
            )

            method = CoCP()
            state_path = model_dir / "cocp.pt"

            if cfg.project.cache_models and state_path.exists() and not cfg.project.force_retrain:
                cached = torch_load(state_path, map_location="cpu")
                state = cached["state"]
                train_time = float(cached.get("train_time", 0.0))
                loaded_from_cache = True
                logger.info(f"[{ds_name}] Run {run_id}: load cached model")
            else:
                t0 = time.perf_counter()
                state = method.fit(
                    data["X_train"], data["y_train"],
                    data["X_val"], data["y_val"],
                    data["X_cal"], data["y_cal"],
                    ctx=ctx,
                    cfg=cfg,
                )
                train_time = time.perf_counter() - t0
                loaded_from_cache = False

                if cfg.eval.save_models or cfg.project.cache_models:
                    torch_save(state_path, {"state": state, "train_time": train_time})

            t0 = time.perf_counter()
            lo, hi = method.predict(data["X_test"], state=state, ctx=ctx)
            infer_time = time.perf_counter() - t0
            total_time = train_time + infer_time

            if cfg.eval.save_predictions:
                np.savez(
                    run_dir / "predictions.npz",
                    X=data["X_test"],
                    y=data["y_test"],
                    lo=lo,
                    hi=hi,
                    alpha=cfg.conformal.alpha,
                )

            if is_synthetic:
                cov_vec, len_vec = evaluate_synthetic_intervals(
                    dataset=dataset,
                    X_test=data["X_test"],
                    lo=lo,
                    hi=hi,
                    n_mc=int(cfg.eval.n_mc),
                    seed=seed + 2024,
                )
                stats = summarize_cov_len(cov_vec, len_vec, target_cov)

                record = {
                    "Dataset": ds_name,
                    "Method": "CoCP",
                    "Run": run_id,
                    "Coverage": float(stats["cov_mean"]),
                    "Length": float(stats["len_mean"]),
                    "ConMAE": float(stats["cov_mae_to_target"]),
                    "ConMSE": float(stats["cov_mse_to_target"]),
                    "Train_Time": float(train_time),
                    "Infer_Time": float(infer_time),
                    "Total_Time": float(total_time),
                    "Loaded_From_Cache": bool(loaded_from_cache),
                }

                if cfg.plots.enabled and dataset.supports_1d_plot() and run_id == 0:
                    oracle = dataset.oracle_interval(data["X_test"], alpha=float(cfg.conformal.alpha))
                    save_synth_1d_plot(
                        X_test=data["X_test"],
                        y_test=data["y_test"],
                        lo=lo,
                        hi=hi,
                        oracle=oracle,
                        out_path=fig_dir / "interval_plot.png",
                        title=f"{ds_name} - CoCP",
                        n_show=cfg.plots.n_show,
                    )

            else:
                metrics = compute_real_metrics(
                    X_test=data["X_test"],
                    y_test=data["y_test"],
                    lo=lo,
                    hi=hi,
                    alpha=float(cfg.conformal.alpha)
                )

                record = {
                    "Dataset": ds_name,
                    "Method": "CoCP",
                    "Run": run_id,
                    "Coverage": float(metrics["Coverage"]),
                    "Length": float(metrics["Length"]),
                    "MSCE": float(metrics["MSCE"]),
                    "WSC": float(metrics["WSC"]),
                    "l1ERT": float(metrics["l1ERT"]),
                    "l2ERT": float(metrics["l2ERT"]),
                    "Train_Time": float(train_time),
                    "Infer_Time": float(infer_time),
                    "Total_Time": float(total_time),
                    "Loaded_From_Cache": bool(loaded_from_cache),
                }

                if cfg.plots.enabled and run_id == 0:
                    save_real_centered_plot(
                        y_test=data["y_test"],
                        lo=lo,
                        hi=hi,
                        out_path=fig_dir / "centered_interval_plot.png",
                        title=f"{ds_name} - CoCP",
                        n_show=cfg.plots.n_show,
                    )

            save_json(run_dir / "result.json", record)
            ds_records.append(record)
            all_records.append(record)

            logger.info(
                f"[{ds_name}] Run {run_id} | "
                f"Cov={record['Coverage']:.4f}, Len={record['Length']:.4f}, "
                f"Train={record['Train_Time']:.2f}s, Infer={record['Infer_Time']:.2f}s"
            )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        df_ds = pd.DataFrame(ds_records)
        df_ds.to_csv(ds_out_dir / "all_runs_detailed.csv", index=False)

        ds_summary = _summarize_records(df_ds)
        ds_summary.to_csv(ds_out_dir / "final_metrics_summary.csv", index=False)

        if cfg.plots.enabled:
            save_summary_metrics_bar_chart(
                df_summary=ds_summary,
                target_cov=target_cov,
                out_path=ds_out_dir / "summary_metrics_comparison.png",
                dataset_name=ds_name,
            )

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(out_root / "all_runs_detailed.csv", index=False)

    global_summary = _summarize_records(df_all)
    global_summary.to_csv(out_root / "all_datasets_summary_metrics.csv", index=False)

    logger.info(f"All done. Results saved to: {out_root}")
    return df_all