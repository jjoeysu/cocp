# cocp/sensitivity.py
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import copy
import gc
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .config import Config
from .data import make_dataset, prepare_data_for_run
from .methods import CoCP, FitContext
from .metrics import evaluate_synthetic_intervals, summarize_cov_len, compute_real_metrics
from .plots import save_sensitivity_lineplots
from .utils import ensure_dir, make_logger, set_seed, torch_save, torch_load, save_json


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            a, b = c
            cols.append(f"{a}_{b}" if b else a)
        else:
            cols.append(c)
    df.columns = cols
    return df


def _make_summary(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    group_cols = ["Experiment", "Dataset", "Study", "Param_Name", "Param_Value", "Beta", "K", "T"]
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in group_cols and c != "Run"
    ]

    summary = (
        df.groupby(group_cols, dropna=False)[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    return _flatten_multiindex_columns(summary)


def _build_variant_cfg(base_cfg: Config, beta=None, k=None, t=None) -> Config:
    cfg_i = copy.deepcopy(base_cfg)
    if beta is not None:
        cfg_i.training.cocp["beta_start"] = float(beta)
        cfg_i.training.cocp["beta_end"] = float(beta)
    if k is not None:
        cfg_i.training.cocp["n_folds"] = int(k)
    if t is not None:
        cfg_i.training.cocp["n_alt_iters"] = int(t)
    return cfg_i


def _cache_key(dataset: str, run_id: int, seed: int, beta: float, k: int, t: int) -> str:
    payload = {
        "dataset": str(dataset),
        "run_id": int(run_id),
        "seed": int(seed),
        "beta": float(beta),
        "k": int(k),
        "t": int(t),
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _cache_paths(run_dir: Path, key: str):
    cache_dir = ensure_dir(run_dir / "fit_cache")
    return cache_dir / f"{key}.pt"


def _fit_with_cache(
    method: CoCP,
    run_dir: Path,
    dataset_name: str,
    run_id: int,
    seed: int,
    beta: float,
    k: int,
    t: int,
    cfg_variant: Config,
    data: dict,
    ctx: FitContext,
    logger,
):
    key = _cache_key(dataset_name, run_id, seed, beta, k, t)
    path = _cache_paths(run_dir, key)

    if path.exists() and cfg_variant.project.cache_models and not cfg_variant.project.force_retrain:
        cached = torch_load(path, map_location="cpu")
        logger.info(f"[CACHE HIT] dataset={dataset_name} run={run_id} beta={beta} K={k} T={t}")
        return cached["state"], float(cached["train_time"]), True

    logger.info(f"[CACHE MISS] dataset={dataset_name} run={run_id} beta={beta} K={k} T={t}")

    t0 = time.perf_counter()
    state = method.fit(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_cal"], data["y_cal"],
        ctx=ctx,
        cfg=cfg_variant,
    )
    train_time = time.perf_counter() - t0

    torch_save(path, {"state": state, "train_time": train_time})
    return state, train_time, False


def _evaluate_setting(cfg_variant: Config, dataset, is_synthetic: bool, data: dict, method: CoCP, state, train_time: float, ctx: FitContext, loaded_from_cache: bool):
    t0 = time.perf_counter()
    lo, hi = method.predict(data["X_test"], state=state, ctx=ctx)
    infer_time = time.perf_counter() - t0
    total_time = train_time + infer_time

    if is_synthetic:
        cov_vec, len_vec = evaluate_synthetic_intervals(
            dataset=dataset,
            X_test=data["X_test"],
            lo=lo,
            hi=hi,
            n_mc=int(cfg_variant.eval.n_mc),
            seed=ctx.seed + 2024,
        )
        stats = summarize_cov_len(cov_vec, len_vec, target=1.0 - float(cfg_variant.conformal.alpha))
        record = {
            "Coverage": float(stats["cov_mean"]),
            "Length": float(stats["len_mean"]),
            "ConMAE": float(stats["cov_mae_to_target"]),
            "ConMSE": float(stats["cov_mse_to_target"]),
        }
    else:
        metrics = compute_real_metrics(
            X_test=data["X_test"],
            y_test=data["y_test"],
            lo=lo,
            hi=hi,
            alpha=float(cfg_variant.conformal.alpha),
        )
        record = {
            "Coverage": float(metrics["Coverage"]),
            "Length": float(metrics["Length"]),
            "MSCE": float(metrics["MSCE"]),
            "WSC": float(metrics["WSC"]),
            "l1ERT": float(metrics["l1ERT"]),
            "l2ERT": float(metrics["l2ERT"]),
        }

    record.update({
        "Method": "CoCP",
        "Train_Time": float(train_time),
        "Infer_Time": float(infer_time),
        "Total_Time": float(total_time),
        "Loaded_From_Cache": bool(loaded_from_cache),
    })
    return record


def run_cocp_sensitivity(
    cfg: Config,
    config_path: str | None = None,
    beta_values=None,
    k_values=None,
    t_values=None,
    out_subdir: str = "sensitivity",
):
    exp_name = cfg.project.experiment_name
    out_root = ensure_dir(Path(cfg.project.out_dir) / out_subdir / exp_name)
    logger = make_logger(f"cocp.sensitivity.{exp_name}", out_root / "sensitivity.log")

    baseline_beta = float(cfg.training.cocp.get("beta_start", 0.005))
    baseline_k = int(cfg.training.cocp.get("n_folds", 4))
    baseline_t = int(cfg.training.cocp.get("n_alt_iters", 5))

    if beta_values is None:
        beta_values = [0.002, 0.005, 0.01, 0.02, 0.05]
    if k_values is None:
        k_values = [2, 3, 4, 5]
    if t_values is None:
        t_values = list(range(0, baseline_t + 1))

    beta_values = sorted(set(float(x) for x in beta_values))
    k_values = sorted(set(int(x) for x in k_values))
    t_values = sorted(set(int(x) for x in t_values))

    with open(out_root / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)

    save_json(
        out_root / "study_spec.json",
        {
            "config_path": config_path,
            "experiment_name": exp_name,
            "baseline": {
                "beta": baseline_beta,
                "K": baseline_k,
                "T": baseline_t,
            },
            "studies": {
                "beta_values": beta_values,
                "k_values": k_values,
                "t_values": t_values,
            },
        },
    )

    all_records = []
    target_cov = 1.0 - float(cfg.conformal.alpha)

    for ds_name in cfg.data.dataset_names:
        logger.info(f"{'=' * 20} DATASET: {ds_name} {'=' * 20}")
        ds_out_dir = ensure_dir(out_root / ds_name)
        dataset, is_synthetic = make_dataset(ds_name, cfg.data)
        ds_records = []

        for run_id in range(int(cfg.repro.n_runs)):
            seed = int(cfg.repro.base_seed + run_id)
            set_seed(seed, deterministic=bool(cfg.repro.deterministic))

            run_dir = ensure_dir(ds_out_dir / f"run_{run_id:02d}")
            data = prepare_data_for_run(cfg, dataset, seed, is_synthetic)

            # beta study
            for beta in beta_values:
                cfg_i = _build_variant_cfg(cfg, beta=beta, k=baseline_k, t=baseline_t)
                ctx = FitContext(device=cfg_i.project.device, run_dir=str(run_dir), seed=seed, alpha=float(cfg_i.conformal.alpha))
                method = CoCP()

                state, train_time, loaded = _fit_with_cache(
                    method=method,
                    run_dir=run_dir,
                    dataset_name=ds_name,
                    run_id=run_id,
                    seed=seed,
                    beta=float(beta),
                    k=int(baseline_k),
                    t=int(baseline_t),
                    cfg_variant=cfg_i,
                    data=data,
                    ctx=ctx,
                    logger=logger,
                )

                record = _evaluate_setting(cfg_i, dataset, is_synthetic, data, method, state, train_time, ctx, loaded)
                record.update({
                    "Experiment": exp_name,
                    "Dataset": ds_name,
                    "Run": run_id,
                    "Study": "beta",
                    "Param_Name": "beta",
                    "Param_Value": float(beta),
                    "Beta": float(beta),
                    "K": int(baseline_k),
                    "T": int(baseline_t),
                })
                ds_records.append(record)
                all_records.append(record)

            # K study
            for k in k_values:
                cfg_i = _build_variant_cfg(cfg, beta=baseline_beta, k=k, t=baseline_t)
                ctx = FitContext(device=cfg_i.project.device, run_dir=str(run_dir), seed=seed, alpha=float(cfg_i.conformal.alpha))
                method = CoCP()

                state, train_time, loaded = _fit_with_cache(
                    method=method,
                    run_dir=run_dir,
                    dataset_name=ds_name,
                    run_id=run_id,
                    seed=seed,
                    beta=float(baseline_beta),
                    k=int(k),
                    t=int(baseline_t),
                    cfg_variant=cfg_i,
                    data=data,
                    ctx=ctx,
                    logger=logger,
                )

                record = _evaluate_setting(cfg_i, dataset, is_synthetic, data, method, state, train_time, ctx, loaded)
                record.update({
                    "Experiment": exp_name,
                    "Dataset": ds_name,
                    "Run": run_id,
                    "Study": "K",
                    "Param_Name": "K",
                    "Param_Value": int(k),
                    "Beta": float(baseline_beta),
                    "K": int(k),
                    "T": int(baseline_t),
                })
                ds_records.append(record)
                all_records.append(record)

            # T study
            for t in t_values:
                cfg_i = _build_variant_cfg(cfg, beta=baseline_beta, k=baseline_k, t=t)
                ctx = FitContext(device=cfg_i.project.device, run_dir=str(run_dir), seed=seed, alpha=float(cfg_i.conformal.alpha))
                method = CoCP()

                state, train_time, loaded = _fit_with_cache(
                    method=method,
                    run_dir=run_dir,
                    dataset_name=ds_name,
                    run_id=run_id,
                    seed=seed,
                    beta=float(baseline_beta),
                    k=int(baseline_k),
                    t=int(t),
                    cfg_variant=cfg_i,
                    data=data,
                    ctx=ctx,
                    logger=logger,
                )

                record = _evaluate_setting(cfg_i, dataset, is_synthetic, data, method, state, train_time, ctx, loaded)
                record.update({
                    "Experiment": exp_name,
                    "Dataset": ds_name,
                    "Run": run_id,
                    "Study": "T",
                    "Param_Name": "T",
                    "Param_Value": int(t),
                    "Beta": float(baseline_beta),
                    "K": int(baseline_k),
                    "T": int(t),
                })
                ds_records.append(record)
                all_records.append(record)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        df_ds = pd.DataFrame(ds_records)
        df_ds.to_csv(ds_out_dir / "raw_records.csv", index=False)

        summary_all = _make_summary(df_ds)
        summary_all.to_csv(ds_out_dir / "summary_all.csv", index=False)

        for study in ["beta", "K", "T"]:
            df_study = df_ds[df_ds["Study"] == study].copy()
            if len(df_study) == 0:
                continue

            summary_study = _make_summary(df_study)
            summary_study.to_csv(ds_out_dir / f"summary_{study}.csv", index=False)

            save_sensitivity_lineplots(
                df_summary=summary_study,
                study=study,
                out_path=ds_out_dir / f"{study}_sensitivity.png",
                dataset_name=ds_name,
                target_cov=target_cov,
            )

    df_all = pd.DataFrame(all_records)
    df_all.to_csv(out_root / "all_runs.csv", index=False)

    summary_all = _make_summary(df_all)
    summary_all.to_csv(out_root / "summary_all.csv", index=False)

    for study in ["beta", "K", "T"]:
        df_study = df_all[df_all["Study"] == study].copy()
        if len(df_study) == 0:
            continue
        summary_study = _make_summary(df_study)
        summary_study.to_csv(out_root / f"summary_{study}.csv", index=False)

    logger.info(f"CoCP sensitivity finished. Results saved to: {out_root}")
    return df_all