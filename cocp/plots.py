# cocp/plots.py
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_summary_metrics_bar_chart(df_summary: pd.DataFrame, target_cov: float, out_path: Path, dataset_name: str):
    if len(df_summary) == 0:
        return

    metric_order = [
        "Coverage", "Length",
        "ConMAE", "ConMSE",
        "MSCE", "WSC", "l1ERT", "l2ERT",
        "Train_Time", "Infer_Time", "Total_Time",
    ]
    metrics = [m for m in metric_order if f"{m}_mean" in df_summary.columns]
    if len(metrics) == 0:
        return

    methods = df_summary["Method"].tolist()
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.2 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(methods))]

    label_map = {
        "Coverage": "Coverage",
        "Length": "Length",
        "ConMAE": "ConMAE",
        "ConMSE": "ConMSE",
        "MSCE": "MSCE",
        "WSC": "WSC",
        "l1ERT": "l1ERT",
        "l2ERT": "l2ERT",
        "Train_Time": "Train Time (s)",
        "Infer_Time": "Infer Time (s)",
        "Total_Time": "Total Time (s)",
    }

    for ax, metric in zip(axes, metrics):
        means = df_summary[f"{metric}_mean"].values
        stds = df_summary[f"{metric}_std"].values if f"{metric}_std" in df_summary.columns else np.zeros_like(means)

        x = np.arange(len(methods))
        ax.bar(x, means, yerr=stds, color=colors, edgecolor="black", capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=25, ha="right")
        ax.set_title(label_map.get(metric, metric))
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if metric == "Coverage":
            ax.axhline(target_cov, color="crimson", linestyle="--", linewidth=1.5)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(f"{dataset_name}: summary metrics (mean ± std)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_synth_1d_plot(X_test, y_test, lo, hi, out_path: Path, title: str, n_show: int = 2500, oracle=None):
    x = np.asarray(X_test).reshape(-1)
    y = np.asarray(y_test).reshape(-1)
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)

    order = np.argsort(x)
    order = order[:min(n_show, len(order))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(x[order], y[order], s=8, alpha=0.2, color="gray", label="test samples")
    ax.plot(x[order], lo[order], color="#d62728", lw=2, label="CoCP lower")
    ax.plot(x[order], hi[order], color="#1f77b4", lw=2, label="CoCP upper")
    ax.fill_between(x[order], lo[order], hi[order], color="#9ecae1", alpha=0.25)

    if oracle is not None:
        o_lo, o_hi = oracle
        o_lo = np.asarray(o_lo).reshape(-1)
        o_hi = np.asarray(o_hi).reshape(-1)
        ax.plot(x[order], o_lo[order], color="black", ls="--", lw=1.5, label="oracle lower")
        ax.plot(x[order], o_hi[order], color="black", ls="--", lw=1.5, label="oracle upper")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_real_centered_plot(y_test, lo, hi, out_path: Path, title: str, n_show: int = 1000):
    y_test = np.asarray(y_test).reshape(-1)
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)

    n_total = len(y_test)
    n_show = min(n_show, n_total)
    idx = np.random.choice(n_total, n_show, replace=False)

    y_sub = y_test[idx]
    lo_sub = lo[idx]
    hi_sub = hi[idx]

    order = np.argsort(y_sub)
    rel_lo = lo_sub[order] - y_sub[order]
    rel_hi = hi_sub[order] - y_sub[order]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(np.arange(n_show), rel_lo, rel_hi, color="skyblue", alpha=0.35)
    ax.plot(rel_lo, color="tab:blue", alpha=0.4)
    ax.plot(rel_hi, color="tab:blue", alpha=0.4)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1.5)

    ylim_val = np.percentile(np.abs(np.concatenate([rel_lo, rel_hi])), 98) * 1.5
    ylim_val = max(ylim_val, 1e-3)
    ax.set_ylim(-ylim_val, ylim_val)

    ax.set_title(title)
    ax.set_xlabel("samples sorted by y")
    ax.set_ylabel("boundary - y")
    ax.grid(True, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_sensitivity_lineplots(df_summary: pd.DataFrame, study: str, out_path: Path, dataset_name: str, target_cov: float):
    if len(df_summary) == 0:
        return

    df = df_summary.copy()
    df["Param_Value"] = pd.to_numeric(df["Param_Value"], errors="coerce")
    df = df.sort_values("Param_Value").reset_index(drop=True)

    synth_order = ["Coverage", "Length", "ConMAE", "ConMSE", "Train_Time", "Infer_Time", "Total_Time"]
    real_order = ["Coverage", "Length", "MSCE", "WSC", "l1ERT", "l2ERT", "Train_Time", "Infer_Time", "Total_Time"]

    if "ConMAE_mean" in df.columns:
        metric_order = synth_order
    else:
        metric_order = real_order

    metrics = [m for m in metric_order if f"{m}_mean" in df.columns]
    if len(metrics) == 0:
        return

    n_cols = 3 if len(metrics) >= 3 else len(metrics)
    n_rows = math.ceil(len(metrics) / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).reshape(-1)
    x = df["Param_Value"].values.astype(float)

    for ax, metric in zip(axes, metrics):
        y = df[f"{metric}_mean"].values.astype(float)
        y_std = df[f"{metric}_std"].values.astype(float) if f"{metric}_std" in df.columns else None

        ax.plot(x, y, marker="o", color="#d62728", lw=2)
        if y_std is not None:
            ax.fill_between(x, y - y_std, y + y_std, color="#d62728", alpha=0.15)

        if metric == "Coverage":
            ax.axhline(target_cov, color="crimson", linestyle="--", linewidth=1.4)

        ax.set_title(metric)
        ax.set_xlabel(study)
        ax.grid(True, alpha=0.25)

    for ax in axes[len(metrics):]:
        ax.axis("off")

    fig.suptitle(f"{dataset_name}: {study} sensitivity", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)