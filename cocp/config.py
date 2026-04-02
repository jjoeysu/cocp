# cocp/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


@dataclass
class ProjectConfig:
    out_dir: str = "results"
    experiment_name: str = "default"
    device: str = "cuda"
    force_retrain: bool = False
    cache_models: bool = True


@dataclass
class ReproConfig:
    base_seed: int = 0
    n_runs: int = 5
    deterministic: bool = True


@dataclass
class DataConfig:
    dataset_names: List[str] = field(default_factory=lambda: ["Synth_Normal"])
    base_path: str = "datasets"
    n_total: int = 20000
    dim: int = 1
    relevant_dim: int = 1
    x_range: Tuple[float, float] = (-2.0, 2.0)


@dataclass
class SplitsConfig:
    train: float = 0.5
    val: float = 0.1
    cal: float = 0.2
    test: float = 0.2
    shuffle: bool = True


@dataclass
class ConformalConfig:
    alpha: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 512
    max_epochs: int = 1000
    patience: int = 100
    grad_clip: float = 5.0
    cocp: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    n_mc: int = 1000
    save_predictions: bool = False
    save_models: bool = True


@dataclass
class PlotsConfig:
    enabled: bool = True
    n_show: int = 2500


@dataclass
class Config:
    project: ProjectConfig
    repro: ReproConfig
    data: DataConfig
    splits: SplitsConfig
    conformal: ConformalConfig
    training: TrainingConfig
    eval: EvalConfig
    plots: PlotsConfig


def _deep_update(base: dict, upd: dict) -> dict:
    out = dict(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str, overrides: Optional[Dict[str, Any]] = None) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _deep_update(raw, overrides or {})

    cfg = Config(
        project=ProjectConfig(**raw.get("project", {})),
        repro=ReproConfig(**raw.get("repro", {})),
        data=DataConfig(**raw.get("data", {})),
        splits=SplitsConfig(**raw.get("splits", {})),
        conformal=ConformalConfig(**raw.get("conformal", {})),
        training=TrainingConfig(**raw.get("training", {})),
        eval=EvalConfig(**raw.get("eval", {})),
        plots=PlotsConfig(**raw.get("plots", {})),
    )
    return cfg


def resolved_out_dir(cfg: Config) -> Path:
    return Path(cfg.project.out_dir) / cfg.project.experiment_name