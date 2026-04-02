# cocp/data.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import StandardScaler

from .config import Config


# =========================================================
# Basic containers
# =========================================================

@dataclass
class DatasetBundle:
    X: np.ndarray
    y: np.ndarray
    meta: Dict


@dataclass
class SplitIdx:
    train: np.ndarray
    val: np.ndarray
    cal: np.ndarray
    test: np.ndarray


class BaseDataset:
    name: str = "BaseDataset"

    def generate_or_load(self, seed: int) -> DatasetBundle:
        raise NotImplementedError

    def supports_1d_plot(self) -> bool:
        return False

    def true_conditional_sampler(self):
        return None

    def oracle_interval(self, x: np.ndarray, alpha: float):
        return None

    def get_cdf(self, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError


# =========================================================
# Split
# =========================================================

def make_splits(n: int, train: float, val: float, cal: float, test: float, seed: int, shuffle: bool = True) -> SplitIdx:
    s = float(train) + float(val) + float(cal) + float(test)
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {s}")

    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    n_test = int(round(test * n))
    test_idx = idx[:n_test]
    rest = idx[n_test:]

    rem = 1.0 - test
    train_r = train / rem
    val_r = val / rem

    n_rest = len(rest)
    n_train = int(round(train_r * n_rest))
    n_val = int(round(val_r * n_rest))
    n_cal = n_rest - n_train - n_val

    train_idx = rest[:n_train]
    val_idx = rest[n_train:n_train + n_val]
    cal_idx = rest[n_train + n_val:n_train + n_val + n_cal]

    return SplitIdx(train=train_idx, val=val_idx, cal=cal_idx, test=test_idx)


# =========================================================
# Synthetic datasets
# =========================================================

def m_func(x_eff: np.ndarray) -> np.ndarray:
    return (0.5 * np.sin(1.5 * x_eff)).astype(np.float32)


def s_func(x_eff: np.ndarray) -> np.ndarray:
    return (0.15 + 0.25 * (x_eff ** 2)).astype(np.float32)


def find_hdi_generic(dist, confidence=0.90) -> Tuple[float, float]:
    def interval_width(low_tail):
        low_val = dist.ppf(low_tail)
        high_val = dist.ppf(low_tail + confidence)
        return high_val - low_val

    res = minimize_scalar(interval_width, bounds=(0, 1 - confidence), method="bounded")
    return float(dist.ppf(res.x)), float(dist.ppf(res.x + confidence))


class BaseSynthDataset(BaseDataset):
    def __init__(self, n_total: int = 20000, dim: int = 1, relevant_dim: int = 1, x_range=(-2.0, 2.0)):
        self.n_total = int(n_total)
        self.dim = int(dim)
        self.relevant_dim = min(int(relevant_dim), self.dim)
        self.x_range = tuple(x_range)

        beta = np.zeros(self.dim, dtype=np.float32)
        beta[:self.relevant_dim] = 1.0
        self.beta = beta / np.linalg.norm(beta)

    def _get_x_eff(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        return (x @ self.beta).astype(np.float32).ravel()

    def generate_or_load(self, seed: int) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        X = rng.uniform(self.x_range[0], self.x_range[1], size=(self.n_total, self.dim)).astype(np.float32)
        y = self.true_conditional_sampler()(X, seed=seed + 1).reshape(-1, 1).astype(np.float32)
        return DatasetBundle(X=X, y=y, meta={"dim": self.dim, "relevant_dim": self.relevant_dim})

    def supports_1d_plot(self) -> bool:
        return self.dim == 1

    def get_oracle_mode(self, x: np.ndarray) -> np.ndarray:
        return m_func(self._get_x_eff(x)).reshape(-1, 1)

    def get_dist(self, x_eff: np.ndarray):
        raise NotImplementedError

    def true_conditional_sampler(self):
        raise NotImplementedError

    def oracle_interval(self, x: np.ndarray, alpha: float):
        raise NotImplementedError

    def get_cdf(self, x: np.ndarray, y: np.ndarray):
        x_eff = self._get_x_eff(x)
        y_flat = np.asarray(y).reshape(-1)
        return self.get_dist(x_eff).cdf(y_flat).reshape(-1, 1)


class SynthNormalDataset(BaseSynthDataset):
    name = "Synth_Normal"

    def get_dist(self, x_eff):
        return stats.norm(loc=m_func(x_eff), scale=s_func(x_eff))

    def true_conditional_sampler(self):
        def _sampler(x, seed):
            x_eff = self._get_x_eff(x)
            rng = np.random.default_rng(seed)
            return (m_func(x_eff) + rng.normal(0, s_func(x_eff), size=len(x_eff))).astype(np.float32)
        return _sampler

    def oracle_interval(self, x, alpha):
        x_eff = self._get_x_eff(x)
        z_lo, z_hi = stats.norm.interval(1.0 - alpha)
        lo = m_func(x_eff) + z_lo * s_func(x_eff)
        hi = m_func(x_eff) + z_hi * s_func(x_eff)
        return lo.reshape(-1, 1), hi.reshape(-1, 1)


class SynthExponentialDataset(BaseSynthDataset):
    name = "Synth_Exponential"

    def get_dist(self, x_eff):
        return stats.expon(loc=m_func(x_eff), scale=s_func(x_eff))

    def true_conditional_sampler(self):
        def _sampler(x, seed):
            x_eff = self._get_x_eff(x)
            rng = np.random.default_rng(seed)
            return (m_func(x_eff) + rng.exponential(s_func(x_eff), size=len(x_eff))).astype(np.float32)
        return _sampler

    def oracle_interval(self, x, alpha):
        x_eff = self._get_x_eff(x)
        lo = m_func(x_eff)
        hi = m_func(x_eff) + stats.expon.ppf(1.0 - alpha) * s_func(x_eff)
        return lo.reshape(-1, 1), hi.reshape(-1, 1)


class SynthLogNormalDataset(BaseSynthDataset):
    name = "Synth_LogNormal"

    def __init__(self, n_total=20000, dim=1, relevant_dim=1, x_range=(-2.0, 2.0), sigma=0.6):
        super().__init__(n_total=n_total, dim=dim, relevant_dim=relevant_dim, x_range=x_range)
        self.sigma = sigma
        self.mode_raw = np.exp(-(sigma ** 2))

    def get_dist(self, x_eff):
        sc = s_func(x_eff)
        return stats.lognorm(
            s=self.sigma,
            loc=m_func(x_eff) - self.mode_raw * sc,
            scale=sc,
        )

    def true_conditional_sampler(self):
        def _sampler(x, seed):
            x_eff = self._get_x_eff(x)
            sc = s_func(x_eff)
            rng = np.random.default_rng(seed)
            return (
                m_func(x_eff)
                + rng.lognormal(0, self.sigma, size=len(x_eff)).astype(np.float32) * sc
                - self.mode_raw * sc
            ).astype(np.float32)
        return _sampler

    def oracle_interval(self, x, alpha):
        x_eff = self._get_x_eff(x)
        dist_std = stats.lognorm(s=self.sigma, scale=1.0)
        h_l, h_h = find_hdi_generic(dist_std, confidence=1.0 - alpha)
        sc = s_func(x_eff)
        lo = m_func(x_eff) + h_l * sc - self.mode_raw * sc
        hi = m_func(x_eff) + h_h * sc - self.mode_raw * sc
        return lo.reshape(-1, 1), hi.reshape(-1, 1)


# =========================================================
# Real datasets
# =========================================================

class RealDataset(BaseDataset):
    def __init__(self, dataset_name: str, base_path: str = "datasets"):
        self.dataset_name = dataset_name
        self.base_path = Path(base_path)

    def generate_or_load(self, seed: int) -> DatasetBundle:
        X, y = load_real_dataset(self.dataset_name, self.base_path)
        return DatasetBundle(
            X=X.astype(np.float32),
            y=y.reshape(-1, 1).astype(np.float32),
            meta={"name": self.dataset_name, "is_real": True},
        )

    def supports_1d_plot(self) -> bool:
        return False


def load_real_dataset(name: str, base_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    name = str(name)
    base_path = Path(base_path)

    if name == "homes":
        df = pd.read_csv(base_path / "kc_house_data.csv")
        y = df["price"].values
        X = df.drop(["id", "date", "price"], axis=1).values

    elif name == "facebook_1":
        df = pd.read_csv(base_path / "facebook" / "Features_Variant_1.csv")
        y = np.log1p(df.iloc[:, 53].values)
        X = df.iloc[:, :53].values

    elif name == "facebook_2":
        df = pd.read_csv(base_path / "facebook" / "Features_Variant_2.csv")
        y = np.log1p(df.iloc[:, 53].values)
        X = df.iloc[:, :53].values

    elif name == "bio":
        df = pd.read_csv(base_path / "CASP.csv")
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

    elif name == "blog":
        df = pd.read_csv(base_path / "blogData_train.csv", header=None)
        X = df.iloc[:, :280].values
        y = np.log1p(df.iloc[:, -1].values)

    elif name == "bike":
        df = pd.read_csv(base_path / "bike_train.csv")

        season = pd.get_dummies(df["season"], prefix="season")
        weather = pd.get_dummies(df["weather"], prefix="weather")
        df = pd.concat([df, season, weather], axis=1)
        df.drop(["season", "weather"], axis=1, inplace=True)

        dt = pd.DatetimeIndex(df["datetime"])
        df["hour"] = dt.hour
        df["day"] = dt.dayofweek
        df["month"] = dt.month
        df["year"] = dt.year.map({2011: 0, 2012: 1})

        df.drop("datetime", axis=1, inplace=True)
        df.drop(["casual", "registered"], axis=1, inplace=True)

        X = df.drop("count", axis=1).values
        y = df["count"].values

    elif name == "superconductivity":
        df = pd.read_csv(base_path / "superconductivty_train.csv")
        y = df["critical_temp"].values
        X = df.drop("critical_temp", axis=1).values

    else:
        raise ValueError(
            f"Unsupported real dataset '{name}'. "
            f"Supported: bike, bio, blog, facebook_1, facebook_2, homes, superconductivity"
        )

    return X.astype(np.float32), y.astype(np.float32)


# =========================================================
# Factory
# =========================================================

def make_dataset(name: str, cfg_data) -> tuple[BaseDataset, bool]:
    if name == "Synth_Normal":
        return SynthNormalDataset(
            n_total=cfg_data.n_total,
            dim=cfg_data.dim,
            relevant_dim=cfg_data.relevant_dim,
            x_range=cfg_data.x_range,
        ), True

    if name == "Synth_LogNormal":
        return SynthLogNormalDataset(
            n_total=cfg_data.n_total,
            dim=cfg_data.dim,
            relevant_dim=cfg_data.relevant_dim,
            x_range=cfg_data.x_range,
        ), True

    if name == "Synth_Exponential":
        return SynthExponentialDataset(
            n_total=cfg_data.n_total,
            dim=cfg_data.dim,
            relevant_dim=cfg_data.relevant_dim,
            x_range=cfg_data.x_range,
        ), True

    return RealDataset(name, base_path=cfg_data.base_path), False


# =========================================================
# Run-time preprocessing
# =========================================================

def prepare_data_for_run(cfg: Config, dataset: BaseDataset, seed: int, is_synthetic: bool):
    bundle = dataset.generate_or_load(seed=seed + 123)
    X = np.asarray(bundle.X, dtype=np.float32)
    y = np.asarray(bundle.y, dtype=np.float32).reshape(-1)

    split = make_splits(
        n=len(X),
        train=cfg.splits.train,
        val=cfg.splits.val,
        cal=cfg.splits.cal,
        test=cfg.splits.test,
        seed=seed + 999,
        shuffle=cfg.splits.shuffle,
    )

    X_train = X[split.train]
    y_train = y[split.train]

    X_val = X[split.val]
    y_val = y[split.val]

    X_cal = X[split.cal]
    y_cal = y[split.cal]

    X_test = X[split.test]
    y_test = y[split.test]

    if not is_synthetic:
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train).astype(np.float32)
        X_val = scaler_x.transform(X_val).astype(np.float32)
        X_cal = scaler_x.transform(X_cal).astype(np.float32)
        X_test = scaler_x.transform(X_test).astype(np.float32)

        y_scale = float(np.mean(np.abs(y_train)))
        if y_scale == 0.0:
            y_scale = 1.0

        y_train = (y_train / y_scale).astype(np.float32)
        y_val = (y_val / y_scale).astype(np.float32)
        y_cal = (y_cal / y_scale).astype(np.float32)
        y_test = (y_test / y_scale).astype(np.float32)
    else:
        y_scale = 1.0

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_cal": X_cal,
        "y_cal": y_cal,
        "X_test": X_test,
        "y_test": y_test,
        "y_scale": y_scale,
    }