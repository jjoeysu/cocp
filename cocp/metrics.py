# cocp/metrics.py
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def evaluate_synthetic_intervals(dataset, X_test, lo, hi, n_mc: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(X_test)
    lo = np.asarray(lo).reshape(-1, 1)
    hi = np.asarray(hi).reshape(-1, 1)

    try:
        cov = dataset.get_cdf(x, hi) - dataset.get_cdf(x, lo)
        cov = np.clip(np.asarray(cov).reshape(-1), 0.0, 1.0).astype(np.float32)
    except Exception:
        rng = np.random.default_rng(seed)
        sampler = dataset.true_conditional_sampler()
        n = len(x)
        cov = np.zeros(n, dtype=np.float32)
        lo_flat = lo.reshape(-1)
        hi_flat = hi.reshape(-1)

        for _ in range(int(n_mc)):
            y_samp = sampler(x, seed=int(rng.integers(0, 2**31 - 1))).reshape(-1)
            cov += ((y_samp >= lo_flat) & (y_samp <= hi_flat)).astype(np.float32)

        cov /= float(n_mc)

    length = (hi.reshape(-1) - lo.reshape(-1)).astype(np.float32)
    return cov, length


def summarize_cov_len(cov: np.ndarray, length: np.ndarray, target: float) -> Dict:
    cov = np.asarray(cov)
    length = np.asarray(length)
    return {
        "target": float(target),
        "cov_mean": float(np.mean(cov)),
        "cov_std": float(np.std(cov)),
        "cov_mae_to_target": float(np.mean(np.abs(cov - target))),
        "cov_mse_to_target": float(np.mean((cov - target) ** 2)),
        "len_mean": float(np.mean(length)),
        "len_std": float(np.std(length)),
        "cov_all": cov,
        "len_all": length,
    }


def compute_msce(X, coverage_indicator, tau, K=10, random_state=42):
    X = np.asarray(X)
    Z = np.asarray(coverage_indicator).astype(float)
    n = len(Z)
    if n == 0:
        return np.nan

    K = min(int(K), n)
    if K <= 1:
        return float(np.mean((Z - tau) ** 2))

    Xs = StandardScaler().fit_transform(X)
    clusters = KMeans(n_clusters=K, n_init=10, random_state=random_state).fit_predict(Xs)

    msce = 0.0
    for k in range(K):
        mask = clusters == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        p_hat = float(Z[mask].mean())
        msce += (nk / n) * (p_hat - tau) ** 2

    return float(msce)


def _min_slab_coverage_for_direction(proj, Z, delta):
    n = len(proj)
    if n == 0:
        return np.nan

    m = int(np.ceil(delta * n))
    m = max(1, min(m, n))

    order = np.argsort(proj)
    Zs = Z[order]
    csum = np.concatenate([[0.0], np.cumsum(Zs)])

    best = 1.0
    for i in range(0, n - m + 1):
        j = i + m
        avg = (csum[j] - csum[i]) / m
        if avg < best:
            best = avg
    return float(best)


def compute_wsc(X, coverage_indicator, n_vectors=1000, delta=0.1, random_state=42, standardize=True):
    X = np.asarray(X)
    Z = np.asarray(coverage_indicator).astype(float)

    if len(Z) == 0:
        return np.nan

    Xs = StandardScaler().fit_transform(X) if standardize else X
    n, d = Xs.shape

    rng = np.random.default_rng(random_state)
    best = 1.0

    for _ in range(int(n_vectors)):
        v = rng.standard_normal(d)
        nv = np.linalg.norm(v)
        if nv == 0:
            continue
        v = v / nv
        proj = Xs @ v
        wsc_v = _min_slab_coverage_for_direction(proj, Z, delta)
        if not np.isnan(wsc_v):
            best = min(best, wsc_v)

    return float(best)


def _loss_proper(p, y, loss, tau=None):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    if loss == "brier":
        return (p - y) ** 2

    if loss == "l1":
        if tau is None:
            raise ValueError("tau must be provided for l1 loss")
        return np.sign(p - float(tau)) * (float(tau) - y)

    if loss == "log":
        eps = 1e-12
        p = np.clip(p, eps, 1 - eps)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    raise ValueError(f"Unknown loss: {loss}")


def compute_ert(X, coverage_indicator, alpha, loss="brier", n_splits=5, random_state=42, standardize=True):
    X = np.asarray(X)
    Z = np.asarray(coverage_indicator).astype(int)
    n = len(Z)
    if n == 0:
        return np.nan

    tau = 1.0 - float(alpha)
    p_oof = np.zeros(n, dtype=float)

    if len(np.unique(Z)) == 1:
        const_p = float(np.mean(Z))
        loss_const = _loss_proper(tau, Z, loss, tau=tau)
        loss_model = _loss_proper(const_p, Z, loss, tau=tau)
        return float(np.mean(loss_const - loss_model))

    kf = KFold(n_splits=min(int(n_splits), n), shuffle=True, random_state=random_state)

    for train_idx, val_idx in kf.split(X):
        X_tr = X[train_idx]
        X_va = X[val_idx]
        Z_tr = Z[train_idx]

        const_p = float(np.mean(Z_tr))
        if len(np.unique(Z_tr)) < 2:
            p_oof[val_idx] = const_p
            continue

        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_va = scaler.transform(X_va)

        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=2000)
        clf.fit(X_tr, Z_tr)
        p_oof[val_idx] = clf.predict_proba(X_va)[:, 1]

    loss_const = _loss_proper(tau, Z, loss, tau=tau)
    loss_model = _loss_proper(p_oof, Z, loss, tau=tau)
    return float(np.mean(loss_const - loss_model))


# By default, real-data metric evaluation uses the implementations from
# the `covmetrics` package for WSC and ERT.
#
# Set `use_covmetrics=False` to fall back to the internal implementations
# (`compute_wsc` and `compute_ert`) defined in this file. 
def compute_real_metrics(
    X_test,
    y_test,
    lo,
    hi,
    alpha: float,
    use_covmetrics: bool = True,
) -> Dict:
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)

    z = ((y_test >= lo) & (y_test <= hi)).astype(np.float32)
    length = (hi - lo).astype(np.float32)
    target_cov = 1.0 - float(alpha)

    msce = float(compute_msce(X_test, z, target_cov, K=10))

    if use_covmetrics:
        from covmetrics import ERT, WSC
        from covmetrics.losses import brier_score, L1_miscoverage

        wsc = float(WSC().evaluate(X_test, z, delta=0.1))
        ert_lr = ERT(model_cls=LogisticRegression, max_iter=2000)
        l1ert = float(ert_lr.evaluate(X_test, z, alpha=alpha, loss=L1_miscoverage))
        l2ert = float(ert_lr.evaluate(X_test, z, alpha=alpha, loss=brier_score))
    else:
        wsc = float(compute_wsc(X_test, z, n_vectors=1000, delta=0.1))
        l1ert = float(compute_ert(X_test, z, alpha=alpha, loss="l1"))
        l2ert = float(compute_ert(X_test, z, alpha=alpha, loss="brier"))

    return {
        "Coverage": float(np.mean(z)),
        "Length": float(np.mean(length)),
        "MSCE": msce,
        "WSC": wsc,
        "l1ERT": l1ert,
        "l2ERT": l2ert,
        "cov_all": z,
        "len_all": length,
    }