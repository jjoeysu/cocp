# cocp/methods.py
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
import copy
import math
from typing import Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from .models import MeanNet, ThresholdNet


@dataclass
class FitContext:
    device: str
    run_dir: str
    seed: int
    alpha: float


def pinball(y: torch.Tensor, q: torch.Tensor, tau: float) -> torch.Tensor:
    u = y - q
    return torch.maximum(tau * u, (tau - 1.0) * u)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration scores are empty.")
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def _fit_single_fold(payload: Dict[str, Any]) -> Dict[str, Any]:
    fold_idx = int(payload["fold_idx"])
    X_train = payload["X_train"]
    y_train = payload["y_train"]
    X_val = payload["X_val"]
    y_val = payload["y_val"]
    mu_train_idx = payload["mu_train_idx"]
    h_train_idx = payload["h_train_idx"]
    params = payload["params"]

    fold_num_threads = int(params.get("fold_num_threads", 0))
    if fold_num_threads > 0:
        torch.set_num_threads(fold_num_threads)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(max(1, min(fold_num_threads, 2)))
            except RuntimeError:
                pass

    fold_seed = int(params["seed"]) + fold_idx
    np.random.seed(fold_seed)
    torch.manual_seed(fold_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(fold_seed)

    device_name = str(params["device"])
    if "cuda" in device_name and torch.cuda.is_available():
        device = torch.device(device_name)
    else:
        device = torch.device("cpu")

    tau = float(params["tau"])
    input_dim = int(params["input_dim"])
    num_hidden = int(params["num_hidden"])
    num_layers = int(params["num_layers"])
    dropout = float(params["dropout"])
    weight_decay = float(params["weight_decay"])
    batch_size = int(params["batch_size"])
    grad_clip = float(params["grad_clip"])

    warmup_mu_max_epochs = int(params["warmup_mu_max_epochs"])
    warmup_mu_patience = int(params["warmup_mu_patience"])
    warmup_mu_min_delta = float(params["warmup_mu_min_delta"])

    refine_mu_max_epochs = int(params["refine_mu_max_epochs"])
    refine_mu_patience = int(params["refine_mu_patience"])
    refine_mu_min_delta = float(params["refine_mu_min_delta"])

    h_max_epochs = int(params["h_max_epochs"])
    h_patience = int(params["h_patience"])
    h_min_delta = float(params["h_min_delta"])

    lr_mu_max = float(params["lr_mu_max"])
    lr_mu_min = float(params["lr_mu_min"])
    lr_h_max = float(params["lr_h_max"])
    lr_h_min = float(params["lr_h_min"])
    beta_start = float(params["beta_start"])
    beta_end = float(params["beta_end"])
    n_alt_iters = int(params["n_alt_iters"])
    persistent_blocks = bool(params["persistent_blocks"])
    verbose = bool(params["verbose"])

    val_x = torch.from_numpy(X_val).float().to(device)
    val_y = torch.from_numpy(y_val).float().view(-1).to(device)

    loader_mu = CoCP._make_loader(X_train, y_train, mu_train_idx, batch_size=batch_size, shuffle=True)
    loader_h = CoCP._make_loader(X_train, y_train, h_train_idx, batch_size=batch_size, shuffle=True)

    mu_net = MeanNet(input_dim, num_hidden, num_layers, dropout).to(device)
    h_net = ThresholdNet(input_dim, num_hidden, num_layers, dropout).to(device)

    if persistent_blocks:
        mu_total_epochs = max(1, warmup_mu_max_epochs + n_alt_iters * refine_mu_max_epochs)
        h_total_epochs = max(1, (n_alt_iters + 1) * h_max_epochs)

        opt_mu = torch.optim.AdamW(mu_net.parameters(), lr=lr_mu_max, weight_decay=weight_decay)
        sch_mu = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_mu,
            T_max=mu_total_epochs,
            eta_min=lr_mu_min,
        )
        opt_h = torch.optim.AdamW(h_net.parameters(), lr=lr_h_max, weight_decay=weight_decay)
        sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_h,
            T_max=h_total_epochs,
            eta_min=lr_h_min,
        )

    if not persistent_blocks:
        opt_mu = torch.optim.AdamW(mu_net.parameters(), lr=lr_mu_max, weight_decay=weight_decay)
        sch_mu = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_mu,
            T_max=warmup_mu_max_epochs,
            eta_min=lr_mu_min,
        )

    CoCP._train_phase(
        model=mu_net,
        optimizer=opt_mu,
        scheduler=sch_mu,
        loader=loader_mu,
        val_x=val_x,
        val_y=val_y,
        loss_fn=lambda xb, yb, epoch: F.mse_loss(mu_net(xb).view(-1), yb),
        device=device,
        max_epochs=warmup_mu_max_epochs,
        patience=warmup_mu_patience,
        min_delta=warmup_mu_min_delta,
        grad_clip=grad_clip,
        phase_name=f"F{fold_idx}-WarmupMu",
        verbose=verbose,
        restore_aux_state=persistent_blocks,
    )

    for it in range(n_alt_iters):
        loss_h = CoCP._make_h_loss(mu_net, h_net, tau)
        if not persistent_blocks:
            opt_h = torch.optim.AdamW(h_net.parameters(), lr=lr_h_max, weight_decay=weight_decay)
            sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_h,
                T_max=h_max_epochs,
                eta_min=lr_h_min,
            )

        CoCP._train_phase(
            model=h_net,
            optimizer=opt_h,
            scheduler=sch_h,
            loader=loader_h,
            val_x=val_x,
            val_y=val_y,
            loss_fn=loss_h,
            device=device,
            max_epochs=h_max_epochs,
            patience=h_patience,
            min_delta=h_min_delta,
            grad_clip=grad_clip,
            phase_name=f"F{fold_idx}-It{it + 1}-TrainH",
            verbose=verbose,
            restore_aux_state=persistent_blocks,
        )

        loss_mu = CoCP._make_mu_cov_loss(mu_net, h_net, refine_mu_max_epochs, beta_start, beta_end)
        if persistent_blocks:
            opt_mu_refine = opt_mu
            sch_mu_refine = sch_mu
        else:
            opt_mu_refine = torch.optim.AdamW(mu_net.parameters(), lr=lr_mu_max, weight_decay=weight_decay)
            sch_mu_refine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt_mu_refine,
                T_max=refine_mu_max_epochs,
                eta_min=lr_mu_min,
            )

        CoCP._train_phase(
            model=mu_net,
            optimizer=opt_mu_refine,
            scheduler=sch_mu_refine,
            loader=loader_mu,
            val_x=val_x,
            val_y=val_y,
            loss_fn=loss_mu,
            device=device,
            max_epochs=refine_mu_max_epochs,
            patience=refine_mu_patience,
            min_delta=refine_mu_min_delta,
            grad_clip=grad_clip,
            phase_name=f"F{fold_idx}-It{it + 1}-RefineMu",
            verbose=verbose,
            restore_aux_state=persistent_blocks,
        )

    loss_h_final = CoCP._make_h_loss(mu_net, h_net, tau)
    if persistent_blocks:
        opt_h_final = opt_h
        sch_h_final = sch_h
    else:
        opt_h_final = torch.optim.AdamW(h_net.parameters(), lr=lr_h_max, weight_decay=weight_decay)
        sch_h_final = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_h_final,
            T_max=h_max_epochs,
            eta_min=lr_h_min,
        )

    CoCP._train_phase(
        model=h_net,
        optimizer=opt_h_final,
        scheduler=sch_h_final,
        loader=loader_h,
        val_x=val_x,
        val_y=val_y,
        loss_fn=loss_h_final,
        device=device,
        max_epochs=h_max_epochs,
        patience=h_patience,
        min_delta=h_min_delta,
        grad_clip=grad_clip,
        phase_name=f"F{fold_idx}-FinalH",
        verbose=verbose,
        restore_aux_state=persistent_blocks,
    )

    return {
        "fold_idx": fold_idx,
        "mu_state": CoCP._cpu_state_dict(mu_net),
        "h_state": CoCP._cpu_state_dict(h_net),
    }


class CoCP:
    """
    CoCP (Co-optimization for Adaptive Conformal Prediction)

    Training logic:
    1. Warmup Mu with MSE
    2. Alternate:
       - train H by pinball loss on |y - mu(x)|
       - refine Mu by smooth coverage objective
    3. Final H adjustment
    4. Optional conformal calibration on calibration set
    """

    name = "CoCP"

    def __init__(self, persistent_blocks: bool = False):
        self.persistent_blocks = bool(persistent_blocks)
        self.name = "CoCP-Persistent" if self.persistent_blocks else "CoCP"

    @staticmethod
    def _device(ctx: FitContext) -> torch.device:
        if "cuda" in ctx.device and torch.cuda.is_available():
            return torch.device(ctx.device)
        return torch.device("cpu")

    @staticmethod
    def _cpu_state_dict(model: torch.nn.Module):
        return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    @staticmethod
    def _make_loader(X, y, indices, batch_size, shuffle=True):
        ds = TensorDataset(
            torch.from_numpy(X[indices]).float(),
            torch.from_numpy(y[indices]).float().view(-1),
        )
        return DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle)

    @staticmethod
    def _get_exponential_beta(epoch: int, max_epochs: int, start: float, end: float) -> float:
        if start <= 0 or end <= 0:
            raise ValueError("beta_start and beta_end must be > 0.")
        if max_epochs <= 1:
            return float(end)
        return float(start * ((end / start) ** (epoch / (max_epochs - 1))))

    @staticmethod
    def _train_phase(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loader,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        loss_fn,
        device: torch.device,
        max_epochs: int,
        patience: int,
        min_delta: float,
        grad_clip: float,
        phase_name: str = "",
        verbose: bool = True,
        restore_aux_state: bool = False,
    ):
        best_val = float("inf")
        best_state = None
        best_opt_state = None
        best_sched_state = None
        bad_epochs = 0

        for epoch in range(int(max_epochs)):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device).view(-1)

                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(xb, yb, epoch)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(val_x, val_y, epoch).item())

            if val_loss < (best_val - float(min_delta)):
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
                if restore_aux_state:
                    best_opt_state = copy.deepcopy(optimizer.state_dict())
                    best_sched_state = copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None
                bad_epochs = 0
            else:
                bad_epochs += 1

            if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
                print(f"[{phase_name}] epoch={epoch + 1} val={val_loss:.6f}")

            if bad_epochs >= int(patience):
                if verbose:
                    print(f"[{phase_name}] early stop at epoch {epoch + 1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
            if restore_aux_state and best_opt_state is not None:
                optimizer.load_state_dict(best_opt_state)
                if scheduler is not None and best_sched_state is not None:
                    scheduler.load_state_dict(best_sched_state)

    @staticmethod
    def _make_h_loss(mu_net: MeanNet, h_net: ThresholdNet, tau: float):
        mu_net.eval()

        def loss_h(xb, yb, epoch):
            with torch.no_grad():
                mu_y = mu_net(xb).view(-1)
            diff = torch.abs(yb - mu_y)
            h_val = h_net(xb).view(-1)
            return pinball(diff, h_val, tau).mean()

        return loss_h

    @staticmethod
    def _make_mu_cov_loss(mu_net: MeanNet, h_net: ThresholdNet, max_epochs: int, beta_start: float, beta_end: float):
        h_net.eval()

        def loss_mu(xb, yb, epoch):
            curr_beta = CoCP._get_exponential_beta(epoch, max_epochs, beta_start, beta_end)
            mu_y = mu_net(xb).view(-1)
            with torch.no_grad():
                h_y = h_net(xb).view(-1)
            return -torch.sigmoid((h_y - torch.abs(yb - mu_y)) / curr_beta).mean()

        return loss_mu

    def _predict_ensemble(self, xt: torch.Tensor, state: Dict[str, Any], device: torch.device):
        input_dim = int(state["input_dim"])
        num_hidden = int(state["num_hidden"])
        num_layers = int(state["num_layers"])
        dropout = float(state["dropout"])
        ensemble_states = state["ensemble_states"]

        tmp_mu = MeanNet(input_dim, num_hidden, num_layers, dropout).to(device)
        tmp_h = ThresholdNet(input_dim, num_hidden, num_layers, dropout).to(device)

        all_mu = []
        all_h = []

        with torch.no_grad():
            for member in ensemble_states:
                tmp_mu.load_state_dict(member["mu_state"])
                tmp_h.load_state_dict(member["h_state"])
                tmp_mu.eval()
                tmp_h.eval()
                all_mu.append(tmp_mu(xt).view(-1))
                all_h.append(tmp_h(xt).view(-1))

        mu_ens = torch.stack(all_mu, dim=0).mean(dim=0)
        h_ens = torch.stack(all_h, dim=0).mean(dim=0)
        return mu_ens, h_ens

    def _calibrate(self, X_cal, y_cal, state, ctx: FitContext, use_cp: bool = True) -> float:
        if not use_cp:
            return 1.0

        device = self._device(ctx)
        xc_t = torch.from_numpy(X_cal).float().to(device)
        yc_t = torch.from_numpy(y_cal).float().view(-1).to(device)

        mu_ens, h_ens = self._predict_ensemble(xc_t, state, device)
        scores = (torch.abs(yc_t - mu_ens) / torch.clamp(h_ens, min=1e-6)).detach().cpu().numpy()
        return conformal_quantile(scores, alpha=ctx.alpha)

    def fit(self, X_train, y_train, X_val, y_val, X_cal, y_cal, ctx: FitContext, cfg):
        acfg = cfg.training.cocp
        device = self._device(ctx)
        tau = 1.0 - float(ctx.alpha)

        input_dim = int(X_train.shape[1])
        num_hidden = int(acfg.get("num_hidden", 64))
        num_layers = int(acfg.get("num_layers", 2))
        dropout = float(acfg.get("dropout", 0.0))
        weight_decay = float(acfg.get("weight_decay", 1e-5))

        batch_size = int(acfg.get("batch_size", cfg.training.batch_size))
        max_epochs = int(acfg.get("max_epochs", cfg.training.max_epochs))
        patience = int(acfg.get("patience", cfg.training.patience))
        min_delta = float(acfg.get("min_delta", 0.0))
        grad_clip = float(cfg.training.grad_clip)

        warmup_mu_max_epochs = int(acfg.get("warmup_mu_max_epochs", max_epochs))
        warmup_mu_patience = int(acfg.get("warmup_mu_patience", patience))
        warmup_mu_min_delta = float(acfg.get("warmup_mu_min_delta", min_delta))

        refine_mu_max_epochs = int(acfg.get("refine_mu_max_epochs", max_epochs))
        refine_mu_patience = int(acfg.get("refine_mu_patience", patience))
        refine_mu_min_delta = float(acfg.get("refine_mu_min_delta", min_delta))

        h_max_epochs = int(acfg.get("h_max_epochs", max_epochs))
        h_patience = int(acfg.get("h_patience", patience))
        h_min_delta = float(acfg.get("h_min_delta", min_delta))

        lr_mu_max = float(acfg.get("lr_mu_max", 1e-3))
        lr_mu_min = float(acfg.get("lr_mu_min", 1e-5))
        lr_h_max = float(acfg.get("lr_h_max", 1e-3))
        lr_h_min = float(acfg.get("lr_h_min", 1e-5))

        beta_start = float(acfg.get("beta_start", 0.01))
        beta_end = float(acfg.get("beta_end", 0.01))

        n_folds = int(acfg.get("n_folds", 4))
        n_alt_iters = int(acfg.get("n_alt_iters", 5))
        n_fold_workers = int(acfg.get("n_fold_workers", 1))
        fold_parallel_backend = str(acfg.get("fold_parallel_backend", "thread")).lower()
        fold_num_threads = int(acfg.get("fold_num_threads", 1 if n_fold_workers > 1 else 0))
        use_cp = bool(acfg.get("use_cp", True))
        verbose = bool(acfg.get("verbose", True))

        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")

        val_x = torch.from_numpy(X_val).float().to(device)
        val_y = torch.from_numpy(y_val).float().view(-1).to(device)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=ctx.seed)
        folds = list(kf.split(X_train))

        ensemble_states = []
        fold_params = {
            "device": ctx.device,
            "seed": ctx.seed,
            "tau": tau,
            "input_dim": input_dim,
            "num_hidden": num_hidden,
            "num_layers": num_layers,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "grad_clip": grad_clip,
            "warmup_mu_max_epochs": warmup_mu_max_epochs,
            "warmup_mu_patience": warmup_mu_patience,
            "warmup_mu_min_delta": warmup_mu_min_delta,
            "refine_mu_max_epochs": refine_mu_max_epochs,
            "refine_mu_patience": refine_mu_patience,
            "refine_mu_min_delta": refine_mu_min_delta,
            "h_max_epochs": h_max_epochs,
            "h_patience": h_patience,
            "h_min_delta": h_min_delta,
            "lr_mu_max": lr_mu_max,
            "lr_mu_min": lr_mu_min,
            "lr_h_max": lr_h_max,
            "lr_h_min": lr_h_min,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "n_alt_iters": n_alt_iters,
            "persistent_blocks": self.persistent_blocks,
            "verbose": verbose,
            "fold_num_threads": fold_num_threads,
        }

        print(
            f"\n>>> Start {self.name} | K={n_folds}, T={n_alt_iters}, "
            f"beta=({beta_start}, {beta_end})"
        )
        fold_payloads = []
        for fold_idx, (mu_train_idx, h_train_idx) in enumerate(folds):
            fold_payloads.append({
                "fold_idx": fold_idx,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "mu_train_idx": mu_train_idx,
                "h_train_idx": h_train_idx,
                "params": fold_params,
            })

        can_parallelize = (
            n_fold_workers > 1
            # and device.type == "cpu"
            and n_folds > 1
        )

        if can_parallelize:
            print(
                f"--- Running {n_folds} folds in parallel with {n_fold_workers} "
                f"{fold_parallel_backend} workers ---"
            )
            try:
                if fold_parallel_backend == "process":
                    executor_cls = concurrent.futures.ProcessPoolExecutor
                else:
                    executor_cls = concurrent.futures.ThreadPoolExecutor
                with executor_cls(max_workers=n_fold_workers) as executor:
                    fold_results = list(executor.map(_fit_single_fold, fold_payloads))
            except PermissionError:
                print("--- Process backend unavailable here; falling back to thread workers ---")
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_fold_workers) as executor:
                    fold_results = list(executor.map(_fit_single_fold, fold_payloads))
        else:
            fold_results = []
            for fold_idx, payload in enumerate(fold_payloads):
                print(f"--- Fold {fold_idx + 1}/{n_folds} ---")
                fold_results.append(_fit_single_fold(payload))

        fold_results = sorted(fold_results, key=lambda item: int(item["fold_idx"]))
        for fold_result in fold_results:
            ensemble_states.append({
                "mu_state": fold_result["mu_state"],
                "h_state": fold_result["h_state"],
            })

        state = {
            "ensemble_states": ensemble_states,
            "input_dim": input_dim,
            "num_hidden": num_hidden,
            "num_layers": num_layers,
            "dropout": dropout,
            "cal_factor": 1.0,
        }

        state["cal_factor"] = float(self._calibrate(X_cal, y_cal, state, ctx, use_cp=use_cp))
        print(f">>> Calibration factor = {state['cal_factor']:.6f}")
        return state

    def predict(self, X, state, ctx: FitContext):
        device = self._device(ctx)
        xt = torch.from_numpy(X).float().to(device)

        mu_ens, h_ens = self._predict_ensemble(xt, state, device)
        mu_np = mu_ens.detach().cpu().numpy()
        h_np = h_ens.detach().cpu().numpy()
        cal = float(state.get("cal_factor", 1.0))

        lo = (mu_np - cal * h_np).astype(np.float32)
        hi = (mu_np + cal * h_np).astype(np.float32)
        return lo, hi
