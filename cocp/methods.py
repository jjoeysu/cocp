# cocp/methods.py
from __future__ import annotations

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
        grad_clip: float,
        phase_name: str = "",
        verbose: bool = True,
    ):
        best_val = float("inf")
        best_state = None
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

            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(val_x, val_y, epoch).item())

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(model.state_dict())
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
        grad_clip = float(cfg.training.grad_clip)

        lr_mu_max = float(acfg.get("lr_mu_max", 1e-3))
        lr_mu_min = float(acfg.get("lr_mu_min", 1e-5))
        lr_h_max = float(acfg.get("lr_h_max", 1e-3))
        lr_h_min = float(acfg.get("lr_h_min", 1e-5))

        beta_start = float(acfg.get("beta_start", 0.01))
        beta_end = float(acfg.get("beta_end", 0.01))

        n_folds = int(acfg.get("n_folds", 4))
        n_alt_iters = int(acfg.get("n_alt_iters", 5))
        use_cp = bool(acfg.get("use_cp", True))
        verbose = bool(acfg.get("verbose", True))

        if n_folds < 2:
            raise ValueError("n_folds must be >= 2")

        val_x = torch.from_numpy(X_val).float().to(device)
        val_y = torch.from_numpy(y_val).float().view(-1).to(device)

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=ctx.seed)
        folds = list(kf.split(X_train))

        ensemble_states = []

        print(f"\n>>> Start CoCP | K={n_folds}, T={n_alt_iters}, beta=({beta_start}, {beta_end})")

        for fold_idx, (mu_train_idx, h_train_idx) in enumerate(folds):
            print(f"--- Fold {fold_idx + 1}/{n_folds} ---")

            loader_mu = self._make_loader(X_train, y_train, mu_train_idx, batch_size=batch_size, shuffle=True)
            loader_h = self._make_loader(X_train, y_train, h_train_idx, batch_size=batch_size, shuffle=True)

            mu_net = MeanNet(input_dim, num_hidden, num_layers, dropout).to(device)
            h_net = ThresholdNet(input_dim, num_hidden, num_layers, dropout).to(device)

            # Phase 1: warmup Mu
            opt_mu = torch.optim.AdamW(mu_net.parameters(), lr=lr_mu_max, weight_decay=weight_decay)
            sch_mu = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mu, T_max=max_epochs, eta_min=lr_mu_min)

            self._train_phase(
                model=mu_net,
                optimizer=opt_mu,
                scheduler=sch_mu,
                loader=loader_mu,
                val_x=val_x,
                val_y=val_y,
                loss_fn=lambda xb, yb, epoch: F.mse_loss(mu_net(xb).view(-1), yb),
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                grad_clip=grad_clip,
                phase_name=f"F{fold_idx}-WarmupMu",
                verbose=verbose,
            )

            # Phase 2/3: alternating optimization
            for it in range(n_alt_iters):
                print(f"  > Alternating iteration {it + 1}/{n_alt_iters}")

                loss_h = self._make_h_loss(mu_net, h_net, tau)
                opt_h = torch.optim.AdamW(h_net.parameters(), lr=lr_h_max, weight_decay=weight_decay)
                sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=max_epochs, eta_min=lr_h_min)

                self._train_phase(
                    model=h_net,
                    optimizer=opt_h,
                    scheduler=sch_h,
                    loader=loader_h,
                    val_x=val_x,
                    val_y=val_y,
                    loss_fn=loss_h,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    grad_clip=grad_clip,
                    phase_name=f"F{fold_idx}-It{it + 1}-TrainH",
                    verbose=verbose,
                )

                loss_mu = self._make_mu_cov_loss(mu_net, h_net, max_epochs, beta_start, beta_end)
                opt_mu_refine = torch.optim.AdamW(mu_net.parameters(), lr=lr_mu_max, weight_decay=weight_decay)
                sch_mu_refine = torch.optim.lr_scheduler.CosineAnnealingLR(opt_mu_refine, T_max=max_epochs, eta_min=lr_mu_min)

                self._train_phase(
                    model=mu_net,
                    optimizer=opt_mu_refine,
                    scheduler=sch_mu_refine,
                    loader=loader_mu,
                    val_x=val_x,
                    val_y=val_y,
                    loss_fn=loss_mu,
                    device=device,
                    max_epochs=max_epochs,
                    patience=patience,
                    grad_clip=grad_clip,
                    phase_name=f"F{fold_idx}-It{it + 1}-RefineMu",
                    verbose=verbose,
                )

            # Phase 4: final H adjustment
            loss_h_final = self._make_h_loss(mu_net, h_net, tau)
            opt_h_final = torch.optim.AdamW(h_net.parameters(), lr=lr_h_max, weight_decay=weight_decay)
            sch_h_final = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h_final, T_max=max_epochs, eta_min=lr_h_min)

            self._train_phase(
                model=h_net,
                optimizer=opt_h_final,
                scheduler=sch_h_final,
                loader=loader_h,
                val_x=val_x,
                val_y=val_y,
                loss_fn=loss_h_final,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                grad_clip=grad_clip,
                phase_name=f"F{fold_idx}-FinalH",
                verbose=verbose,
            )

            ensemble_states.append({
                "mu_state": self._cpu_state_dict(mu_net),
                "h_state": self._cpu_state_dict(h_net),
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