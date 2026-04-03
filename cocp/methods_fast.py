from __future__ import annotations

from copy import deepcopy

from .config import Config
from .methods import CoCP


class CoCPFast(CoCP):
    """
    Drop-in accelerated variant of CoCP.

    The prediction objective and conformal calibration remain unchanged.
    This variant only changes the training schedule by:
    - using phase-specific early stopping thresholds
    - shortening warmup/refinement/H budgets
    - enabling optional fold-level parallelism
    """

    name = "CoCP-Fast"

    def __init__(self):
        super().__init__(persistent_blocks=False)
        self.name = "CoCP-Fast"

    @staticmethod
    def _with_fast_defaults(cfg: Config) -> Config:
        cfg_fast = deepcopy(cfg)
        acfg = cfg_fast.training.cocp

        defaults = {
            "lr_mu_max": 1.5e-3,
            "lr_h_max": 8e-4,
            "warmup_mu_max_epochs": 300,
            "warmup_mu_patience": 30,
            "warmup_mu_min_delta": 1e-4,
            "refine_mu_max_epochs": 120,
            "refine_mu_patience": 15,
            "refine_mu_min_delta": 5e-5,
            "h_max_epochs": 80,
            "h_patience": 12,
            "h_min_delta": 5e-5,
            "n_fold_workers": 2,
            "fold_parallel_backend": "thread",
            "fold_num_threads": 1,
            "persistent_blocks": False,
        }
        for key, value in defaults.items():
            acfg.setdefault(key, value)
        return cfg_fast

    def fit(self, X_train, y_train, X_val, y_val, X_cal, y_cal, ctx, cfg):
        cfg_fast = self._with_fast_defaults(cfg)
        return super().fit(X_train, y_train, X_val, y_val, X_cal, y_cal, ctx, cfg_fast)
