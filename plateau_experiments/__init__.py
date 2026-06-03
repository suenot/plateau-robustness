"""Controlled validation of plateau-robustness metrics for backtest overfitting."""

from .baselines import deflated_sharpe, pbo_cscv, probabilistic_sharpe
from .metrics import plateau_metrics
from .simulate import run_batch, run_experiment
from .surfaces import SurfaceConfig, canonical_configs, sample_surface_config, true_sharpe_surface

__all__ = [
    "SurfaceConfig",
    "true_sharpe_surface",
    "sample_surface_config",
    "canonical_configs",
    "plateau_metrics",
    "pbo_cscv",
    "deflated_sharpe",
    "probabilistic_sharpe",
    "run_experiment",
    "run_batch",
]
__version__ = "0.1.0"
