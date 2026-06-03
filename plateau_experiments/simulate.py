"""One experiment = one simulated optimization problem, end to end.

We draw a ground-truth Sharpe surface, simulate independent in-sample (IS) and
out-of-sample (OOS) returns, select a configuration two ways (naive argmax of IS
Sharpe vs a plateau-aware "smoothed" selection that prefers broad maxima), and
record every diagnostic together with the ground-truth outcomes (true regret,
IS->OOS inflation gap, realized OOS Sharpe). Batches of these records are the
raw material for the paper's analysis.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
from scipy.ndimage import gaussian_filter

from .baselines import deflated_sharpe, pbo_cscv
from .metrics import plateau_metrics
from .surfaces import (
    SurfaceConfig,
    sample_surface_config,
    sharpe,
    simulate_returns,
    true_sharpe_surface,
)


def surrogate_surface(sr_is: np.ndarray, cfg: SurfaceConfig, bandwidth_frac: float) -> np.ndarray:
    """Denoised view of the IS Sharpe surface.

    A single finite backtest gives a per-configuration Sharpe with estimation
    noise of order ``sqrt((1 + SR^2/2)/T)`` (annualized ~1.0 at T=252) -- so the
    raw per-cell surface is dominated by noise and *any* argmax looks like a
    needle. Practitioners never read raw per-point values; they assess the
    landscape through a surrogate (Optuna's contour/slice plots and fANOVA
    effectively smooth across trials). We model that surrogate with a Gaussian
    kernel of fixed disclosed bandwidth and assess plateau geometry on it.
    """
    grid = sr_is.reshape((cfg.grid_size,) * cfg.dim)
    sigma = bandwidth_frac * (cfg.grid_size - 1)
    return gaussian_filter(grid, sigma=sigma, mode="nearest").ravel()


def run_experiment(
    cfg: SurfaceConfig,
    rng: np.random.Generator,
    *,
    pbo_blocks: int = 10,
    delta_sharpe: float = 0.25,
    bandwidth_frac: float = 0.06,
) -> dict:
    sr_true = true_sharpe_surface(cfg)
    r_is = simulate_returns(sr_true, cfg.t_is, cfg, rng)
    r_oos = simulate_returns(sr_true, cfg.t_oos, cfg, rng)

    sr_is = sharpe(r_is, cfg.periods_per_year)     # (N,) raw IS annualized Sharpe surface
    sr_oos = sharpe(r_oos, cfg.periods_per_year)   # (N,) OOS annualized Sharpe
    surrogate = surrogate_surface(sr_is, cfg, bandwidth_frac)  # denoised landscape

    naive = int(np.nanargmax(sr_is))               # over-eager: argmax of raw IS
    robust = int(np.nanargmax(surrogate))          # plateau-aware: argmax of surrogate
    oracle = int(np.nanargmax(sr_true))

    # --- diagnostics: plateau geometry on the surrogate, stats on raw ---
    pm = plateau_metrics(surrogate, cfg, delta_sharpe=delta_sharpe)
    pbo = pbo_cscv(r_is, s_blocks=pbo_blocks)
    is_sharpes_per_obs = sr_is / np.sqrt(cfg.periods_per_year)
    dsr = deflated_sharpe(r_is[:, naive], is_sharpes_per_obs, n_trials=sr_is.size)

    # --- ground-truth outcomes -----------------------------------------
    record = {
        # geometry / config
        **{f"cfg_{k}": (v if not isinstance(v, tuple) else list(v)) for k, v in asdict(cfg).items()},
        "n_strategies": int(sr_is.size),
        # selections
        "sel_naive": naive,
        "sel_robust": robust,
        "sel_oracle": oracle,
        # ground truth
        "sr_true_oracle": float(sr_true[oracle]),
        "sr_true_naive": float(sr_true[naive]),
        "sr_true_robust": float(sr_true[robust]),
        "true_regret": float(sr_true[oracle] - sr_true[naive]),
        "true_regret_robust": float(sr_true[oracle] - sr_true[robust]),
        # realized performance
        "sr_is_naive": float(sr_is[naive]),
        "sr_oos_naive": float(sr_oos[naive]),
        "sr_oos_robust": float(sr_oos[robust]),
        "is_oos_gap": float(sr_is[naive] - sr_oos[naive]),
        "bad_selection": int(sr_oos[naive] < 0.0),          # deployed a loser OOS
        "robust_beats_naive_oos": int(sr_oos[robust] > sr_oos[naive]),
        # plateau-geometry diagnostics
        "robustness_score": pm["robustness_score"],
        "plateau_width": pm["plateau_width"],
        "min_plateau_width": pm["min_plateau_width"],
        "sensitivity": pm["sensitivity"],
        "max_sensitivity": pm["max_sensitivity"],
        "curvature": pm["curvature"],
        "outlier_ratio": pm["outlier_ratio"],
        "peak_is_sharpe": pm["peak_is_sharpe"],
        # statistical baselines
        "pbo": pbo["pbo"],
        "pbo_logit_mean": pbo["logit_mean"],
        "dsr": dsr["dsr"],
        "psr0": dsr["psr0"],
        "dsr_sr0": dsr["sr0"],
    }
    return record


def run_batch(
    n_experiments: int,
    *,
    dim: int = 1,
    grid_size: int = 41,
    seed: int = 0,
    pbo_blocks: int = 10,
    kinds: tuple[str, ...] = ("random",),
    progress_every: int = 0,
) -> list[dict]:
    """Run ``n_experiments`` with independently-seeded RNGs (reproducible)."""
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_experiments)
    records = []
    for i, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        kind = kinds[i % len(kinds)]
        cfg = sample_surface_config(rng, dim=dim, grid_size=grid_size, kind=kind)
        records.append(run_experiment(cfg, rng, pbo_blocks=pbo_blocks))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i + 1}/{n_experiments}", flush=True)
    return records
