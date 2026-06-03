"""Generative model of a strategy-optimization landscape with known ground truth.

We model backtest hyperparameter optimization as selecting, from a grid of
candidate configurations indexed by ``theta`` in ``[0,1]^d``, the one that
maximizes an *in-sample* (IS) Sharpe estimate. Each grid point is a "strategy"
that emits per-period returns; the *population* (true) Sharpe surface is what we
control, so for every simulated experiment we know the oracle optimum and can
measure how badly a noisy selection generalizes out of sample (OOS).

Returns carry a shared market factor, so nearby/all strategies are
cross-correlated — this is realistic and is exactly the structure that the
Probability of Backtest Overfitting (PBO/CSCV) and the Deflated Sharpe Ratio
were designed for, letting those baselines run on the *same* simulated data as
the plateau-geometry metrics.

Everything here is deterministic given a seeded ``numpy`` Generator.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

PERIODS_PER_YEAR = 252  # daily bars -> annualization factor sqrt(252)


@dataclass(frozen=True)
class SurfaceConfig:
    """Ground-truth geometry + noise for one simulated optimization problem."""

    dim: int = 1                 # number of hyperparameters (1 or 2 in the paper)
    grid_size: int = 41          # points per axis; N = grid_size ** dim strategies

    # --- true (population) annualized-Sharpe surface ---------------------
    base_level: float = 0.0      # SR floor away from any bump (can be < 0)
    plateau_amp: float = 1.0     # height of the broad concave region
    plateau_width: float = 0.25  # Gaussian sigma of the broad region (in [0,1] units)
    plateau_center: tuple[float, ...] = (0.5,)
    spike_amp: float = 0.0       # height of an additional NARROW peak (0 = none)
    spike_width: float = 0.04    # Gaussian sigma of the narrow peak
    spike_center: tuple[float, ...] = (0.5,)
    sr_cap: float = 4.0          # clip true annualized Sharpe to a sane ceiling

    # --- noise / sampling ----------------------------------------------
    factor_share: float = 0.3    # fraction of return variance from the shared factor
                                 # (== cross-strategy correlation), in [0, 1)
    t_is: int = 252              # in-sample observations
    t_oos: int = 252             # out-of-sample observations
    periods_per_year: int = PERIODS_PER_YEAR

    label: str = "custom"        # bookkeeping tag (e.g. "plateau"/"needle"/"null")


# --------------------------------------------------------------------------- #
# grid + true surface
# --------------------------------------------------------------------------- #
def make_grid(dim: int, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, axis)`` where ``points`` is ``(N, dim)`` in row-major
    order matching ``np.reshape`` to ``(grid_size,)*dim`` and ``axis`` is the
    shared per-axis coordinate vector ``linspace(0,1,grid_size)``."""
    axis = np.linspace(0.0, 1.0, grid_size)
    mesh = np.meshgrid(*([axis] * dim), indexing="ij")
    points = np.stack([m.ravel() for m in mesh], axis=-1)
    return points, axis


def _gaussian_bump(points: np.ndarray, center: tuple[float, ...], width: float) -> np.ndarray:
    c = np.asarray(center[: points.shape[1]], dtype=float)
    if c.shape[0] < points.shape[1]:  # broadcast a scalar center across axes
        c = np.full(points.shape[1], center[0], dtype=float)
    d2 = np.sum((points - c) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * width**2))


def true_sharpe_surface(cfg: SurfaceConfig) -> np.ndarray:
    """Population annualized-Sharpe value at every grid point, shape ``(N,)``."""
    points, _ = make_grid(cfg.dim, cfg.grid_size)
    sr = cfg.base_level + cfg.plateau_amp * _gaussian_bump(points, cfg.plateau_center, cfg.plateau_width)
    if cfg.spike_amp > 0:
        sr = sr + cfg.spike_amp * _gaussian_bump(points, cfg.spike_center, cfg.spike_width)
    return np.clip(sr, -cfg.sr_cap, cfg.sr_cap)


# --------------------------------------------------------------------------- #
# return simulation
# --------------------------------------------------------------------------- #
def simulate_returns(sr_true: np.ndarray, t: int, cfg: SurfaceConfig, rng: np.random.Generator) -> np.ndarray:
    """Simulate a ``(t, N)`` matrix of per-period returns whose population
    annualized Sharpe equals ``sr_true``.

    Each column has unit per-period variance; a shared factor ``F_t`` injects
    cross-strategy correlation equal to ``factor_share``. Per-period mean is
    ``sr_true / sqrt(periods_per_year)`` so that mean/sd * sqrt(P) == sr_true.
    """
    n = sr_true.shape[0]
    c = float(cfg.factor_share)
    per_period_mean = sr_true / np.sqrt(cfg.periods_per_year)  # (N,)

    factor = rng.standard_normal((t, 1))                       # shared market factor
    idio = rng.standard_normal((t, n))                         # idiosyncratic
    shocks = np.sqrt(c) * factor + np.sqrt(1.0 - c) * idio     # unit variance, corr = c
    return per_period_mean[None, :] + shocks


def sharpe(returns: np.ndarray, periods_per_year: int = PERIODS_PER_YEAR, axis: int = 0) -> np.ndarray:
    """Annualized Sharpe ratio per column (ddof=1), guarding zero variance."""
    mu = returns.mean(axis=axis)
    sd = returns.std(axis=axis, ddof=1)
    sd = np.where(sd <= 1e-12, np.nan, sd)
    return mu / sd * np.sqrt(periods_per_year)


# --------------------------------------------------------------------------- #
# random geometry sampler for Monte-Carlo batches
# --------------------------------------------------------------------------- #
GeometryKind = Literal["plateau", "needle", "mixed", "null", "random"]


def sample_surface_config(
    rng: np.random.Generator,
    dim: int = 1,
    grid_size: int = 41,
    kind: GeometryKind = "random",
    t_is_choices: tuple[int, ...] = (120, 252, 504),
    t_oos: int = 504,
) -> SurfaceConfig:
    """Draw a diverse random landscape. ``kind`` biases the geometry so the
    batch spans flat-null, wide-plateau, sharp-needle and mixed cases."""
    def centre() -> tuple[float, ...]:
        return tuple(rng.uniform(0.3, 0.7, size=dim).tolist())

    t_is = int(rng.choice(t_is_choices))
    factor_share = float(rng.uniform(0.05, 0.5))
    base = float(rng.uniform(-0.3, 0.3))

    if kind == "random":
        kind = rng.choice(["plateau", "needle", "mixed", "null"], p=[0.35, 0.25, 0.25, 0.15])

    if kind == "null":
        plateau_amp = float(rng.uniform(0.0, 0.15))
        spike_amp = 0.0
        width = float(rng.uniform(0.15, 0.4))
        spike_width = 0.04
    elif kind == "plateau":
        plateau_amp = float(rng.uniform(0.8, 2.2))
        spike_amp = 0.0
        width = float(rng.uniform(0.18, 0.45))
        spike_width = 0.04
    elif kind == "needle":
        plateau_amp = float(rng.uniform(0.0, 0.4))
        spike_amp = float(rng.uniform(1.0, 2.5))
        width = float(rng.uniform(0.15, 0.4))
        spike_width = float(rng.uniform(0.02, 0.055))
    else:  # mixed: a broad plateau plus a taller narrow spike elsewhere
        plateau_amp = float(rng.uniform(0.6, 1.4))
        spike_amp = float(rng.uniform(0.6, 1.6))
        width = float(rng.uniform(0.18, 0.4))
        spike_width = float(rng.uniform(0.02, 0.06))

    return SurfaceConfig(
        dim=dim,
        grid_size=grid_size,
        base_level=base,
        plateau_amp=plateau_amp,
        plateau_width=width,
        plateau_center=centre(),
        spike_amp=spike_amp,
        spike_width=spike_width,
        spike_center=centre(),
        factor_share=factor_share,
        t_is=t_is,
        t_oos=t_oos,
        label=kind,
    )


# canonical illustrative configs used by the figure that explains the setup
def canonical_configs() -> dict[str, SurfaceConfig]:
    return {
        "plateau": SurfaceConfig(dim=1, plateau_amp=1.4, plateau_width=0.28, plateau_center=(0.5,),
                                 spike_amp=0.0, t_is=252, label="plateau"),
        "needle": SurfaceConfig(dim=1, plateau_amp=0.2, plateau_width=0.3, plateau_center=(0.5,),
                                spike_amp=1.8, spike_width=0.035, spike_center=(0.5,), t_is=252, label="needle"),
        "mixed": SurfaceConfig(dim=1, plateau_amp=1.0, plateau_width=0.3, plateau_center=(0.35,),
                               spike_amp=1.1, spike_width=0.03, spike_center=(0.72,), t_is=252, label="mixed"),
        "null": SurfaceConfig(dim=1, plateau_amp=0.05, plateau_width=0.3, spike_amp=0.0,
                              base_level=0.0, t_is=252, label="null"),
    }
