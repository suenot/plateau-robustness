"""Local-geometry "plateau" robustness metrics, computed from the in-sample
Sharpe surface only.

These re-implement the three metrics from the source article (sensitivity,
relative plateau width, combined robustness score) but fix two estimator flaws
identified in review:

1. **Ceteris-paribus slices.** Per-parameter quantities are evaluated along the
   axis through the optimum with *other parameters held fixed* (exact on a grid),
   rather than regressing over top trials in which all parameters co-vary.
2. **Robust width via a connected super-level set.** Plateau width is the
   contiguous interval around the optimum where the surface stays within a
   margin of the peak — not ``max - min`` over all "good" points, which a single
   distant lucky point inflates.

Sensitivity uses a finite, well-posed local drop / curvature instead of the
article's elasticity (whose first-derivative term vanishes at an interior
maximum). All functions take the flat ``(N,)`` IS-Sharpe surface and the
``SurfaceConfig`` and return plain floats/dicts.
"""

from __future__ import annotations

import numpy as np

from .surfaces import SurfaceConfig


def _as_grid(surface: np.ndarray, cfg: SurfaceConfig) -> np.ndarray:
    return surface.reshape((cfg.grid_size,) * cfg.dim)


def _opt_index(grid: np.ndarray) -> tuple[int, ...]:
    return np.unravel_index(int(np.nanargmax(grid)), grid.shape)


def _axis_line(grid: np.ndarray, opt: tuple[int, ...], axis: int) -> np.ndarray:
    """1-D slice through ``opt`` varying ``axis`` with all other axes fixed."""
    idx: list[object] = list(opt)
    idx[axis] = slice(None)
    return grid[tuple(idx)]


# --------------------------------------------------------------------------- #
# per-axis primitives
# --------------------------------------------------------------------------- #
def connected_width(line: np.ndarray, opt_pos: int, threshold: float) -> float:
    """Fraction of the axis covered by the contiguous run around ``opt_pos`` for
    which ``line >= threshold``. In [0, 1]."""
    n = line.shape[0]
    if n < 2 or not np.isfinite(line[opt_pos]) or line[opt_pos] < threshold:
        return 0.0
    lo = opt_pos
    while lo - 1 >= 0 and np.isfinite(line[lo - 1]) and line[lo - 1] >= threshold:
        lo -= 1
    hi = opt_pos
    while hi + 1 < n and np.isfinite(line[hi + 1]) and line[hi + 1] >= threshold:
        hi += 1
    return (hi - lo) / (n - 1)


def local_sensitivity(line: np.ndarray, opt_pos: int, step_frac: float) -> float:
    """Absolute Sharpe drop when stepping ``step_frac`` of the axis range away
    from the optimum (mean of the two sides; one-sided at a boundary). Larger =
    more fragile. Well-defined for any sign of the optimum."""
    n = line.shape[0]
    step = max(1, int(round(step_frac * (n - 1))))
    peak = line[opt_pos]
    drops = []
    for j in (opt_pos - step, opt_pos + step):
        if 0 <= j < n and np.isfinite(line[j]):
            drops.append(peak - line[j])
    return float(np.mean(drops)) if drops else 0.0


def local_curvature(line: np.ndarray, opt_pos: int, half_window: int = 3) -> float:
    """Curvature ``-2a`` of a quadratic fit on a small window around the optimum
    (positive at a maximum; larger = sharper). Normalized by squared grid step
    so it is comparable across grid sizes."""
    n = line.shape[0]
    lo, hi = max(0, opt_pos - half_window), min(n, opt_pos + half_window + 1)
    xs = np.arange(lo, hi) / (n - 1)
    ys = line[lo:hi]
    mask = np.isfinite(ys)
    if mask.sum() < 3:
        return 0.0
    a = np.polyfit(xs[mask], ys[mask], 2)[0]
    return float(max(0.0, -2.0 * a))


# --------------------------------------------------------------------------- #
# fANOVA-style first-order importance weights
# --------------------------------------------------------------------------- #
def fanova_weights(grid: np.ndarray) -> np.ndarray:
    """First-order functional-ANOVA importance per axis: variance of each axis's
    main effect divided by total variance. Returns weights summing to 1 (falls
    back to uniform if the surface is flat)."""
    dim = grid.ndim
    if dim == 1:
        return np.ones(1)
    grand = float(np.nanmean(grid))
    total_var = float(np.nanvar(grid))
    if total_var <= 1e-12:
        return np.ones(dim) / dim
    importances = np.empty(dim)
    for ax in range(dim):
        other = tuple(a for a in range(dim) if a != ax)
        main_effect = np.nanmean(grid, axis=other) - grand  # marginal effect along ax
        importances[ax] = np.nanmean(main_effect**2)
    if importances.sum() <= 1e-12:
        return np.ones(dim) / dim
    return importances / importances.sum()


# --------------------------------------------------------------------------- #
# top-level metric bundle
# --------------------------------------------------------------------------- #
def plateau_metrics(
    surface: np.ndarray,
    cfg: SurfaceConfig,
    delta_sharpe: float = 0.25,
    step_frac: float = 0.10,
    eps: float = 1e-3,
) -> dict:
    """Compute the full plateau-robustness bundle from an IS-Sharpe surface.

    ``delta_sharpe`` is the absolute annualized-Sharpe margin defining the
    plateau super-level set (peak minus margin). ``step_frac`` is the relative
    step used by the sensitivity drop.
    """
    grid = _as_grid(surface, cfg)
    opt = _opt_index(grid)
    peak = float(grid[opt])
    threshold = peak - delta_sharpe

    widths, sens, curv = [], [], []
    for ax in range(cfg.dim):
        line = _axis_line(grid, opt, ax)
        widths.append(connected_width(line, opt[ax], threshold))
        sens.append(local_sensitivity(line, opt[ax], step_frac))
        curv.append(local_curvature(line, opt[ax]))

    weights = fanova_weights(grid)
    widths_arr = np.asarray(widths)
    log_score = float(np.sum(weights * np.log(np.maximum(widths_arr, eps))))
    robustness_score = float(np.exp(log_score))

    # cheap extra diagnostic from the article: are the best points outliers?
    flat = surface[np.isfinite(surface)]
    top = np.sort(flat)[::-1]
    top3 = float(np.mean(top[:3])) if top.size >= 3 else float(top.max())
    med = float(np.median(flat))
    outlier_ratio = top3 / med if med > 1e-6 else float("inf")

    return {
        "opt_index": opt,
        "peak_is_sharpe": peak,
        "robustness_score": robustness_score,
        "plateau_width": float(np.dot(weights, widths_arr)),   # importance-weighted mean width
        "min_plateau_width": float(widths_arr.min()),
        "sensitivity": float(np.dot(weights, np.asarray(sens))),
        "max_sensitivity": float(np.max(sens)),
        "curvature": float(np.dot(weights, np.asarray(curv))),
        "outlier_ratio": outlier_ratio,
        "per_axis": {
            "weights": weights.tolist(),
            "widths": widths,
            "sensitivity": sens,
            "curvature": curv,
        },
    }
