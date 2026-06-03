"""Established backtest-overfitting diagnostics used as comparators.

- **PBO via CSCV** — Combinatorially-Symmetric Cross-Validation estimate of the
  Probability of Backtest Overfitting (Bailey, Borwein, Lopez de Prado, Zhu,
  *The Probability of Backtest Overfitting*, J. Computational Finance, 2017).
- **PSR / DSR** — Probabilistic and Deflated Sharpe Ratios (Bailey &
  Lopez de Prado, *The Sharpe Ratio Efficient Frontier*, 2012, and *The Deflated
  Sharpe Ratio*, 2014).

Both operate on the same simulated return matrix as the plateau metrics, so the
comparison is apples-to-apples. Implementations are vectorized; CSCV uses
per-block sufficient statistics so each of the C(S, S/2) splits is O(S*N).
"""

from __future__ import annotations

from itertools import combinations
from math import e as _E

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------- #
# PBO via CSCV
# --------------------------------------------------------------------------- #
def _block_stats(returns: np.ndarray, s_blocks: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split ``returns`` (T, N) into ``s_blocks`` equal row blocks (trimming the
    remainder) and return per-block (sum, sumsq, count) each shaped (S, N)."""
    t = returns.shape[0]
    block_len = t // s_blocks
    usable = block_len * s_blocks
    r = returns[:usable]
    blocks = r.reshape(s_blocks, block_len, -1)
    bsum = blocks.sum(axis=1)
    bsq = (blocks**2).sum(axis=1)
    bcount = np.full(s_blocks, block_len, dtype=float)
    return bsum, bsq, bcount


def _sharpe_from_stats(sums: np.ndarray, sqs: np.ndarray, count: float) -> np.ndarray:
    """Per-observation Sharpe (mean/std, ddof=1) for each strategy from pooled
    sufficient statistics."""
    mean = sums / count
    var = (sqs - sums**2 / count) / (count - 1)
    var = np.where(var <= 1e-18, np.nan, var)
    return mean / np.sqrt(var)


def pbo_cscv(returns: np.ndarray, s_blocks: int = 10) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    ``returns`` is (T, N): T observations, N candidate strategies. Returns a dict
    with ``pbo`` in [0, 1] (probability the in-sample-best strategy lands below
    the OOS median) plus the mean/median of the logit distribution.
    """
    n_strats = returns.shape[1]
    if n_strats < 2:
        return {"pbo": float("nan"), "logit_mean": float("nan"), "n_splits": 0}
    if s_blocks % 2 == 1:
        s_blocks += 1
    bsum, bsq, bcount = _block_stats(returns, s_blocks)

    block_ids = range(s_blocks)
    logits = []
    half = s_blocks // 2
    for train in combinations(block_ids, half):
        train = list(train)
        test = [b for b in block_ids if b not in train]

        tr_sharpe = _sharpe_from_stats(bsum[train].sum(0), bsq[train].sum(0), bcount[train].sum())
        te_sharpe = _sharpe_from_stats(bsum[test].sum(0), bsq[test].sum(0), bcount[test].sum())

        n_star = int(np.nanargmax(tr_sharpe))
        # relative OOS rank of the IS-best (1=worst .. N=best) -> omega in (0,1)
        finite = np.isfinite(te_sharpe)
        rank = int(np.sum(te_sharpe[finite] <= te_sharpe[n_star]))  # # strategies it beats or ties
        n_eff = int(finite.sum())
        omega = rank / (n_eff + 1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        logits.append(np.log(omega / (1 - omega)))

    logits = np.asarray(logits)
    return {
        "pbo": float(np.mean(logits <= 0)),
        "logit_mean": float(np.mean(logits)),
        "logit_median": float(np.median(logits)),
        "n_splits": int(logits.size),
    }


# --------------------------------------------------------------------------- #
# Probabilistic / Deflated Sharpe Ratio
# --------------------------------------------------------------------------- #
def _moments(returns_1d: np.ndarray) -> tuple[float, float, float]:
    """Per-observation Sharpe, skewness, (non-excess) kurtosis of one series."""
    r = returns_1d[np.isfinite(returns_1d)]
    mu = r.mean()
    sd = r.std(ddof=1)
    sr = mu / sd if sd > 1e-18 else 0.0
    skew = float(stats.skew(r, bias=False))
    kurt = float(stats.kurtosis(r, fisher=False, bias=False))  # 3 for normal
    return float(sr), skew, kurt


def probabilistic_sharpe(returns_1d: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """PSR: probability the true (per-observation) Sharpe exceeds ``sr_benchmark``."""
    sr, skew, kurt = _moments(returns_1d)
    t = int(np.isfinite(returns_1d).sum())
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2))
    z = (sr - sr_benchmark) * np.sqrt(t - 1) / denom
    return float(stats.norm.cdf(z))


def expected_max_sharpe(sharpe_dispersion: float, n_trials: int) -> float:
    """E[max] of N i.i.d. zero-mean Sharpe estimates with given cross-sectional
    standard deviation (the DSR null benchmark SR0)."""
    if n_trials < 2 or sharpe_dispersion <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * _E))
    return float(sharpe_dispersion * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(
    selected_returns: np.ndarray,
    all_is_sharpes: np.ndarray,
    n_trials: int | None = None,
) -> dict:
    """Deflated Sharpe Ratio of the selected strategy.

    ``all_is_sharpes`` are the per-observation IS Sharpe estimates across *all*
    candidate strategies (used for the cross-sectional dispersion that sets the
    multiple-testing benchmark SR0). DSR in [0, 1]; low => the selected Sharpe is
    not significant after deflating for trials + non-normality (overfit risk).
    """
    s = all_is_sharpes[np.isfinite(all_is_sharpes)]
    n = int(n_trials if n_trials is not None else s.size)
    sr0 = expected_max_sharpe(float(np.std(s, ddof=1)) if s.size > 1 else 0.0, n)
    dsr = probabilistic_sharpe(selected_returns, sr_benchmark=sr0)
    return {"dsr": dsr, "sr0": sr0, "psr0": probabilistic_sharpe(selected_returns, 0.0)}
