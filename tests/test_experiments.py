"""Fast sanity tests for the simulation, metrics and baselines.

Run: python -m pytest -q   (from the project root)
"""

from __future__ import annotations

import numpy as np
import pytest

from plateau_experiments import (
    canonical_configs,
    deflated_sharpe,
    pbo_cscv,
    plateau_metrics,
    run_experiment,
    true_sharpe_surface,
)
from plateau_experiments.surfaces import sharpe, simulate_returns


def test_sharpe_recovers_true_surface():
    """With a huge sample, the estimated surface matches the population one."""
    cfg = canonical_configs()["plateau"]
    sr_true = true_sharpe_surface(cfg)
    r = simulate_returns(sr_true, 200_000, cfg, np.random.default_rng(0))
    assert np.nanmax(np.abs(sharpe(r) - sr_true)) < 0.1


def test_factor_share_sets_cross_correlation():
    cfg = canonical_configs()["plateau"]
    cfg = type(cfg)(**{**cfg.__dict__, "factor_share": 0.4})
    r = simulate_returns(true_sharpe_surface(cfg), 50_000, cfg, np.random.default_rng(1))
    corr = np.corrcoef(r.T)
    off = corr[np.triu_indices_from(corr, k=1)]
    assert abs(np.mean(off) - 0.4) < 0.05  # average pairwise corr ~ factor_share


def test_plateau_wider_than_needle():
    """A broad surrogate plateau scores more robust than a sharp needle."""
    cfgs = canonical_configs()
    plat = np.mean([run_experiment(cfgs["plateau"], np.random.default_rng(s))["plateau_width"] for s in range(20)])
    need = np.mean([run_experiment(cfgs["needle"], np.random.default_rng(s))["plateau_width"] for s in range(20)])
    assert plat > need


def test_needle_sharper_curvature():
    cfgs = canonical_configs()
    plat = np.mean([run_experiment(cfgs["plateau"], np.random.default_rng(s))["curvature"] for s in range(20)])
    need = np.mean([run_experiment(cfgs["needle"], np.random.default_rng(s))["curvature"] for s in range(20)])
    assert need > plat


def test_pbo_and_dsr_in_unit_interval():
    cfg = canonical_configs()["null"]
    r = simulate_returns(true_sharpe_surface(cfg), cfg.t_is, cfg, np.random.default_rng(2))
    pbo = pbo_cscv(r, s_blocks=8)["pbo"]
    is_sr = sharpe(r) / np.sqrt(cfg.periods_per_year)
    dsr = deflated_sharpe(r[:, int(np.nanargmax(is_sr))], is_sr)["dsr"]
    assert 0.0 <= pbo <= 1.0
    assert 0.0 <= dsr <= 1.0


def test_dsr_null_below_plateau():
    """No-edge (null) surfaces should deflate to a lower DSR than a real plateau."""
    cfgs = canonical_configs()
    dsr_null = np.mean([run_experiment(cfgs["null"], np.random.default_rng(s))["dsr"] for s in range(20)])
    dsr_plat = np.mean([run_experiment(cfgs["plateau"], np.random.default_rng(s))["dsr"] for s in range(20)])
    assert dsr_null < dsr_plat


def test_run_experiment_is_deterministic():
    cfg = canonical_configs()["mixed"]
    a = run_experiment(cfg, np.random.default_rng(123))
    b = run_experiment(cfg, np.random.default_rng(123))
    assert a["sr_oos_naive"] == b["sr_oos_naive"]
    assert a["robustness_score"] == b["robustness_score"]


def test_metric_bundle_keys():
    cfg = canonical_configs()["plateau"]
    sr_true = true_sharpe_surface(cfg)
    r = simulate_returns(sr_true, cfg.t_is, cfg, np.random.default_rng(0))
    m = plateau_metrics(sharpe(r), cfg)
    for k in ("robustness_score", "plateau_width", "sensitivity", "curvature", "outlier_ratio"):
        assert k in m and np.isfinite(m[k])
