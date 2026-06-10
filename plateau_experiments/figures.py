"""Generate the paper's figures (vector PDF) from the saved results.

    python -m plateau_experiments.figures      # writes paper/figures/*.pdf

Reads results/records_dim1.csv, results/results.json and the canonical surfaces.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .surfaces import canonical_configs, sharpe, simulate_returns, true_sharpe_surface, make_grid
from .simulate import surrogate_surface

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGDIR = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "figure.dpi": 120, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_TRUE, C_SURR, C_PTS = "#1f3b73", "#c0392b", "#9aa6b2"


def fig_setup(path: Path) -> None:
    """Four canonical landscapes: true Sharpe, noisy IS samples, surrogate."""
    cfgs = canonical_configs()
    _, axis = make_grid(1, next(iter(cfgs.values())).grid_size)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    rng = np.random.default_rng(3)
    for ax, (name, cfg) in zip(axes, cfgs.items()):
        srt = true_sharpe_surface(cfg)
        r_is = simulate_returns(srt, cfg.t_is, cfg, rng)
        sr_is = sharpe(r_is, cfg.periods_per_year)
        surr = surrogate_surface(sr_is, cfg, 0.06)
        ax.scatter(axis, sr_is, s=8, color=C_PTS, alpha=0.6, label="noisy IS estimate", zorder=1)
        ax.plot(axis, srt, color=C_TRUE, lw=2, label="true Sharpe", zorder=3)
        ax.plot(axis, surr, color=C_SURR, lw=1.6, ls="--", label="surrogate", zorder=2)
        ax.axvline(axis[int(np.argmax(sr_is))], color=C_PTS, lw=1, ls=":")
        ax.set_title(f"{name}")
        ax.set_xlabel(r"hyperparameter $\theta$")
    axes[0].set_ylabel("annualized Sharpe")
    axes[0].legend(loc="lower center", fontsize=6.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_auc_heatmap(path: Path, summary: dict) -> None:
    """AUC of each diagnostic for each overfitting mode (dim=1)."""
    auc = pd.DataFrame(summary["auc"]).set_index("diagnostic")
    order = ["robustness_score", "plateau_width", "curvature", "sensitivity",
             "outlier_gap", "pbo", "dsr", "psr0"]
    cols = ["auc_no_edge", "auc_fragile", "auc_oos_loss"]
    M = auc.loc[order, cols].to_numpy()
    fig, ax = plt.subplots(figsize=(4.4, 4.3))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.30, vmax=0.70, aspect="auto")
    ax.set_xticks(range(len(cols)), ["no-edge", "fragile", "OOS-loss"])
    ax.set_yticks(range(len(order)),
                  ["robustness R", "plateau width", "curvature", "sensitivity",
                   "outlier gap", "PBO", "DSR", "PSR"])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(M[i, j] - 0.5) > 0.12 else "black", fontsize=8)
    ax.axhline(4.5, color="k", lw=1.2)  # separate geometry diagnostics from statistical
    ax.set_title("Overfitting-mode detection (AUC)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ROC AUC")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_complementarity(path: Path, summary: dict) -> None:
    comb = pd.DataFrame(summary["combined_auc"])
    ci = summary.get("auc_ci", {})
    labels = ["no_edge", "fragile", "oos_loss"]
    feats = ["plateau (R)", "PBO", "DSR", "PSR", "plateau+stat"]
    colors = ["#9aa6b2", "#e0a458", "#c0392b", "#7a4f9e", "#1f3b73"]
    x = np.arange(len(labels)); w = 0.16
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for k, (f, c) in enumerate(zip(feats, colors)):
        vals, errs = [], [[], []]
        for lb in labels:
            v = comb[(comb.label == lb) & (comb.features == f)]["auc"].values[0]
            vals.append(v)
            e = ci.get(lb, {}).get("auc", {}).get(f"combined:{f}")
            errs[0].append(v - e["lo"] if e else 0.0)
            errs[1].append(e["hi"] - v if e else 0.0)
        ax.bar(x + (k - 2) * w, vals, w, label=f, color=c,
               yerr=np.array(errs), error_kw={"lw": 0.8}, capsize=2)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(x, ["no-edge", "fragile", "OOS-loss"])
    ax.set_ylim(0.45, 0.88)
    ax.set_ylabel("cross-validated AUC")
    ax.set_title("Single diagnostics vs. a combined geometry+statistics classifier")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_selection(path: Path, summary: dict, df1: pd.DataFrame) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    # left: OOS gain by curvature tercile
    q1, q2 = df1["curvature"].quantile([1 / 3, 2 / 3])
    terc = pd.cut(df1["curvature"], [-np.inf, q1, q2, np.inf], labels=["low", "mid", "high"])
    gain = df1["sr_oos_robust"] - df1["sr_oos_naive"]
    means = gain.groupby(terc, observed=True).mean().reindex(["low", "mid", "high"])
    sems = gain.groupby(terc, observed=True).sem().reindex(["low", "mid", "high"])
    axL.bar(["low", "mid", "high"], means.values, yerr=sems.values, color=["#9aa6b2", "#e0a458", "#c0392b"], capsize=3)
    axL.axhline(0, color="k", lw=0.8)
    axL.set_xlabel("surrogate curvature tercile (sharpness at the surrogate optimum)")
    axL.set_ylabel(r"OOS Sharpe gain: robust $-$ naive")
    axL.set_title("Plateau-aware selection helps where the\noptimum is a sharp peak")
    # right: bandwidth sweep
    sw = pd.DataFrame(summary["bandwidth_sweep"])
    axR.plot(sw["bandwidth_frac"], sw["mean_oos_gain"], "o-", color=C_TRUE, label="all landscapes")
    axR.plot(sw["bandwidth_frac"], sw["mean_oos_gain_high_curvature"], "s--", color=C_SURR, label="sharp-peak tercile")
    axR.axhline(0, color="k", lw=0.8)
    axR.set_xlabel("surrogate bandwidth (fraction of axis)")
    axR.set_ylabel("mean OOS Sharpe gain")
    axR.set_title("Benefit peaks at a moderate bandwidth")
    axR.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    summary = json.loads((RESULTS / "results.json").read_text())
    df1 = pd.read_csv(RESULTS / "records_dim1.csv")
    d1 = summary["dim1"]
    d1["bandwidth_sweep"] = summary["bandwidth_sweep"]["dim1"]
    fig_setup(FIGDIR / "fig_setup.pdf")
    fig_auc_heatmap(FIGDIR / "fig_auc.pdf", d1)
    fig_complementarity(FIGDIR / "fig_complementarity.pdf", d1)
    fig_selection(FIGDIR / "fig_selection.pdf", d1, df1)
    print(f"wrote 4 figures to {FIGDIR}")


if __name__ == "__main__":
    main()
