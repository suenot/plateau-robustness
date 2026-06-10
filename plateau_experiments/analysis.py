"""Turn a batch of experiment records into the paper's quantitative results.

Three questions:
  1. Do the diagnostics *rank* overfitting (Spearman vs true regret / IS->OOS gap)?
  2. How well does each *classify* the two overfitting modes (sharp-peak fragility
     vs no-edge), measured by ROC AUC? Do plateau-geometry and statistical
     diagnostics specialize in different modes, and does combining them beat any
     single one (complementarity)?
  3. Does selecting on the plateau-aware surrogate beat naive argmax out of sample?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# diagnostic -> (+1 if larger means MORE overfitting risk, else -1)
# outlier_gap (top-3 minus median of the surrogate): a LARGE gap means the best
# cells stand far above typical performance, i.e. some real edge exists, so a
# SMALL gap signals the no-edge overfitting mode -> orientation -1.
RISK_ORIENTATION = {
    "robustness_score": -1,
    "plateau_width": -1,
    "min_plateau_width": -1,
    "curvature": +1,
    "sensitivity": +1,
    "max_sensitivity": +1,
    "outlier_gap": -1,
    "pbo": +1,
    "dsr": -1,
    "psr0": -1,
    # geometry anchored at the naive (raw argmax) point instead of the
    # surrogate argmax -- anchoring-sensitivity check, same signs as above
    "robustness_score_naive_anchor": -1,
    "plateau_width_naive_anchor": -1,
    "sensitivity_naive_anchor": +1,
    "curvature_naive_anchor": +1,
}
PLATEAU_DIAGS = ["robustness_score", "plateau_width", "curvature", "sensitivity", "outlier_gap"]
STAT_DIAGS = ["pbo", "dsr", "psr0"]

# binary overfitting-mode labels
LABELS = {
    "no_edge": lambda df: (df["sr_true_naive"] < 0.25).astype(int),      # deployed ~no real edge
    "fragile": lambda df: (df["true_regret"] > 0.5).astype(int),         # left real edge on the table
    "oos_loss": lambda df: (df["sr_oos_naive"] < 0.0).astype(int),       # realized OOS loser
}
TARGETS = ["is_oos_gap", "true_regret"]  # continuous overfitting severity


def to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _risk(df: pd.DataFrame, diag: str) -> np.ndarray:
    return RISK_ORIENTATION[diag] * df[diag].to_numpy(dtype=float)


def spearman_table(df: pd.DataFrame, diags=None, targets=TARGETS) -> pd.DataFrame:
    diags = diags or list(RISK_ORIENTATION)
    rows = []
    for d in diags:
        row = {"diagnostic": d}
        for t in targets:
            ok = np.isfinite(df[d]) & np.isfinite(df[t])
            rho, p = stats.spearmanr(_risk(df[ok], d), df.loc[ok, t])
            row[f"rho_{t}"] = rho
            row[f"p_{t}"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def auc_table(df: pd.DataFrame, diags=None, labels=LABELS) -> pd.DataFrame:
    diags = diags or list(RISK_ORIENTATION)
    label_cols = {name: fn(df) for name, fn in labels.items()}
    rows = []
    for d in diags:
        row = {"diagnostic": d}
        score = _risk(df, d)
        for name, y in label_cols.items():
            ok = np.isfinite(score) & np.isfinite(y)
            yv = y[ok].to_numpy()
            row[f"auc_{name}"] = roc_auc_score(yv, score[ok]) if 0 < yv.sum() < yv.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows), {k: int(v.sum()) for k, v in label_cols.items()}, len(df)


def combined_auc(df: pd.DataFrame, feature_sets: dict[str, list[str]], labels=LABELS, seed: int = 0) -> pd.DataFrame:
    """Cross-validated logistic-regression AUC for each feature set / label, to
    test whether combining plateau + statistical diagnostics beats either alone.
    Scaling is done inside a Pipeline so each CV fold fits its own scaler (no
    train/test leakage through the standardization)."""
    rows = []
    for label_name, fn in labels.items():
        y = fn(df).to_numpy()
        if not (0 < y.sum() < y.size):
            continue
        for set_name, feats in feature_sets.items():
            X = np.column_stack([_risk(df, f) for f in feats])
            ok = np.all(np.isfinite(X), axis=1)
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            proba = cross_val_predict(clf, X[ok], y[ok], cv=5, method="predict_proba")[:, 1]
            rows.append({"label": label_name, "features": set_name,
                         "auc": roc_auc_score(y[ok], proba)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# bootstrap confidence intervals for the AUCs
# --------------------------------------------------------------------------- #
def _fast_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Mann-Whitney AUC via average ranks (ties handled; == roc_auc_score)."""
    ranks = stats.rankdata(score)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_auc_cis(
    df: pd.DataFrame,
    diags: list[str] | None = None,
    feature_sets: dict[str, list[str]] | None = None,
    labels=LABELS,
    n_boot: int = 1000,
    seed: int = 7,
    combined_name: str = "plateau+stat",
) -> dict:
    """95% bootstrap CIs (percentile, >=1000 resamples of experiment records)
    for (a) each single-diagnostic AUC, (b) each combined-classifier AUC, and
    (c) the paired difference combined-minus-single for every feature set.

    For the combined classifiers the cross-validated out-of-fold probabilities
    are computed once on the full sample and the (y, score) pairs are
    bootstrap-resampled -- the standard resampling scheme for CIs on a
    cross-validated AUC. Paired differences reuse the same bootstrap indices,
    so the difference CI accounts for the correlation between classifiers.
    """
    diags = diags or [d for d in RISK_ORIENTATION if not d.endswith("_naive_anchor")]
    feature_sets = feature_sets or {}
    rng = np.random.default_rng(seed)
    out: dict = {}
    for label_name, fn in labels.items():
        y_all = fn(df).to_numpy()
        # score vector per scorer (single diagnostics + combined classifiers)
        scores: dict[str, np.ndarray] = {d: _risk(df, d) for d in diags}
        for set_name, feats in feature_sets.items():
            X = np.column_stack([_risk(df, f) for f in feats])
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            scores[f"combined:{set_name}"] = cross_val_predict(
                clf, X, y_all, cv=5, method="predict_proba")[:, 1]
        n = len(df)
        boot_auc = {k: np.empty(n_boot) for k in scores}
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y_all[idx]
            if not (0 < yb.sum() < yb.size):  # degenerate resample: redraw
                idx = rng.integers(0, n, size=n)
                yb = y_all[idx]
            for k, s in scores.items():
                boot_auc[k][b] = _fast_auc(yb, s[idx])
        entry: dict = {"auc": {}, "combined_minus_single": {}}
        for k, s in scores.items():
            lo, hi = np.nanpercentile(boot_auc[k], [2.5, 97.5])
            entry["auc"][k] = {"auc": _fast_auc(y_all, s), "lo": float(lo), "hi": float(hi)}
        comb_key = f"combined:{combined_name}"
        if comb_key in boot_auc:
            for set_name in feature_sets:
                if set_name == combined_name:
                    continue
                d = boot_auc[comb_key] - boot_auc[f"combined:{set_name}"]
                lo, hi = np.nanpercentile(d, [2.5, 97.5])
                entry["combined_minus_single"][set_name] = {
                    "diff": float(np.nanmean(d)), "lo": float(lo), "hi": float(hi),
                    "significant": bool(lo > 0 or hi < 0),
                }
        out[label_name] = entry
    return out


def diagnostic_correlations(df: pd.DataFrame, diags=None) -> pd.DataFrame:
    """Spearman correlation among the risk-oriented diagnostics (low cross-family
    correlation => they carry complementary information)."""
    diags = diags or (PLATEAU_DIAGS + STAT_DIAGS)
    mat = np.column_stack([_risk(df, d) for d in diags])
    rho, _ = stats.spearmanr(mat, nan_policy="omit")
    return pd.DataFrame(rho, index=diags, columns=diags)


def selection_payoff(df: pd.DataFrame) -> dict:
    """Plateau-aware (surrogate-argmax) vs naive selection, out of sample."""
    diff = df["sr_oos_robust"] - df["sr_oos_naive"]
    reg = df["true_regret"] - df["true_regret_robust"]
    nonties = diff[np.isfinite(diff) & (diff != 0)]
    res = stats.wilcoxon(nonties)
    out = {
        "mean_oos_naive": float(df["sr_oos_naive"].mean()),
        "mean_oos_robust": float(df["sr_oos_robust"].mean()),
        "mean_oos_gain": float(diff.mean()),
        "median_oos_gain": float(diff.median()),
        "win_rate": float((diff > 0).mean()),
        "tie_share": float((diff == 0).mean()),       # both rules picked the same cell
        "win_rate_nonties": float((nonties > 0).mean()),
        "median_oos_gain_nonties": float(nonties.median()),
        "wilcoxon_p": float(res.pvalue),
        "mean_regret_reduction": float(reg.mean()),
    }
    # where it helps most: condition on the sharp-peak (high-curvature) tercile
    q1, q2 = df["curvature"].quantile([1 / 3, 2 / 3])
    for name, mask in (("low", df["curvature"] <= q1),
                       ("mid", (df["curvature"] > q1) & (df["curvature"] < q2)),
                       ("high", df["curvature"] >= q2)):
        g = (df.loc[mask, "sr_oos_robust"] - df.loc[mask, "sr_oos_naive"])
        t = stats.ttest_1samp(g, 0.0)
        out[f"oos_gain_{name}_curvature"] = float(g.mean())
        out[f"oos_gain_{name}_curvature_p"] = float(t.pvalue)
    return out


def threshold_sensitivity(
    df: pd.DataFrame,
    diags: list[str] | None = None,
    no_edge_thresholds: tuple[float, ...] = (0.15, 0.25, 0.35),
    fragile_thresholds: tuple[float, ...] = (0.35, 0.5, 0.65),
) -> dict:
    """Re-score the main AUC table under perturbed label thresholds (no new
    simulations -- the labels are deterministic functions of stored columns)."""
    diags = diags or ["robustness_score", "plateau_width", "curvature",
                      "sensitivity", "outlier_gap", "pbo", "dsr", "psr0"]
    out = {"no_edge": [], "fragile": []}
    for th in no_edge_thresholds:
        labels = {"no_edge": lambda d, th=th: (d["sr_true_naive"] < th).astype(int)}
        auc, counts, n = auc_table(df, diags=diags, labels=labels)
        out["no_edge"].append({"threshold": th, "base_rate": counts["no_edge"] / n,
                               "auc": auc.set_index("diagnostic")["auc_no_edge"].to_dict()})
    for th in fragile_thresholds:
        labels = {"fragile": lambda d, th=th: (d["true_regret"] > th).astype(int)}
        auc, counts, n = auc_table(df, diags=diags, labels=labels)
        out["fragile"].append({"threshold": th, "base_rate": counts["fragile"] / n,
                               "auc": auc.set_index("diagnostic")["auc_fragile"].to_dict()})
    return out


def _expected_max_factor(n: np.ndarray) -> np.ndarray:
    """Bailey-LdP expected-max multiplier g(n) with SR0 = dispersion * g(n)."""
    from .baselines import EULER_GAMMA
    from math import e as _E
    n = np.asarray(n, dtype=float)
    z1 = stats.norm.ppf(1.0 - 1.0 / n)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n * _E))
    return (1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2


def dsr_neff_check(df: pd.DataFrame, fracs: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)) -> list[dict]:
    """AUC of the DSR recomputed as if only ``frac * N`` of the grid trials were
    effectively independent (the correlated-trials caveat of Bailey-LdP).

    Reconstructed exactly from stored columns: with z0 = Phi^-1(psr0) and
    z = Phi^-1(dsr), the deflation term is sr0*k = z0 - z, and SR0 scales as
    g(n_eff)/g(N), so z' = z0 - (z0 - z) * g(n_eff)/g(N). Saturated
    probabilities are clipped at 1e-12 (immaterial for rank-based AUC).
    """
    eps = 1e-12
    z0 = stats.norm.ppf(np.clip(df["psr0"].to_numpy(), eps, 1 - eps))
    z = stats.norm.ppf(np.clip(df["dsr"].to_numpy(), eps, 1 - eps))
    n_full = df["n_strategies"].to_numpy()
    y_no_edge = LABELS["no_edge"](df).to_numpy()
    y_oos = LABELS["oos_loss"](df).to_numpy()
    rows = []
    for frac in fracs:
        n_eff = np.maximum(2, np.round(frac * n_full))
        scale = _expected_max_factor(n_eff) / _expected_max_factor(n_full)
        dsr_adj = stats.norm.cdf(z0 - (z0 - z) * scale)
        rows.append({
            "frac_independent": frac,
            "auc_no_edge": _fast_auc(y_no_edge, -dsr_adj),
            "auc_oos_loss": _fast_auc(y_oos, -dsr_adj),
        })
    return rows


def summarize(df: pd.DataFrame, n_boot: int = 1000) -> dict:
    """One call -> every table the paper reports, as JSON-able objects."""
    auc, label_counts, n = auc_table(df)
    feature_sets = {
        "plateau (R)": ["robustness_score"],
        "PBO": ["pbo"],
        "DSR": ["dsr"],
        "PSR": ["psr0"],
        "plateau+stat": ["robustness_score", "curvature", "pbo", "dsr"],
    }
    return {
        "n_experiments": int(n),
        "label_base_rates": {k: v / n for k, v in label_counts.items()},
        "spearman": spearman_table(df).to_dict(orient="records"),
        "auc": auc.to_dict(orient="records"),
        "combined_auc": combined_auc(df, feature_sets).to_dict(orient="records"),
        "auc_ci": bootstrap_auc_cis(df, feature_sets=feature_sets, n_boot=n_boot),
        "selection_payoff": selection_payoff(df),
        "label_threshold_sensitivity": threshold_sensitivity(df),
        "dsr_neff_check": dsr_neff_check(df),
        "diagnostic_correlations": diagnostic_correlations(df).round(3).to_dict(),
    }
