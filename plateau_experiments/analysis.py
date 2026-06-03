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
from sklearn.preprocessing import StandardScaler

# diagnostic -> (+1 if larger means MORE overfitting risk, else -1)
RISK_ORIENTATION = {
    "robustness_score": -1,
    "plateau_width": -1,
    "min_plateau_width": -1,
    "curvature": +1,
    "sensitivity": +1,
    "max_sensitivity": +1,
    "outlier_ratio": +1,
    "pbo": +1,
    "dsr": -1,
    "psr0": -1,
}
PLATEAU_DIAGS = ["robustness_score", "plateau_width", "curvature", "sensitivity", "outlier_ratio"]
STAT_DIAGS = ["pbo", "dsr"]

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
    test whether combining plateau + statistical diagnostics beats either alone."""
    rows = []
    for label_name, fn in labels.items():
        y = fn(df).to_numpy()
        if not (0 < y.sum() < y.size):
            continue
        for set_name, feats in feature_sets.items():
            X = np.column_stack([_risk(df, f) for f in feats])
            ok = np.all(np.isfinite(X), axis=1)
            Xs = StandardScaler().fit_transform(X[ok])
            clf = LogisticRegression(max_iter=1000)
            proba = cross_val_predict(clf, Xs, y[ok], cv=5, method="predict_proba")[:, 1]
            rows.append({"label": label_name, "features": set_name,
                         "auc": roc_auc_score(y[ok], proba)})
    return pd.DataFrame(rows)


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
    res = stats.wilcoxon(diff[np.isfinite(diff) & (diff != 0)])
    out = {
        "mean_oos_naive": float(df["sr_oos_naive"].mean()),
        "mean_oos_robust": float(df["sr_oos_robust"].mean()),
        "mean_oos_gain": float(diff.mean()),
        "median_oos_gain": float(diff.median()),
        "win_rate": float((diff > 0).mean()),
        "wilcoxon_p": float(res.pvalue),
        "mean_regret_reduction": float(reg.mean()),
    }
    # where it helps most: condition on the sharp-peak (high-curvature) tercile
    hi = df["curvature"] >= df["curvature"].quantile(2 / 3)
    out["oos_gain_high_curvature"] = float((df.loc[hi, "sr_oos_robust"] - df.loc[hi, "sr_oos_naive"]).mean())
    lo = df["curvature"] <= df["curvature"].quantile(1 / 3)
    out["oos_gain_low_curvature"] = float((df.loc[lo, "sr_oos_robust"] - df.loc[lo, "sr_oos_naive"]).mean())
    return out


def summarize(df: pd.DataFrame) -> dict:
    """One call -> every table the paper reports, as JSON-able objects."""
    auc, label_counts, n = auc_table(df)
    feature_sets = {
        "plateau (R)": ["robustness_score"],
        "PBO": ["pbo"],
        "DSR": ["dsr"],
        "plateau+stat": ["robustness_score", "curvature", "pbo", "dsr"],
    }
    return {
        "n_experiments": int(n),
        "label_base_rates": {k: v / n for k, v in label_counts.items()},
        "spearman": spearman_table(df).to_dict(orient="records"),
        "auc": auc.to_dict(orient="records"),
        "combined_auc": combined_auc(df, feature_sets).to_dict(orient="records"),
        "selection_payoff": selection_payoff(df),
        "diagnostic_correlations": diagnostic_correlations(df).round(3).to_dict(),
    }
