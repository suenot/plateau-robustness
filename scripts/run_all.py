"""Reproduce every number and figure input in the paper.

    python scripts/run_all.py            # full run -> results/results.json + record CSVs
    python scripts/run_all.py --quick    # small batch for a smoke check

Deterministic given the fixed seeds below. No wall-clock / randomness leaks into
results (timestamps are not used). Run from the project root.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plateau_experiments import analysis as A
from plateau_experiments import __version__
from plateau_experiments.simulate import run_batch, run_experiment
from plateau_experiments.surfaces import sample_surface_config

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def bandwidth_sweep(n: int, seed: int, bandwidths: list[float], dim: int = 1) -> list[dict]:
    """Mean OOS gain of plateau-aware selection over naive argmax, vs surrogate
    bandwidth — to show the selection benefit is not a knife-edge artifact."""
    out = []
    for bw in bandwidths:
        ss = np.random.SeedSequence(seed).spawn(n)
        recs = []
        for cs in ss:
            rng = np.random.default_rng(cs)
            cfg = sample_surface_config(rng, dim=dim)
            recs.append(run_experiment(cfg, rng, bandwidth_frac=bw))
        import pandas as pd
        df = pd.DataFrame(recs)
        diff = df["sr_oos_robust"] - df["sr_oos_naive"]
        hi = df["curvature"] >= df["curvature"].quantile(2 / 3)
        out.append({
            "bandwidth_frac": bw,
            "mean_oos_gain": float(diff.mean()),
            "win_rate": float((diff > 0).mean()),
            "mean_oos_gain_high_curvature": float((df.loc[hi, "sr_oos_robust"] - df.loc[hi, "sr_oos_naive"]).mean()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    n1 = 800 if args.quick else 5000
    n2 = 600 if args.quick else 4000
    RESULTS.mkdir(exist_ok=True)

    print(f"[1/4] dim=1 batch (n={n1}) ...", flush=True)
    rec1 = run_batch(n1, dim=1, seed=101, pbo_blocks=10, progress_every=n1 // 4)
    df1 = A.to_frame(rec1)
    df1.to_csv(RESULTS / "records_dim1.csv", index=False)

    print(f"[2/4] dim=2 batch (n={n2}) ...", flush=True)
    rec2 = run_batch(n2, dim=2, seed=202, pbo_blocks=10, progress_every=n2 // 4)
    df2 = A.to_frame(rec2)
    df2.to_csv(RESULTS / "records_dim2.csv", index=False)

    print("[3/4] bandwidth sensitivity sweeps (d=1, d=2) ...", flush=True)
    bandwidths = [0.03, 0.045, 0.06, 0.075, 0.09, 0.12, 0.16]
    sweep1 = bandwidth_sweep(1500 if not args.quick else 400, seed=303,
                             bandwidths=bandwidths, dim=1)
    sweep2 = bandwidth_sweep(600 if not args.quick else 200, seed=404,
                             bandwidths=bandwidths, dim=2)

    print("[4/4] summaries ...", flush=True)
    results = {
        "meta": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_dim1": n1, "n_dim2": n2,
            "seeds": {"dim1": 101, "dim2": 202, "sweep_dim1": 303, "sweep_dim2": 404},
            "notes": "Deterministic; reproduce with python scripts/run_all.py",
        },
        "dim1": A.summarize(df1),
        "dim2": A.summarize(df2),
        "bandwidth_sweep": {"dim1": sweep1, "dim2": sweep2},
    }
    (RESULTS / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {RESULTS/'results.json'} and record CSVs.")

    # headline lines to stdout
    for dname in ("dim1", "dim2"):
        d = results[dname]
        print(f"\n--- {dname} headline ---")
        print("AUC (no_edge / fragile / oos_loss):")
        for r in d["auc"]:
            if r["diagnostic"] in ("robustness_score", "curvature", "outlier_gap", "pbo", "dsr", "psr0"):
                print(f"  {r['diagnostic']:16} {r['auc_no_edge']:.3f} {r['auc_fragile']:.3f} {r['auc_oos_loss']:.3f}")
        print("combined AUC:", [(r["features"], round(r["auc"], 3)) for r in d["combined_auc"] if r["label"] == "no_edge"])
        sp = d["selection_payoff"]
        print(f"selection: naive {sp['mean_oos_naive']:.3f} -> robust {sp['mean_oos_robust']:.3f} "
              f"(gain {sp['mean_oos_gain']:+.3f}, high-curv {sp['oos_gain_high_curvature']:+.3f})")


if __name__ == "__main__":
    main()
