"""Verify that every number in paper/main.tex's tables (and the headline inline
claims) matches results/results.json.

    python scripts/check_paper_numbers.py            # assert mode (CI-style)
    python scripts/check_paper_numbers.py --print    # print the expected LaTeX rows

The paper's rounding convention: AUCs and Sharpe values to 3 decimals in tables,
2 decimals inline. This script recomputes every formatted string from
results.json and (a) re-parses the two tables in main.tex cell by cell, (b)
asserts each inline headline string occurs in the source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
RESULTS = ROOT / "results" / "results.json"

# Table 1 rows: LaTeX row label fragment -> diagnostic key in results.json
TAB1_ROWS = [
    ("Robustness score", "robustness_score"),
    ("Plateau width", "plateau_width"),
    ("Curvature", "curvature"),
    ("Outlier gap", "outlier_gap"),
    ("PBO", "pbo"),
    ("DSR", "dsr"),
    ("PSR", "psr0"),
]
TAB1_COLS = ["auc_no_edge", "auc_fragile", "auc_oos_loss"]


def f3(x: float) -> str:
    return f"{x:.3f}"


def f2(x: float) -> str:
    return f"{x:.2f}"


def table_block(tex: str, label: str) -> str:
    m = re.search(r"\\label\{" + re.escape(label) + r"\}.*?\\end\{table", tex, re.S)
    if not m:
        sys.exit(f"FAIL: table {label} not found in main.tex")
    return m.group(0)


def row_numbers(block: str, row_label: str) -> list[str]:
    for line in block.splitlines():
        clean = line.replace(r"\textbf{", "").replace("}", "").replace("$", "")
        if clean.lstrip().startswith(row_label):
            return [n.lstrip("+") for n in re.findall(r"[-+]?\d+\.\d+", clean)]
    sys.exit(f"FAIL: row '{row_label}' not found")


def auc_map(summary: dict) -> dict:
    return {r["diagnostic"]: r for r in summary["auc"]}


def check_table1(tex: str, res: dict, errors: list[str]) -> None:
    block = table_block(tex, "tab:auc")
    a1, a2 = auc_map(res["dim1"]), auc_map(res["dim2"])
    for row_label, diag in TAB1_ROWS:
        got = row_numbers(block, row_label)
        want = [f3(a1[diag][c]) for c in TAB1_COLS] + [f3(a2[diag][c]) for c in TAB1_COLS]
        if got != want:
            errors.append(f"tab:auc row '{row_label}': tex {got} != results {want}")


def check_table2(tex: str, res: dict, errors: list[str]) -> None:
    block = table_block(tex, "tab:selection")
    sp1 = res["dim1"]["selection_payoff"]
    sp2 = res["dim2"]["selection_payoff"]
    rows = [
        ("Naive argmax", [f3(sp1["mean_oos_naive"]), f3(sp2["mean_oos_naive"])]),
        ("Plateau-aware", [f3(sp1["mean_oos_robust"]), f3(sp2["mean_oos_robust"])]),
        ("Mean gain", [f3(sp1["mean_oos_gain"]), f3(sp2["mean_oos_gain"])]),
        ("Median gain", [f3(sp1["median_oos_gain_nonties"]), f3(sp2["median_oos_gain_nonties"])]),
        ("Win rate", [f2(sp1["win_rate"]), f2(sp1["tie_share"]), f2(sp1["win_rate_nonties"]),
                      f2(sp2["win_rate"]), f2(sp2["tie_share"]), f2(sp2["win_rate_nonties"])]),
        ("Gain, low-curvature tercile",
         [f3(sp1["oos_gain_low_curvature"]), f3(sp2["oos_gain_low_curvature"])]),
        ("Gain, mid-curvature tercile",
         [f3(sp1["oos_gain_mid_curvature"]), f3(sp2["oos_gain_mid_curvature"])]),
        ("Gain, high-curvature tercile",
         [f3(sp1["oos_gain_high_curvature"]), f3(sp2["oos_gain_high_curvature"])]),
    ]
    for row_label, want in rows:
        got = row_numbers(block, row_label)
        if got != want:
            errors.append(f"tab:selection row '{row_label}': tex {got} != results {want}")


def inline_assertions(res: dict) -> list[tuple[str, str]]:
    """(description, exact string that must appear in main.tex)."""
    d1, d2 = res["dim1"], res["dim2"]
    a1, a2 = auc_map(d1), auc_map(d2)
    sp1, sp2 = d1["selection_payoff"], d2["selection_payoff"]
    ci1 = d1["auc_ci"]
    sw1 = {s["bandwidth_frac"]: s for s in res["bandwidth_sweep"]["dim1"]}
    sw2 = {s["bandwidth_frac"]: s for s in res["bandwidth_sweep"]["dim2"]}
    comb1 = {(r["label"], r["features"]): r["auc"] for r in d1["combined_auc"]}
    comb2 = {(r["label"], r["features"]): r["auc"] for r in d2["combined_auc"]}
    out = [
        ("abstract: d1 mean gain", f2(sp1["mean_oos_gain"])),
        ("abstract: d2 mean gain", f2(sp2["mean_oos_gain"])),
        ("d1 PSR no-edge AUC", f3(a1["psr0"]["auc_no_edge"])),
        ("d1 DSR no-edge AUC", f3(a1["dsr"]["auc_no_edge"])),
        ("d1 R no-edge AUC", f3(a1["robustness_score"]["auc_no_edge"])),
        ("d2 R no-edge AUC", f3(a2["robustness_score"]["auc_no_edge"])),
        ("d1 combined no-edge AUC", f3(comb1[("no_edge", "plateau+stat")])),
        ("d2 combined no-edge AUC", f3(comb2[("no_edge", "plateau+stat")])),
        ("d1 naive-anchor R no-edge AUC", f3(a1["robustness_score_naive_anchor"]["auc_no_edge"])),
        ("d2 naive-anchor R no-edge AUC", f3(a2["robustness_score_naive_anchor"]["auc_no_edge"])),
        ("d1 selection means", f2(sp1["mean_oos_naive"])),
        ("d1 selection means (robust)", f2(sp1["mean_oos_robust"])),
        ("d2 selection means", f2(sp2["mean_oos_naive"])),
        ("d2 selection means (robust)", f2(sp2["mean_oos_robust"])),
        ("d1 high-curv tercile gain", f3(sp1["oos_gain_high_curvature"])),
        ("d2 high-curv tercile gain", f3(sp2["oos_gain_high_curvature"])),
        ("d1 low-curv tercile gain", f3(sp1["oos_gain_low_curvature"])),
        ("d1 low-curv tercile p", f3(sp1["oos_gain_low_curvature_p"])),
        ("d1 outlier-gap no-edge AUC", f3(a1["outlier_gap"]["auc_no_edge"])),
        ("d2 outlier-gap no-edge AUC", f3(a2["outlier_gap"]["auc_no_edge"])),
        ("d2 curvature fragile AUC", f3(a2["curvature"]["auc_fragile"])),
        ("d2 combined fragile AUC", f3(comb2[("fragile", "plateau+stat")])),
        ("d1 PBO no-edge AUC", f3(a1["pbo"]["auc_no_edge"])),
        ("d1 DSR neff-0.1 no-edge AUC", f3(d1["dsr_neff_check"][-1]["auc_no_edge"])),
        ("d1 sweep gain at 0.16", f3(sw1[0.16]["mean_oos_gain"])),
        ("d2 sweep gain at 0.045", f3(sw2[0.045]["mean_oos_gain"])),
        ("d2 sweep gain at 0.16", f3(sw2[0.16]["mean_oos_gain"])),
        ("d1 sweep gain at 0.045", f3(sw1[0.045]["mean_oos_gain"])),
        ("d1 sweep gain at 0.06", f3(sw1[0.06]["mean_oos_gain"])),
        ("d1 sweep gain at 0.075", f3(sw1[0.075]["mean_oos_gain"])),
        ("d1 sweep gain at 0.09", f3(sw1[0.09]["mean_oos_gain"])),
        ("d1 sweep gain at 0.12", f3(sw1[0.12]["mean_oos_gain"])),
    ]
    # combined-vs-single CIs quoted for the no-edge and fragile labels (d1)
    for lbl in ("no_edge", "fragile"):
        for single in ("DSR",):
            e = d1["auc_ci"][lbl]["combined_minus_single"][single]
            out.append((f"d1 {lbl} combined-minus-{single} diff", f3(e["diff"])))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true", help="print expected LaTeX rows")
    args = ap.parse_args()
    res = json.loads(RESULTS.read_text())

    if args.print:
        a1, a2 = auc_map(res["dim1"]), auc_map(res["dim2"])
        print("% ---- tab:auc rows (d1 cols then d2 cols) ----")
        for row_label, diag in TAB1_ROWS:
            cells = [f3(a1[diag][c]) for c in TAB1_COLS] + [f3(a2[diag][c]) for c in TAB1_COLS]
            print(f"{row_label:24s} & " + " & ".join(cells) + r" \\")
        print("% ---- inline ----")
        for desc, s in inline_assertions(res):
            print(f"  {desc:42s} {s}")
        return

    tex = TEX.read_text()
    errors: list[str] = []
    check_table1(tex, res, errors)
    check_table2(tex, res, errors)
    for desc, s in inline_assertions(res):
        if s not in tex:
            errors.append(f"inline '{desc}': expected '{s}' not found in main.tex")
    if errors:
        print("PAPER NUMBER CHECK: FAIL")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"PAPER NUMBER CHECK: OK ({len(TAB1_ROWS)} tab:auc rows, "
          f"8 tab:selection rows, {len(inline_assertions(res))} inline values all match results.json)")


if __name__ == "__main__":
    main()
