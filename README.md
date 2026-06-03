# Plateaus, Peaks, and the Probability of Backtest Overfitting

A short, reproducible research paper (and the code behind it) that puts the
popular **"prefer a plateau over a peak"** advice for trading-strategy
hyperparameter optimization to a controlled test, and benchmarks it against the
**Probability of Backtest Overfitting (PBO)** and the **Deflated Sharpe Ratio
(DSR)**.

> **Headline findings** (from 9,000 simulated optimization problems with known
> ground truth):
> 1. As a *standalone overfitting diagnostic*, local plateau-geometry metrics are
>    weak (ROC AUC ≈ 0.55) and **dominated by the Deflated Sharpe Ratio** (AUC up
>    to 0.79). The popular fixed thresholds (e.g. "robustness score > 0.1") are
>    not calibrated.
> 2. Geometry is nonetheless **complementary** to statistical diagnostics — it
>    catches the *sharp-peak* failure mode they miss and is blind to the
>    *no-edge* mode they catch — so a combined classifier beats any single one.
> 3. As a *selection rule*, preferring the broad optimum **works**: it lifts
>    out-of-sample Sharpe by +0.09 (1 parameter) to +0.24 (2 parameters),
>    concentrated entirely in sharp-peak landscapes. The benefit grows with the
>    number of parameters.

This grew out of a [marketmaker.cc](https://marketmaker.cc) blog post; the paper
is a de-commercialized, experimentally-validated rewrite — see
[`docs/from-blog-to-paper.md`](docs/from-blog-to-paper.md).

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py          # ~3.5 min -> results/results.json + record CSVs
python -m plateau_experiments.figures   # -> paper/figures/*.pdf
```

Everything is deterministic given the seeds in `scripts/run_all.py` (no wall-clock
or unseeded randomness). `--quick` runs a small batch for a smoke check.

## Build the paper

```bash
cd paper && tectonic main.tex          # -> main.pdf   (install: brew install tectonic)
```

The arXiv-ready source bundle (`main.tex`, `main.bbl`, figures) is assembled in
`arxiv_submission/` and `arxiv_submission.tar.gz`. See
[`docs/endorsement-and-submission.md`](docs/endorsement-and-submission.md) for how
to actually submit (including the arXiv endorsement requirement).

## Layout

```
plateau_experiments/
  surfaces.py    # generative model: known true Sharpe surface + correlated returns
  metrics.py     # the three plateau metrics (estimator flaws fixed)
  baselines.py   # PBO via CSCV, Probabilistic/Deflated Sharpe Ratio
  simulate.py    # one experiment end-to-end + Monte-Carlo batches
  analysis.py    # Spearman, ROC AUC, complementarity, selection payoff
  figures.py     # the paper's figures
scripts/run_all.py
tests/           # pytest sanity checks (python -m pytest -q)
paper/           # main.tex, refs.bib, figures/, compiled main.pdf
results/         # results.json + per-experiment CSVs (generated)
```

## Tests

```bash
python -m pytest -q     # 8 sanity tests: surface recovery, metric ordering, PBO/DSR bounds, determinism
```

## License

Code: [MIT](LICENSE). The paper text and figures: CC BY 4.0.
