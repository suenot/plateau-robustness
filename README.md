# Plateaus, Peaks, and the Probability of Backtest Overfitting

A short, reproducible research paper (and the code behind it) that puts the
popular **"prefer a plateau over a peak"** advice for trading-strategy
hyperparameter optimization to a controlled test, and benchmarks it against the
**Probability of Backtest Overfitting (PBO)** and the **Deflated Sharpe Ratio
(DSR)**.

> **Headline findings** (from 9,000 simulated optimization problems with known
> ground truth):
> 1. As *standalone overfitting diagnostics*, the plateau-**shape** metrics are
>    weak (no-edge ROC AUC ≈ 0.50 in 1-D), and the popular fixed thresholds
>    (e.g. "robustness score > 0.1") are not calibrated. The strongest single
>    diagnostic is the **plain PSR against zero** (AUC 0.81), ahead of the DSR
>    (0.79) — the significance test, not the multiplicity deflation, carries
>    the signal here.
> 2. Geometry is **complementary where it has signal**: combining shape metrics
>    with PBO/DSR significantly improves detection of the *fragile* (sharp-peak)
>    mode and of every mode in 2-D, but adds nothing over the plain PSR for
>    no-edge detection in 1-D. The (re-posed) **outlier gap** is the best
>    no-edge detector of any kind in 2-D (AUC 0.84). Geometry must be anchored
>    at the *surrogate* optimum — at the naive argmax it becomes anti-informative.
> 3. As a *selection rule*, preferring the broad optimum of the denoised
>    landscape **works**: it lifts out-of-sample Sharpe by +0.12 (1 parameter)
>    to +0.31 (2 parameters), positive in every curvature tercile and largest
>    where the surrogate optimum is a sharp peak. The benefit grows with the
>    number of parameters.

This grew out of a [marketmaker.cc](https://marketmaker.cc) blog post; the paper
is a de-commercialized, experimentally-validated rewrite — see
[`docs/from-blog-to-paper.md`](docs/from-blog-to-paper.md).

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py          # ~7 min -> results/results.json + record CSVs
python -m plateau_experiments.figures   # -> paper/figures/*.pdf
python scripts/check_paper_numbers.py   # asserts every number in paper/main.tex matches results.json
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
python -m pytest -q     # 12 sanity tests: surface recovery, metric ordering, PBO/DSR bounds, determinism, surrogate edge behavior, anchoring
```

## License

Code: [MIT](LICENSE). The paper text and figures: CC BY 4.0.
