# From blog post to paper: what changed and why

The paper started as the marketmaker.cc blog post *"Plateau Analysis: How to
Distinguish a Robust Optimum from Overfitting."* A blog post is not an arXiv
paper, and submitting one as-is would (rightly) be rejected. Here is exactly what
was changed, for transparency and so the same checklist can be reused on other
posts.

## Removed (would trigger desk-rejection or raise integrity issues)
- **All promotional/product content**: the "Backtests Without Illusions" series
  framing, every internal `marketmaker.cc/blog` cross-link used as a citation,
  the self-citation BibTeX with `version = {0.1.0}`, and the marketing tone.
- **Fabricated results presented as data.** The post's tables ("Strategy A/B/C",
  the example reports) were *illustrative* hand-picked numbers labeled "typical."
  Presenting invented numbers as empirical results under a real name is the core
  integrity problem. They are gone; every number in the paper is computed by
  `scripts/run_all.py`.
- **Decorative figures.** The `.webp` blog images were replaced by figures
  generated from the actual experiments.

## Fixed (methodological flaws found in review)
- **Sensitivity estimator.** The post's elasticity is ill-posed at an interior
  maximum (its first-derivative term vanishes) and was computed by regressing
  over top trials in which *all* parameters co-vary. Replaced by a well-defined
  local drop / curvature on **ceteris-paribus** axis slices.
- **Plateau width.** The post used `max − min` over all "good" points, which a
  single distant lucky point inflates. Replaced by the **connected** super-level
  set around the optimum.

## Added (what makes it a paper)
- **A controlled simulation with ground truth** — the population Sharpe surface
  is known, so selection quality and out-of-sample degradation are measurable.
- **Benchmarks**: the Probability of Backtest Overfitting (PBO/CSCV) and the
  Deflated Sharpe Ratio, implemented and validated, run on the same data.
- **Honest, mixed conclusions**, including negative results: the plateau metrics
  are weak standalone diagnostics and the post's `R > 0.1` thresholds are not
  calibrated. The contribution is the *validation* (what works, what doesn't,
  and the complementarity), not a claim that the metrics are great.
- Standard academic structure, real citations, a reproducibility statement, and
  released code.

## Still to confirm before submission
- **Author identity.** The byline uses *Eugen Soloviov* (from the original
  BibTeX) with *Independent Researcher*. Confirm the exact name/affiliation/email
  you want on a permanent public record; it must match your arXiv account and
  endorsement.
- **Code URL.** Replace the `github.com/<your-org>/plateau-robustness` placeholder
  with the real repository before submitting.
