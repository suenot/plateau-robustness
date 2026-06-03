# arXiv endorsement and submission

This paper targets **q-fin.CP** (Computational Finance) as primary, with
**q-fin.ST** (Statistical Finance) and **stat.ML** as cross-lists. q-fin requires
an **endorsement** for a first-time submitter — plan for this before you try to
submit.

## 1. Endorsement (do this first)

arXiv requires new submitters to be *endorsed* for a subject area before their
first paper there is accepted. ([arxiv.org/help/endorsement](https://info.arxiv.org/help/endorsement.html))

- **You may be auto-endorsed** if you register with a recognized academic/
  institutional email and have a typical affiliation. Independent researchers
  usually are **not**, and must request endorsement.
- **To get endorsed:** start the submission; if arXiv asks for endorsement it
  shows you an *endorsement code* and a link. Send that to a qualified endorser —
  someone who has had several submissions accepted in q-fin (a co-author,
  academic contact, or a colleague in quant finance). They click the link and
  endorse you.
- **No endorser?** You can still email arXiv moderation, but the realistic path
  is to find one qualified person in the field. The released code + a credible
  preprint draft make this easier to ask for.
- Endorsement is **per-archive group** and is a one-time gate; later q-fin
  submissions won't need it.

> Practical note: a single, careful, genuinely-validated paper (this one) is far
> more likely to attract an endorser and survive moderation than a burst of
> reformatted blog posts. Lead with quality.

## 2. Before you submit — final checklist

- [ ] Real **name + affiliation + email** set in `paper/main.tex` (currently
      *Eugen Soloviov, Independent Researcher*) — must match your arXiv account.
- [ ] Replace the **code URL** placeholder `github.com/<your-org>/...` and push
      the repo public (arXiv reviewers and readers will look).
- [ ] `cd paper && tectonic main.tex` builds with no errors.
- [ ] The source bundle is current: re-run the two commands in the README if you
      changed anything, then rebuild `arxiv_submission.tar.gz`.

## 3. Submitting

**What to upload:** the **source** bundle, not the PDF (arXiv prefers LaTeX
source and will reject a TeX-generated PDF-only submission). Use
`arxiv_submission.tar.gz` — it contains `main.tex`, `main.bbl` (so arXiv needs no
bibtex run), `refs.bib`, and `figures/*.pdf`.

### Option A — manual (recommended for your first ever submission)
Do it by hand once so you see every screen:
1. Create/log in at [arxiv.org](https://arxiv.org), complete the endorsement step.
2. *Start a New Submission* → license (the default arXiv non-exclusive license is
   fine, or CC BY 4.0) → upload `arxiv_submission.tar.gz` → let it process/compile.
3. Set metadata: title, authors, abstract (copy from the paper), primary
   **q-fin.CP**, cross-list **q-fin.ST**, **stat.ML**, and a comments line
   (e.g. "9 pages, 4 figures; code at <url>").
4. Preview the arXiv-built PDF, then submit. Expect a moderation hold of a day or
   more.

### Option B — automate with `arxiv_agent`
The sibling tool in `../arxiv_agent` can drive the submission via a stealth
browser once you're endorsed and logged in:
```bash
cd ../arxiv_agent && source .venv/bin/activate
# point a paper entry at this source bundle, or upload it directly:
arxiv-agent login                  # log in to arXiv by hand (persistent profile)
arxiv-agent calibrate              # first run: confirm the submission-form selectors
# then submit the prepared bundle (see arxiv_agent README)
```
For a single first paper, Option A is simpler and lower-risk; reserve the
automation for when you have many papers ready.
