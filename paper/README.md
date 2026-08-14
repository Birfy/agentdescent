# AgentDescent paper (arXiv source)

arXiv-ready LaTeX source for the AgentDescent paper. All figures are TikZ /
pgfplots and compile from source — no external image files.

## Files

- `main.tex` — the paper (structure, figures, tables)
- `references.bib` — bibliography (38 entries)
- `main.pdf` — compiled output (15 pages)

## Build

Any modern TeX distribution works. With [tectonic](https://tectonic-typesetting.github.io):

```bash
tectonic main.tex
```

or with TeX Live:

```bash
latexmk -pdf main.tex
```

## Submitting to arXiv

Upload `main.tex`, `references.bib`, and the generated `main.bbl`
(run the build once locally to produce it — arXiv does not run BibTeX
against `.bib` files, it needs the `.bbl`). Suggested categories:
`cs.AI` (primary), `cs.MA`, `cs.DC`.

Every measured number in the paper comes from the repository's own docs
(`docs/results.md`, `docs/efficiency.md`, `docs/self-evolution-examples.md`,
`docs/matrix-overview.md`) and names the script that produced it.

Note: a few bibliography entries for very recent preprints (Agent0, ROLL
Flash) lack arXiv identifiers — verify author lists and IDs against arXiv
before submission.
