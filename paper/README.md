# AgentDescent paper (arXiv source)

arXiv-ready LaTeX source for the AgentDescent paper. All figures are TikZ /
pgfplots and compile from source — no external image files.

## Files

- `main.tex` — the paper (structure, figures, tables)
- `arxiv.sty` — NeurIPS-derived single-column preprint style; it loads
  `geometry` and `fancyhdr` itself, so `main.tex` requests neither
- `references.bib` — bibliography (47 entries)
- `main.pdf` — compiled output (14 pages)

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

Upload `main.tex`, `arxiv.sty`, `references.bib`, and the generated `main.bbl`
(run the build once locally to produce it — arXiv does not run BibTeX
against `.bib` files, it needs the `.bbl`). `arxiv.sty` is not in arXiv's
TeX Live tree, so omitting it fails the remote build. Suggested categories:
`cs.AI` (primary), `cs.MA`, `cs.DC`.

Every measured number in the paper comes from the repository's own docs
(`docs/results.md`, `docs/efficiency.md`, `docs/self-evolution-examples.md`,
`docs/matrix-overview.md`) and names the script that produced it.

Note: a few bibliography entries for very recent preprints (Agent0, ROLL
Flash) lack arXiv identifiers — verify author lists and IDs against arXiv
before submission.
