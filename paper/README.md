# AgentDescent paper

LaTeX source for the AgentDescent paper, distributed with the repository
rather than posted to a preprint server. All figures are TikZ / pgfplots and
compile from source — no external image files.

## Files

- `main.tex` — the paper (structure, figures, tables)
- `arxiv.sty` — NeurIPS-derived single-column preprint style; it loads
  `geometry` and `fancyhdr` itself, so `main.tex` requests neither
- `references.bib` — bibliography (47 entries)
- `main.pdf` — compiled output (18 pages)

## Build

Any modern TeX distribution works. With [tectonic](https://tectonic-typesetting.github.io):

```bash
tectonic main.tex
```

or with TeX Live:

```bash
latexmk -pdf main.tex
```

## Distribution

The compiled `main.pdf` is committed, so the paper is readable straight from
GitHub with no build step. If you ever do post it, upload `main.tex`,
`arxiv.sty`, `references.bib` and the generated `main.bbl` together —
`arxiv.sty` is not in a stock TeX Live tree and BibTeX is not re-run on the
server.

Every measured number in the paper names the script that produced it; the
Reproducibility section lists them per result, and the raw per-seed JSON for the
merge experiments is in `bench/results/`.

Note: a few bibliography entries for very recent preprints (Agent0, ROLL
Flash) lack arXiv identifiers; their author lists and IDs are worth verifying
against the sources before the paper is circulated.
