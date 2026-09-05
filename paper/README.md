# AgentDescent paper

**This directory lives on the `paper` branch only.** `main` carries the code,
the docs and `bench/results/`; the LaTeX source and the compiled PDF are kept
here so that a paper revision never touches the released code and a code
release never re-compiles the paper. To pick up code changes, merge `main` into
this branch — the two never touch the same files.

LaTeX source for the AgentDescent paper. All figures are TikZ / pgfplots and
compile from source — no external image files.

## Files

- `main.tex` — the paper (structure, figures, tables)
- `arxiv.sty` — NeurIPS-derived single-column preprint style; it loads
  `geometry` and `fancyhdr` itself, so `main.tex` requests neither
- `references.bib` — bibliography (51 entries)
- `main.pdf` — compiled output (19 pages)

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
GitHub with no build step. To submit it anywhere, send `main.tex`, `arxiv.sty`,
`references.bib` and the generated `main.bbl` together — `arxiv.sty` is not in a
stock TeX Live tree, and most servers do not re-run BibTeX.

Every measured number in the paper names the script that produced it; the
Reproducibility section lists them per result, and the raw per-seed JSON for the
merge experiments is in `bench/results/`.

Note: every bibliography entry now carries an author list and an arXiv
identifier. The three that did not — Agent0 (arXiv:2511.16043), ROLL Flash
(arXiv:2510.11345) and CoEvoSkills (arXiv:2604.01687) — were filled in from
their arXiv abstract pages; Agent0's key was `xue2025agent0` against a first
author of Xia, and is now `xia2025agent0`.
