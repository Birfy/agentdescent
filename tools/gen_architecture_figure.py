"""Render the paper's architecture figure into `docs/assets/`, for the README.

    python -m tools.gen_architecture_figure           # write the PNG
    python -m tools.gen_architecture_figure --check   # fail if it has drifted

The README shows the same architecture diagram as the paper. The source is one
`tikzpicture` in `paper/main.tex`, which lives on the `paper` branch -- so the
image here is *derived*, and a derived binary nobody can regenerate is how the
copy on `main` drifted from the paper in the first place. This script is the
recipe, and `--check` is what notices.

It needs a TeX engine on PATH (`tectonic`, else `latexmk`, else `pdflatex`) and
`pymupdf` for the PDF-to-PNG step. Neither is a dependency of the package or of
the test suite; this is a maintainer tool, run when the figure changes.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "assets", "architecture.png")
PAPER_REF = "origin/paper:paper/main.tex"
LABEL = r"\label{fig:architecture}"
DPI = 300

PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath}
\usepackage{xcolor}
%(colors)s
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,backgrounds,calc}
\newcommand{\code}[1]{\texttt{#1}}
\newcommand{\evolve}{\texttt{evolve()}}
\begin{document}
%(tikz)s
\end{document}
"""


def paper_source() -> str:
    """`paper/main.tex` as the `paper` branch has it."""
    try:
        return subprocess.run(["git", "show", PAPER_REF], cwd=ROOT, check=True,
                              capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as exc:  # pragma: no cover - maintainer path
        raise SystemExit(
            f"cannot read {PAPER_REF}: {exc.stderr.strip()}\n"
            "Fetch the paper branch first: git fetch origin paper"
        )


def extract(src: str) -> str:
    """The tikzpicture carrying `fig:architecture`, plus the colours it uses."""
    at = src.index(LABEL)
    start = src.rindex(r"\begin{tikzpicture}", 0, at)
    end = src.index(r"\end{tikzpicture}", start) + len(r"\end{tikzpicture}")
    colors = "\n".join(re.findall(r"\\definecolor\{[^\n]*", src))
    return PREAMBLE % {"colors": colors, "tikz": src[start:end]}


def tex_engine() -> list:
    for exe, args in (("tectonic", ["-X", "compile"]), ("latexmk", ["-pdf"]),
                      ("pdflatex", ["-interaction=nonstopmode"])):
        if shutil.which(exe):
            return [exe] + args
    raise SystemExit("no TeX engine on PATH (tried tectonic, latexmk, pdflatex)")


def render(doc: str) -> bytes:
    try:
        import pymupdf
    except ImportError:  # pragma: no cover - maintainer path
        raise SystemExit("pymupdf is required for the PDF-to-PNG step: pip install pymupdf")

    with tempfile.TemporaryDirectory() as tmp:
        tex = os.path.join(tmp, "figure.tex")
        with open(tex, "w", encoding="utf-8") as fh:
            fh.write(doc)
        cmd = tex_engine()
        extra = ["--outdir", tmp] if cmd[0] == "tectonic" else ["-output-directory", tmp]
        run = subprocess.run(cmd + [tex] + extra, cwd=tmp, capture_output=True, text=True)
        pdf = os.path.join(tmp, "figure.pdf")
        if run.returncode or not os.path.exists(pdf):
            sys.stderr.write(run.stdout[-2000:] + run.stderr[-2000:])
            raise SystemExit("the figure did not compile")
        with pymupdf.open(pdf) as doc_pdf:
            # No alpha: a transparent background renders dark text on dark in
            # GitHub's dark theme. A white card reads on both.
            return doc_pdf[0].get_pixmap(dpi=DPI, colorspace=pymupdf.csRGB).tobytes("png")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the committed PNG is not what the paper renders")
    args = ap.parse_args(argv)

    png = render(extract(paper_source()))

    if args.check:
        if not os.path.exists(OUT):
            print(f"{OUT} is missing", file=sys.stderr)
            return 1
        with open(OUT, "rb") as fh:
            same = fh.read() == png
        # Renderers are not bit-reproducible across versions, so a mismatch is a
        # prompt to look, not a verdict.
        print("figure matches" if same else
              f"{OUT} differs from what the paper renders here -- regenerate and eyeball it")
        return 0 if same else 1

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as fh:
        fh.write(png)
    print(f"wrote {os.path.relpath(OUT, ROOT)} ({len(png) // 1024} KB, {DPI} dpi)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
