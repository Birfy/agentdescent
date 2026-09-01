"""Turn a run's result JSON into a printable report -- HTML, and then a PDF.

The JSON a run writes has everything in it and is unreadable: a tree of SMILES
strings, four score breakdowns per node, and the numbers that explain why the
search went where it did. This renders it as a document a chemist can read --
structures drawn, the trajectory as a table, the cost stated -- and prints it
through headless Chromium, which every machine that can run a browser has.

Labels come in English and Chinese so the same generator serves a report meant
to be read by whoever asked for the run. Nothing else differs between them; the
numbers, the structures and the caveats are the same document.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from examples.porous._depict import svg
from examples.porous._smiles import validate

__all__ = ["render_html", "write_pdf", "LABELS"]

#: Where a headless Chromium might live. The first one that exists is used.
CHROME_CANDIDATES = (
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/opt/pw-browsers/chromium/chrome-linux/chrome",
    "chromium", "chromium-browser", "google-chrome", "google-chrome-stable",
)

LABELS: Dict[str, Dict[str, str]] = {
    "en": {
        "title": "Porous molecular crystal candidates",
        "subtitle": "Flat-PUCT tree search over molecules, on AgentDescent",
        "config": "Run configuration",
        "seed": "Starting molecule",
        "best": "Best molecule found",
        "rubric": "How a candidate is scored",
        "leaderboard": "Every valid candidate, ranked",
        "gallery": "Every node in the tree",
        "trajectory": "The search trajectory",
        "cost": "What the run cost",
        "caveats": "What this is and is not",
        "score": "total score", "formula": "formula", "atoms": "atoms",
        "heldback": "held-back weightings",
        "node": "node", "parent": "parent", "iteration": "iteration",
        "similarity": "similarity to parent", "prior": "prior P(s,a)",
        "promise": "model rating", "change": "what the model changed",
        "gain": "gain over the starting molecule",
        "rigidity": "rigidity", "symmetry": "symmetry",
        "interactions": "directional sites", "packing": "open packing",
        "synthesizability": "synthesizability",
        "duplicate": "duplicate of node", "refused": "refused by the gate",
        "nodes": "nodes in the tree", "wall": "wall clock", "calls": "model calls",
        "tokens": "tokens", "valid_note": "valid", "refused_note": "refused",
        "dup_note": "duplicates", "retried": "failed and retried",
        "minutes": "min",
    },
    "zh": {
        "title": "多孔分子晶体候选分子",
        "subtitle": "基于 AgentDescent 的 flat-PUCT 分子树搜索",
        "config": "运行配置",
        "seed": "起点分子",
        "best": "搜索得到的最优分子",
        "rubric": "打分标准",
        "leaderboard": "全部合法候选分子排名",
        "gallery": "树中每一个节点的分子",
        "trajectory": "搜索轨迹",
        "cost": "本次运行的开销",
        "caveats": "这份结果是什么，不是什么",
        "score": "综合得分", "formula": "分子式", "atoms": "个原子",
        "heldback": "留出权重档",
        "node": "节点", "parent": "父节点", "iteration": "迭代",
        "similarity": "与父分子相似度", "prior": "先验 P(s,a)",
        "promise": "模型自评", "change": "模型所做的修改",
        "gain": "相对起点分子的提升",
        "rigidity": "刚性", "symmetry": "对称性",
        "interactions": "方向性位点", "packing": "开放堆积",
        "synthesizability": "合成可达性",
        "duplicate": "与节点重复", "refused": "被合法性闸门拒绝",
        "nodes": "树中节点", "wall": "墙钟时间", "calls": "模型调用",
        "tokens": "token 总数", "valid_note": "合法", "refused_note": "被拒",
        "dup_note": "重复", "retried": "次失败并重试",
        "minutes": "分钟",
    },
}

TERM_ORDER = ("rigidity", "symmetry", "interactions", "packing",
              "synthesizability")

_CSS = """
@page { size: A4; margin: 16mm 14mm; }
* { box-sizing: border-box; }
body { font-family: "Helvetica Neue", Helvetica, Arial, "PingFang SC",
       "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
       color: #1b2733; font-size: 10.5px; line-height: 1.55; margin: 0; }
h1 { font-size: 21px; margin: 0 0 2px; letter-spacing: -0.2px; }
h2 { font-size: 13px; margin: 22px 0 8px; padding-bottom: 4px;
     border-bottom: 1.5px solid #1b2733; letter-spacing: 0.2px; }
.sub { color: #5a6a7a; font-size: 11px; margin-bottom: 2px; }
.meta { color: #7b8a99; font-size: 9.5px; }
table { border-collapse: collapse; width: 100%; font-size: 9.5px;
        margin-top: 6px; }
th { text-align: left; font-weight: 600; color: #4a5a6a; border-bottom: 1px solid #c8d2dc;
     padding: 4px 6px; }
td { padding: 4px 6px; border-bottom: 1px solid #eef2f6; vertical-align: top; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.smiles { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 8.5px;
          color: #3d4d5d; word-break: break-all; }
.card { border: 1px solid #dde4ea; border-radius: 6px; padding: 10px 12px;
        margin-top: 8px; }
.row { display: flex; gap: 14px; align-items: flex-start; }
.row > .fig { flex: 0 0 auto; }
.row > .body { flex: 1 1 auto; min-width: 0; }
.bar { height: 7px; background: #e8edf2; border-radius: 4px; overflow: hidden;
       width: 100%; }
.bar > span { display: block; height: 100%; background: #3a6ea5; }
.terms { width: 100%; font-size: 9.5px; }
.terms td { border: none; padding: 2px 6px 2px 0; }
.terms td.w { width: 46%; }
.big { font-size: 16px; font-weight: 600; }
.kv { color: #5a6a7a; }
.note { color: #5a6a7a; font-size: 9.5px; }
ul { margin: 6px 0 0 16px; padding: 0; }
li { margin-bottom: 3px; }
.pagebreak { page-break-before: always; }
.thumbgrid { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
.thumb { border: 1px solid #dde4ea; border-radius: 6px; padding: 6px;
         width: 32%; }
.thumb .cap { font-size: 8.5px; color: #5a6a7a; margin-bottom: 2px; }
.thumb { page-break-inside: avoid; }
.refused { height: 165px; display: flex; flex-direction: column; gap: 4px;
           justify-content: center; padding: 0 6px; }
"""


def _esc(text: Any) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def _fig(smiles: str, width: int = 250, height: int = 190) -> str:
    report = validate(smiles)
    if not report.ok or report.molecule is None:
        return '<div class="note">(no structure)</div>'
    return svg(report.molecule, width=width, height=height)


def _terms_table(terms: Dict[str, float], weights: Dict[str, float],
                 lab: Dict[str, str]) -> str:
    rows = []
    for term in TERM_ORDER:
        value = float(terms.get(term, 0.0))
        weight = float(weights.get(term, 0.0))
        rows.append(
            f'<tr><td>{lab[term]}</td><td class="num">{value:.3f}</td>'
            f'<td class="w"><div class="bar"><span style="width:{value * 100:.0f}%">'
            f'</span></div></td><td class="num kv">x{weight:.2f}</td></tr>')
    return f'<table class="terms">{"".join(rows)}</table>'


def _molecule_card(node: Dict[str, Any], weights: Dict[str, float],
                   lab: Dict[str, str], heading: str,
                   held_back: Optional[float] = None) -> str:
    terms = node.get("terms") or {}
    extra = ""
    if held_back is not None:
        extra = (f'&nbsp;·&nbsp; {lab["heldback"]} '
                 f'<b>{held_back:.3f}</b>')
    return f"""
<h2>{heading}</h2>
<div class="row">
  <div class="fig">{_fig(node["smiles"])}</div>
  <div class="body">
    <div><span class="kv">{lab["score"]}</span>
      <span class="big">{node["score"]:.3f}</span>
      <span class="kv">{extra}</span></div>
    <div class="kv">{lab["formula"]}: {_esc(node.get("formula") or "-")}
      &nbsp;·&nbsp; {node.get("atom_count") or "-"} {lab["atoms"]}</div>
    {_terms_table(terms, weights, lab)}
    <div class="smiles">{_esc(node["smiles"])}</div>
  </div>
</div>"""


def _gallery_card(node: Dict[str, Any], lab: Dict[str, str]) -> str:
    """One node of the tree, drawn -- or, when the gate refused it, explained.

    Every node appears, including the dead ends: a tree reported with only its
    successes in it is not the tree the search built, and the refusals are how a
    reader sees what the model kept trying to do that could not be made.
    """
    index = node["index"]
    parent = ("" if node["parent_index"] is None
              else f' &nbsp;·&nbsp; {lab["parent"]} #{node["parent_index"]}')
    if node.get("valid"):
        score = f'{node["score"]:.3f}'
        head = (f'#{index} &nbsp; <b>{score}</b> &nbsp; '
                f'{_esc(node.get("formula") or "")}{parent}')
        if node.get("duplicate_of") is not None:
            head += (f' <span class="kv">[{lab["duplicate"]} '
                     f'#{node["duplicate_of"]}]</span>')
        body = _fig(node["smiles"], width=210, height=165)
    else:
        head = (f'#{index} &nbsp; <b>{lab["refused_note"]}</b>{parent}')
        body = (f'<div class="refused"><div class="smiles">'
                f'{_esc(node["smiles"]) or "(empty reply)"}</div>'
                f'<div class="note">{_esc(node.get("reason"))}</div></div>')
    return f'<div class="thumb"><div class="cap">{head}</div>{body}</div>'


def _cell(value: Optional[float], digits: int = 2) -> str:
    """One right-aligned number, or a dash when the run recorded none."""
    if value is None:
        return '<td class="num">-</td>'
    return f'<td class="num">{float(value):.{digits}f}</td>'


def _trajectory_row(node: Dict[str, Any], lab: Dict[str, str]) -> str:
    parent = "" if node["parent_index"] is None else node["parent_index"]
    note = ""
    if node.get("duplicate_of") is not None:
        note = (f' <span class="kv">[{lab["duplicate"]} '
                f'{node["duplicate_of"]}]</span>')
    elif not node.get("valid"):
        note = (f' <span class="kv">[{lab["refused"]}: '
                f'{_esc(node.get("reason"))}]</span>')
    return (
        f'<tr><td class="num">{node["index"]}</td>'
        f'<td class="num">{parent}</td>'
        f'<td class="num">{node["iteration"]}</td>'
        + _cell(node.get("score"), 3)
        + _cell(node.get("parent_similarity"))
        + _cell(node.get("prior"))
        + _cell(node.get("promise"), 0)
        + f'<td>{_esc(node.get("change") or "")}{note}</td></tr>')


def render_html(payload: Dict[str, Any], *, lang: str = "en",
                rubric_note: str = "", caveats: Sequence[str] = ()) -> str:
    """The whole report as one self-contained HTML string."""
    lab = LABELS.get(lang, LABELS["en"])
    config = payload["run"]["config"]
    weights = config.get("weights", {})
    seed = payload["seed_molecule"]
    best = payload["best_molecule"]
    tree = payload["tree"]["tree"]
    held = payload.get("held_back_weightings", {})

    config_rows = "".join(
        f"<tr><td>{_esc(key)}</td><td>{_esc(value)}</td></tr>"
        for key, value in config.items() if key != "weights")

    valid = sorted((n for n in tree if n.get("valid")),
                   key=lambda n: n["score"], reverse=True)
    board = "".join(
        f'<tr><td class="num">{i + 1}</td><td class="num">{n["score"]:.3f}</td>'
        + "".join(f'<td class="num">{float(n["terms"].get(t, 0)):.2f}</td>'
                  for t in TERM_ORDER)
        + f'<td>{_esc(n.get("formula") or "")}</td>'
          f'<td class="smiles">{_esc(n["smiles"])}</td></tr>'
        for i, n in enumerate(valid))

    trajectory = "".join(_trajectory_row(node, lab) for node in tree)

    gallery = "".join(_gallery_card(n, lab) for n in tree)

    usage = payload.get("usage") or {}
    summary = payload["tree"]
    cost_rows = [
        (lab["nodes"], f'{summary["nodes"]} '
         f'({summary["valid_nodes"]} {lab["valid_note"]}, '
         f'{summary["invalid_nodes"]} {lab["refused_note"]}, '
         f'{summary["duplicates"]} {lab["dup_note"]})'),
        (lab["wall"], f'{payload["run"]["wall_seconds"] / 60:.1f} {lab["minutes"]}'),
    ]
    if usage:
        cost_rows += [
            (lab["calls"], f'{usage.get("calls", 0)} '
                           f'({usage.get("failures", 0)} {lab["retried"]})'),
            (lab["tokens"], f'{usage.get("total_tokens", 0):,}'),
        ]
    cost_rows += [(k, str(v)) for k, v in
                  (payload.get("proposal_counters") or {}).items()]
    cost = "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
                   for k, v in cost_rows)

    caveat_items = "".join(f"<li>{_esc(c)}</li>" for c in caveats)
    gain = payload.get("gain")
    gain_line = ("" if gain is None else
                 f'<div class="kv">{lab["gain"]}: <b>+{gain:.3f}</b></div>')

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(lab["title"])}</title>
<style>{_CSS}</style></head><body>
<h1>{_esc(lab["title"])}</h1>
<div class="sub">{_esc(lab["subtitle"])}</div>
<div class="meta">{_esc(payload["run"]["finished_at"])}</div>

{_molecule_card(seed, weights, lab, lab["seed"],
                (held.get("seed") or {}).get("mean"))}
{_molecule_card(best, weights, lab, lab["best"],
                (held.get("best") or {}).get("mean"))}
{gain_line}

<h2>{_esc(lab["rubric"])}</h2>
<div class="note">{rubric_note}</div>

<h2>{_esc(lab["config"])}</h2>
<table>{config_rows}</table>

<div class="pagebreak"></div>
<h2>{_esc(lab["leaderboard"])}</h2>
<table><tr><th class="num">#</th><th class="num">{_esc(lab["score"])}</th>
{"".join(f'<th class="num">{_esc(lab[t])}</th>' for t in TERM_ORDER)}
<th>{_esc(lab["formula"])}</th><th>SMILES</th></tr>{board}</table>

<div class="pagebreak"></div>
<h2>{_esc(lab["gallery"])}</h2>
<div class="thumbgrid">{gallery}</div>

<div class="pagebreak"></div>
<h2>{_esc(lab["trajectory"])}</h2>
<table><tr><th class="num">{_esc(lab["node"])}</th>
<th class="num">{_esc(lab["parent"])}</th>
<th class="num">{_esc(lab["iteration"])}</th>
<th class="num">{_esc(lab["score"])}</th>
<th class="num">{_esc(lab["similarity"])}</th>
<th class="num">{_esc(lab["prior"])}</th>
<th class="num">{_esc(lab["promise"])}</th>
<th>{_esc(lab["change"])}</th></tr>{trajectory}</table>

<h2>{_esc(lab["cost"])}</h2>
<table>{cost}</table>

<h2>{_esc(lab["caveats"])}</h2>
<ul>{caveat_items}</ul>
</body></html>"""


def _chrome() -> Optional[str]:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def write_pdf(html: str, pdf_path: Path, *, keep_html: bool = True) -> Path:
    """Print ``html`` to ``pdf_path`` with headless Chromium.

    Raises with a readable message when no browser is installed rather than
    leaving a zero-byte PDF behind -- the failure a caller has to be able to
    tell from "the report is empty".
    """
    pdf_path = Path(pdf_path)
    html_path = pdf_path.with_suffix(".html")
    html_path.write_text(html, encoding="utf-8")
    browser = _chrome()
    if browser is None:
        raise RuntimeError(
            "no headless Chromium found; install one or render "
            f"{html_path} yourself")
    subprocess.run(
        [browser, "--headless", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}",
         html_path.as_uri()],
        check=True, capture_output=True, timeout=180)
    if not pdf_path.exists() or pdf_path.stat().st_size < 1000:
        raise RuntimeError(f"Chromium wrote no usable PDF to {pdf_path}")
    if not keep_html:
        html_path.unlink(missing_ok=True)
    return pdf_path
