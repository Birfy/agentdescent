"""Render the candidate-method live matrix as a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_INPUT = Path("bench/results/candidate-methods-framework-final.json")
DEFAULT_OUTPUT = Path("docs/algo-candidate-methods.md")

DISPLAY_NAMES = {
    "promptbreeder": "PromptBreeder",
    "aflow": "AFlow",
    "reflexion": "Reflexion",
    "self_refine": "Self-Refine",
    "voyager": "Voyager",
    "skillweaver": "SkillWeaver",
    "absolute_zero": "Absolute Zero",
    "r_zero": "R-Zero",
    "agent0": "Agent0",
    "sica": "SICA",
    "godel_agent": "Godel Agent",
}

UPSTREAMS = {
    "promptbreeder": (
        "paper (no official released code)",
        "https://arxiv.org/abs/2309.16797",
    ),
    "aflow": (
        "FoundationAgents/AFlow@3f457218",
        "https://github.com/FoundationAgents/AFlow/tree/3f457218fc716093fe53f6df8a5d5e6379d66346",
    ),
    "reflexion": (
        "noahshinn/reflexion@218cf0ef",
        "https://github.com/noahshinn/reflexion/tree/218cf0ef1df84b05ce379dd4a8e47f17766733a0",
    ),
    "self_refine": (
        "madaan/self-refine@9a206d41",
        "https://github.com/madaan/self-refine/tree/9a206d41e5d2d0c241bb441f41eeadb945afaa55",
    ),
    "voyager": (
        "MineDojo/Voyager@55e45a88",
        "https://github.com/MineDojo/Voyager/tree/55e45a880755d0c8c66ca7fb5fe7962ac8974f89",
    ),
    "skillweaver": (
        "OSU-NLP-Group/SkillWeaver@f2a63d65",
        "https://github.com/OSU-NLP-Group/SkillWeaver/tree/f2a63d65d0f6ff46ac30e817cede8797f8f25b97",
    ),
    "absolute_zero": (
        "LeapLabTHU/Absolute-Zero-Reasoner@484afa48",
        "https://github.com/LeapLabTHU/Absolute-Zero-Reasoner/tree/484afa480c8f6fd77faa3d35451f24f287f58ee1",
    ),
    "r_zero": (
        "Chengsong-Huang/R-Zero@5699329d",
        "https://github.com/Chengsong-Huang/R-Zero/tree/5699329d018d79535b7910abdedf5a6eebf355fd",
    ),
    "agent0": (
        "aiming-lab/Agent0@f775b510",
        "https://github.com/aiming-lab/Agent0/tree/f775b5101e62fe92976831adf4a21a38fcc0a767",
    ),
    "sica": (
        "MaximeRobeyns/self_improving_coding_agent@ed8275dc",
        "https://github.com/MaximeRobeyns/self_improving_coding_agent/tree/ed8275dca4d3c5dbf77229964351fe9b424797dc",
    ),
    "godel_agent": (
        "Arvid-pku/Godel_Agent@bbb50879",
        "https://github.com/Arvid-pku/Godel_Agent/tree/bbb508796be31c7140cdfc7106efd830a1324242",
    ),
}

MODE_LABELS = {
    "serial": "serial",
    "sync_parallel": "sync",
    "async_pipeline": "async",
}


def _median(interval: Optional[Sequence[float]]) -> Optional[float]:
    return None if not interval else float(interval[1])


def _number(value: Optional[float], digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _quality(mode: Mapping[str, Any]) -> str:
    before = _median(mode.get("baseline_quality_min_median_max"))
    after = _median(mode.get("final_quality_min_median_max"))
    return f"{_number(before, 3)} -> {_number(after, 3)}"


def _speedup(comparison: Mapping[str, Any]) -> str:
    count = int(comparison.get("paired_runs", 0))
    value = _median(comparison.get("speedup_min_median_max"))
    return "--" if value is None else f"{value:.2f}x (n={count})"


def _ttq(mode: Mapping[str, Any]) -> str:
    reached = int(mode.get("target_reached_runs", 0))
    runs = int(mode.get("runs", 0))
    value = _median(mode.get("ttq_s_min_median_max"))
    return f"{_number(value)} ({reached}/{runs})"


def _mode_costs(
    observations: Sequence[Mapping[str, Any]], modes: Sequence[str]
) -> Dict[str, Dict[str, int]]:
    costs: Dict[str, Dict[str, int]] = {}
    for mode in modes:
        rows = [row for row in observations if row.get("mode") == mode]
        costs[mode] = {
            "runs": len(rows),
            "provider_calls": sum(int(row["usage"]["calls"]) for row in rows),
            "actor_calls": sum(int(row["actor_usage"]["calls"]) for row in rows),
            "proposal_calls": sum(
                int(row["budget"]["observed_proposal_calls"]) for row in rows
            ),
            "rollouts": sum(
                int(row.get("framework", {}).get("rollouts", 0)) for row in rows
            ),
            "tokens": sum(int(row["usage"]["total_tokens"]) for row in rows),
        }
    return costs


def _validate(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "completed":
        raise ValueError(f"matrix status is {payload.get('status')!r}, not 'completed'")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("matrix has no observations")
    config = payload.get("config", {})
    repeats = int(config.get("repeats", 0))
    algorithms = list(config.get("algorithms", ()))
    modes = list(config.get("modes", ()))
    expected = repeats * len(algorithms) * len(modes)
    if len(observations) != expected:
        raise ValueError(f"expected {expected} observations, found {len(observations)}")
    seed = int(config.get("seed", 0))
    expected_keys = {
        (algorithm, mode, seed + repeat * 100)
        for repeat in range(repeats)
        for algorithm in algorithms
        for mode in modes
    }
    actual_keys = {
        (row.get("algorithm"), row.get("mode"), row.get("seed"))
        for row in observations
    }
    if actual_keys != expected_keys:
        raise ValueError("matrix contains duplicate, missing, or unexpected method/mode/seed rows")
    if any(row.get("error") for row in observations):
        raise ValueError("matrix contains failed observations")
    expected_runtime = {
        "serial": "evolve",
        "sync_parallel": "evolve",
        "async_pipeline": "async_evolve",
    }
    if any(
        row.get("framework", {}).get("runtime") != expected_runtime.get(row.get("mode"))
        for row in observations
    ):
        raise ValueError("matrix contains observations outside AgentDescent runtimes")
    if payload.get("summary", {}).get("totals", {}).get("budget_mismatches"):
        raise ValueError("matrix contains candidate/proposal budget mismatches")
    if payload.get("summary", {}).get("totals", {}).get("provider_failures"):
        raise ValueError("matrix contains provider failures")


def render_report(payload: Mapping[str, Any], *, input_path: Path) -> str:
    _validate(payload)
    config = payload["config"]
    summary = payload["summary"]
    totals = summary["totals"]
    algorithms = list(config["algorithms"])
    modes = list(config["modes"])
    repeats = int(config["repeats"])
    observations = list(payload["observations"])
    source_hash = payload.get("implementation", {}).get("source_sha256", "unknown")
    method_word = "method" if len(algorithms) == 1 else "methods"
    seed_word = "seed" if repeats == 1 else "seeds"
    spread_verb = "shows" if repeats == 1 else "show"

    lines: List[str] = [
        "# Remaining candidate methods: live runtime matrix",
        "",
        "## Scope",
        "",
        (
            f"This is a live {config.get('model', 'model')} experiment over "
            f"{len(algorithms)} {method_word}, "
            f"{len(modes)} execution modes, and {repeats} paired {seed_word} "
            f"({totals['observations']} observations). It uses no response replay "
            "and no synthetic latency."
        ),
        "",
        "The comparison changes AgentDescent scheduling, not the candidate or proposal-call budget:",
        "",
        "- `serial`: `evolve(max_concurrency=1)`.",
        "- `sync_parallel`: `evolve(max_concurrency=workers)` with its round barrier.",
        "- `async_pipeline`: `async_evolve(...)` with completion-order merge sweeps.",
        "",
        "A value above 1.0x means the comparison mode is faster. Every speedup is paired by method and seed.",
        "",
        "Port author: `cyanneko`.",
        "",
        "## Main results",
        "",
        "| Method | Fidelity | Serial quality | Sync quality | Async quality | Serial E2E | Sync E2E | Async E2E | Sync / serial | Async / sync |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for algorithm in algorithms:
        modes_summary = summary["algorithms"][algorithm]
        comparisons = summary["algorithm_comparisons"][algorithm]
        fidelity = next(
            row["fidelity"]
            for row in payload["observations"]
            if row["algorithm"] == algorithm
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    DISPLAY_NAMES[algorithm],
                    f"`{fidelity}`",
                    _quality(modes_summary["serial"]),
                    _quality(modes_summary["sync_parallel"]),
                    _quality(modes_summary["async_pipeline"]),
                    _number(_median(modes_summary["serial"]["wall_s_min_median_max"])),
                    _number(_median(modes_summary["sync_parallel"]["wall_s_min_median_max"])),
                    _number(_median(modes_summary["async_pipeline"]["wall_s_min_median_max"])),
                    _speedup(comparisons["sync_vs_serial_wall"]),
                    _speedup(comparisons["async_vs_sync_wall"]),
                )
            )
            + " |"
        )

    aggregate = summary["comparisons"]
    sync_winners = [
        algorithm
        for algorithm in algorithms
        if (
            _median(
                summary["algorithm_comparisons"][algorithm][
                    "sync_vs_serial_wall"
                ].get("speedup_min_median_max")
            )
            or 0.0
        )
        > 1.0
    ]
    ttq_winners = []
    for algorithm in algorithms:
        comparison = summary["algorithm_comparisons"][algorithm][
            "async_vs_sync_ttq"
        ]
        speedup = _median(comparison.get("speedup_min_median_max"))
        pairs = int(comparison.get("paired_runs", 0))
        if speedup is not None and speedup > 1.05 and pairs >= 2:
            ttq_winners.append((speedup, pairs, algorithm))
    ttq_winners.sort(reverse=True)
    stalled_quality = []
    for algorithm in algorithms:
        modes_summary = summary["algorithms"][algorithm]
        gains = [
            _median(modes_summary[mode].get("quality_gain_min_median_max"))
            for mode in modes
        ]
        if all(gain is not None and gain <= 0.0 for gain in gains):
            stalled_quality.append(algorithm)
    lines.extend(
        [
            "",
            "E2E columns are median seconds including disjoint baseline and final tests. Quality cells are median strict test reward before -> after.",
            "",
            "## Framework timing",
            "",
            "| Method | Serial engine | Sync engine | Async engine | Sync / serial | Async / sync |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for algorithm in algorithms:
        modes_summary = summary["algorithms"][algorithm]
        comparisons = summary["algorithm_comparisons"][algorithm]
        lines.append(
            "| "
            + " | ".join(
                (
                    DISPLAY_NAMES[algorithm],
                    _number(
                        _median(
                            modes_summary["serial"][
                                "engine_wall_s_min_median_max"
                            ]
                        )
                    ),
                    _number(
                        _median(
                            modes_summary["sync_parallel"][
                                "engine_wall_s_min_median_max"
                            ]
                        )
                    ),
                    _number(
                        _median(
                            modes_summary["async_pipeline"][
                                "engine_wall_s_min_median_max"
                            ]
                        )
                    ),
                    _speedup(comparisons["sync_vs_serial_engine_wall"]),
                    _speedup(comparisons["async_vs_sync_engine_wall"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Engine columns isolate the framework evolution window; E2E remains the user-visible completion time.",
            "",
            "## Aggregate timing",
            "",
            f"- Sync vs serial end-to-end: **{_speedup(aggregate['sync_vs_serial_wall'])}**.",
            f"- Async vs sync end-to-end: **{_speedup(aggregate['async_vs_sync_wall'])}**.",
            f"- Sync vs serial engine window: **{_speedup(aggregate['sync_vs_serial_engine_wall'])}**.",
            f"- Async vs sync engine window: **{_speedup(aggregate['async_vs_sync_engine_wall'])}**.",
            f"- Sync vs serial time-to-quality: **{_speedup(aggregate['sync_vs_serial_ttq'])}**.",
            f"- Async vs sync time-to-quality: **{_speedup(aggregate['async_vs_sync_ttq'])}**.",
            "",
            "Full min / median / max intervals are in the JSON rather than being hidden behind a point estimate.",
            "",
            "## Interpretation",
            "",
            (
                f"- Sync parallel has a median end-to-end win for "
                f"**{len(sync_winners)}/{len(algorithms)} methods**; across all "
                f"paired method/seeds it is {_speedup(aggregate['sync_vs_serial_wall'])} "
                f"faster end-to-end and {_speedup(aggregate['sync_vs_serial_engine_wall'])} "
                "faster inside the framework evolution window."
            ),
            (
                f"- Async is not a general full-return speedup: its aggregate "
                f"end-to-end result is {_speedup(aggregate['async_vs_sync_wall'])} "
                f"and its engine-window result is {_speedup(aggregate['async_vs_sync_engine_wall'])} "
                "relative to sync parallel."
            ),
            "",
            "## Time to quality",
            "",
            "| Method | Serial TTQ | Sync TTQ | Async TTQ | Async / sync TTQ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for algorithm in algorithms:
        modes_summary = summary["algorithms"][algorithm]
        comparisons = summary["algorithm_comparisons"][algorithm]
        lines.append(
            "| "
            + " | ".join(
                (
                    DISPLAY_NAMES[algorithm],
                    _ttq(modes_summary["serial"]),
                    _ttq(modes_summary["sync_parallel"]),
                    _ttq(modes_summary["async_pipeline"]),
                    _speedup(comparisons["async_vs_sync_ttq"]),
                )
            )
            + " |"
        )

    if ttq_winners:
        winner_text = ", ".join(
            f"{DISPLAY_NAMES[algorithm]} ({speedup:.2f}x, n={pairs})"
            for speedup, pairs, algorithm in ttq_winners
        )
        lines.extend(
            [
                "",
                f"Repeated-seed async TTQ gains above 1.05x: {winner_text}.",
            ]
        )
    if stalled_quality:
        names = ", ".join(DISPLAY_NAMES[name] for name in stalled_quality)
        lines.extend(
            [
                "",
                f"No mode has a positive median independent-test quality gain for: {names}. Timing results for these methods measure fixed-budget execution efficiency only.",
            ]
        )

    lines.extend(
        [
            "",
            "`--` is intentional: a method that did not cross the fixed internal target has no TTQ, even if its final independent test improved.",
            "",
            "## Cost and integrity",
            "",
            f"- Provider calls: **{totals['provider_calls']}**.",
            f"- Framework actor calls: **{totals['actor_calls']}**.",
            f"- Algorithm proposal calls: **{totals['proposal_calls']}**.",
            f"- Logical calls from event traces: **{totals['logical_calls']}**.",
            f"- Tokens: **{totals['tokens']}**.",
            f"- Provider failures: **{totals['provider_failures']}**.",
            f"- Candidate/proposal budget mismatches: **{totals['budget_mismatches']}**.",
            "",
            "| Mode | Runs | Provider calls | Actor calls | Proposal calls | Rollouts | Tokens |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, cost in _mode_costs(observations, modes).items():
        lines.append(
            f"| `{mode}` | {cost['runs']} | {cost['provider_calls']} | "
            f"{cost['actor_calls']} | {cost['proposal_calls']} | "
            f"{cost['rollouts']} | {cost['tokens']} |"
        )
    lines.extend(
        [
            "",
            f"- Candidate-method source SHA-256: `{source_hash}`.",
            "- Raw prompts, responses, generated tasks, learned instructions, and generated source are intentionally absent from the result file.",
            "",
            "## Fidelity boundaries",
            "",
            "| Method | Upstream reference | What this experiment preserves | Boundary |",
            "|---|---|---|---|",
        ]
    )

    boundaries = {
        "promptbreeder": ("task/mutation prompt co-evolution and fitness selection", "compact population and local arithmetic domain"),
        "aflow": ("execution-feedback graph expansion and workflow evaluation", "one MCTS depth and two model nodes"),
        "reflexion": ("attempt, external feedback, verbal reflection, retry", "deterministic evaluator instead of HotpotQA/ALFWorld"),
        "self_refine": ("GENERATE, FEEDBACK, REFINE", "one refinement iteration on a compact rubric"),
        "voyager": ("automatic curriculum, executable skills, repair, critic", "deterministic crafting world instead of Minecraft"),
        "skillweaver": ("Propose, Practice, Verify, Hone and reusable APIs", "deterministic settings site instead of WebArena"),
        "absolute_zero": ("proposer, solver, grounded verifier, self-play curriculum", "verbal policy memory instead of PPO weight updates"),
        "r_zero": ("separate Challenger/Solver roles and separate updates", "verbal role memories instead of two-model GRPO"),
        "agent0": ("curriculum/executor co-evolution and multi-turn tools", "calculator environment and memory instead of RL post-training"),
        "sica": ("real Python self-edit and measured utility gate", "one AST-gated policy function instead of SWE-bench Docker"),
        "godel_agent": ("artifact-owned solve and recursive self-improvement functions", "AST-gated replacement instead of full monkey-patched scaffold"),
    }
    for algorithm in algorithms:
        label, url = UPSTREAMS[algorithm]
        preserved, boundary = boundaries[algorithm]
        lines.append(
            f"| {DISPLAY_NAMES[algorithm]} | [{label}]({url}) | {preserved} | {boundary} |"
        )

    lines.extend(
        [
            "",
            "These fidelity labels are part of the result, not a disclaimer added after seeing the scores. Environment and inference analogues must not be cited as reproductions of paper benchmark numbers.",
            "",
            "## Data provenance",
            "",
            "Only observations whose result records AgentDescent `evolve` or `async_evolve` are included. Earlier hand-scheduled pilots are excluded from every table and aggregate.",
            "",
        ]
    )

    source_runs = list(payload.get("source_runs", ()))
    if source_runs:
        for run in source_runs:
            run_seeds = ", ".join(str(seed) for seed in run.get("seeds", ()))
            lines.append(
                f"- `{run.get('path')}`: seeds {run_seeds}; "
                f"{run.get('observations')} completed observations."
            )
    else:
        lines.append("- This artifact was produced by one atomic benchmark run.")

    lines.extend(["", "## Reproduction", "", "```bash"])
    if source_runs:
        for run in source_runs:
            lines.extend(
                [
                    "python -m bench.candidate_methods --provider "
                    f"{config.get('provider', 'openai')} --model {config.get('model')} \\",
                    f"  --workers {config.get('workers')} --candidates {config.get('candidates')} "
                    f"--repeats {run.get('repeats')} --seed {run.get('seed')} \\",
                    "  --modes serial sync_parallel async_pipeline \\",
                    f"  --thinking {config.get('thinking')} --temperature {config.get('temperature')} "
                    f"--max-tokens {config.get('max_tokens')} \\",
                    f"  --output {run.get('path')} --yes",
                ]
            )
        inputs = " ".join(str(run.get("path")) for run in source_runs)
        seeds = " ".join(str(seed) for seed in config.get("seed_sequence", ()))
        lines.extend(
            [
                f"python -m bench.candidate_methods_merge --inputs {inputs} \\",
                f"  --expected-seeds {seeds} --output {input_path.as_posix()}",
            ]
        )
    else:
        lines.extend(
            [
                "python -m bench.candidate_methods --provider openai --model glm-5.2 \\",
                f"  --workers 2 --candidates 2 --repeats {repeats} \\",
                "  --modes serial sync_parallel async_pipeline \\",
                "  --thinking disabled --temperature 0 --max-tokens 1024 \\",
                f"  --output {input_path.as_posix()} --yes",
            ]
        )
    lines.extend(
        [
            f"python -m bench.candidate_methods_report --input {input_path.as_posix()}",
            "```",
            "",
            "The runner reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment, rotates mode order, writes each paid observation atomically, and stops if the implementation fingerprint changes.",
            "",
            "## Limits",
            "",
            f"- {repeats} paired {seed_word} {spread_verb} an observed spread, not a confidence interval or a paper-scale result.",
            "- Temperature zero does not make a hosted model or network deterministic.",
            "- Final quality differences between modes can arise from completion order, stale candidates, and model variance; they are not evidence that parallelism improves reasoning quality.",
            "- The principal supported claim is timing: equal candidate/proposal work can overlap; framework gate calls are measured because async can perform a different number of merge sweeps.",
            "- Async helps TTQ only when a useful completion can commit before a sync barrier; it need not improve full-return E2E time.",
            "",
            f"Machine-readable source: `{input_path.as_posix()}`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    report = render_report(payload, input_path=args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
