# RRSI — regularized harness self-improvement (microport + evolve() wiring)

This is a **mechanism microport** of Google Research's *Regularized Recursive
Self-Improvement of Agent Harnesses* (RRSI), wired into `evolve()` through the
engine's existing `Policies` seams. It does not claim to reproduce the paper's
benchmark results; the reported numbers are upstream's, not measured here.

## Sources and pin

- Paper: [arXiv:2609.24972v3](https://arxiv.org/abs/2609.24972v3), revised
  2026-10-04.
- Released implementation: [`google-research/rrsi` at
  `be50316e1db05914068a973f322770ef08ed7ba1`](https://github.com/google-research/rrsi/tree/be50316e1db05914068a973f322770ef08ed7ba1).
- Adapted sources: `rrsi/selection.py`, `rrsi/schedule.py`,
  `rrsi/components.py` and `rrsi/history.py`.

## What is preserved so far

### Round selection (`agentdescent/algorithms/rrsi/selection.py`)

A pure function layer for a round whose screened candidates were all measured
from the same incumbent. It applies the upstream historical-best floor
(`S_candidate >= S_best - delta`), both branches of the cost rule,
structural-component novelty, domain guards, and highest-score winner selection
(stable first candidate on ties). With no admissible candidate the caller
retains the incumbent. `update_best_score` keeps the historical best monotone.

The offline regression fixtures reject non-finite, out-of-range `[0, 1]` scores,
negative/non-finite token costs, non-finite historical bests and arithmetic
overflow. A candidate carrying a critic/smoke veto remains rejected even if a
measurement is also attached, and rejected or unevaluated candidates receive no
invented measurement in their decision record. These are defensive validation
choices around the upstream rule, not upstream parity claims.

### Edit-budget schedule (`agentdescent/algorithms/rrsi/schedule.py`)

The paper's cosine anneal on the L0 edit budget:

\[
b_t = \lceil b_{\min} + (b_{\max}-b_{\min})\cdot\tfrac12\bigl(1+\cos(\pi t / T)\bigr)\rceil,
\quad t = 0..T-1
\]

`edit_budget` / `budget_table` preserve the upstream float guard
(`round(v, 9)` before `ceil`), the `T <= 0 → b_max` degenerate case, and
clamping of `t` into `[0, T]`. Defensive validation rejects non-positive or
inverted `b_min`/`b_max` and non-integer inputs; that is an AgentDescent
boundary, not an upstream claim.

### Component vocabulary (`agentdescent/algorithms/rrsi/components.py`)

`K` is the fixed set of harness components an edit can be tagged with, and
`K_STR` (`client_tool`, `skill`, `memory`, `subagent`) the structural subset
that adds machinery. `normalize` keeps a proposer's declared tag only when the
diff carries a signal for it (domain signals first, then generic ones) and
otherwise reclassifies the diff, so a prompt tweak cannot be booked as a skill;
a diff whose changed lines are all string literals or comments is a `prompt`
edit whatever its words say. `novelty` counts structural components the
incumbent has never accepted an edit on; `selection.py` now takes its
structural set from here instead of a copy.

### Per-edit history (`agentdescent/algorithms/rrsi/history.py`)

The JSONL ledger \(L_t\): one record per *edit*, with a bundle's single
measurement copied to each of its edits. Unmeasured candidates (critic reject,
smoke failure) carry `delta_S = None`, are counted by `attempted()`, and stay
out of the tried set \(T_t\) and the yield \(g_t\). From it:

- `yield_g(t, n_prune)`: \(g_t(l) = \max\{\Delta S_i : l_i = l,\ t - t_i \le n_{prune}\}\), `-inf` when a tried component has nothing in the window.
- `prune_set(t, n_prune)`: \(B_t = \{l \in T_t : g_t(l) \le 0\}\), each with the accepted edits still in the incumbent.
- `stall_flag` (\(\sigma_t = 1[S_t - S_{t-w} \le \delta]\)) and `exploration`, which reserves `m_draft` slots for untried components \(U_t = K \setminus T_t\) when stalled.
- `render(n)`: the proposer's view, keeping only the latest few unmeasured records.

## Parity and intentional differences

| Behavior | Pinned upstream | This microport |
|---|---|---|
| Historical score floor | `S' >= S* - delta` | Preserved |
| Large-gain cost branch | `dS > delta`, require `dC <= beta0 + beta1*dS` | Preserved |
| Within-band branch | require `w_s*dS - w_c*dC + w_n*novelty > 0` | Preserved |
| Missing/zero token estimate | `dC = 0` if either side is falsy | Preserved; this can admit without telemetry |
| Invalid scores and costs | Not guarded by `selection.py` itself | Rejected before decision arithmetic |
| Critic-rejected candidate with an evaluation attached | Caller normally supplies no evaluation | Veto is authoritative and measurement omitted from decision |
| Round candidates | Evaluate from same base, choose admissible argmax | Caller contract documented; this module does not orchestrate evaluation |
| Cosine edit budget | `edit_budget` / `budget_table` | Preserved, plus defensive bound checks |
| Invalid `b_min`/`b_max` | Not guarded by `schedule.py` itself | Rejected before anneal arithmetic |
| Declared non-structural tag (`config`, `control_flow`, ...) | Kept only if a domain signal backs it; generic signals cover structural components only | Preserved |
| Tried component with no record in the prune window | `g_t = -inf`, so it is in `B_t` | Preserved |
| Candidate with no edits | One placeholder record (`C1`, no component, `bundle = 0`) | Preserved |
| Non-finite `delta_S` / `delta_C` / `S` / `C` | Written as non-standard `NaN` / `Infinity` JSON | Rejected with `ValueError` |
| Torn final JSONL line after a crash | `records()` raises; the next append extends the fragment | Ignored by `records()` (also when cut inside a multibyte UTF-8 character), dropped by the next append; invalid JSON / UTF-8 or a non-object line elsewhere raises `ValueError` with its line number |
| `replace_round` rewrite | In-place `write_text` | Atomic temp file (fsynced, original file mode kept) + `os.replace` |
| numpy scalar measurements (`np.float32`) | Written as-is | Accepted as any `numbers.Real` except `bool` |
| Text-only diff (string literals / comments) declared structural, prose mentions a signal word | Declared tag kept | Recorded as `prompt` |


## Wired into `evolve()` (`agentdescent/algorithms/rrsi/rrsi_policies.py`)

From a spec: `"algorithm": "rrsi"` (kind `text`), or
`{"name": "rrsi", "params": {"b_max": 3, "protected": ["t3"]}}`; the parameters are
`RRsiConfig`'s fields plus `protected` and `base_score` (`agentdescent algorithms
show rrsi`). The code lives in the package (it moved from `examples/rrsi/`,
whose modules are now aliases of it), so this works from the wheel. A spec run
builds the bundle for you, so its `RRsiState` is not returned; call
`build_rrsi_bundle` from Python when you need to read `S*` afterwards.

`build_rrsi_bundle(complete, protected=..., reviewer=..., config=...,
base_score=..., history_path=...)` assembles the three slots into one
`Policies` bundle and returns `(bundle, state)`. The three share one
:class:`RRsiState` — `S*`, the measured per-edit history, the component
bookkeeping and the annealed round counter — which is what makes them one
algorithm rather than three knobs.

**Fidelity.** The wiring consumes the pinned fixtures, not approximations:
component attribution goes through :mod:`~agentdescent.algorithms.rrsi.components` (the
upstream `K` / `K_STR` / `normalize` evidence rules — `skills/x/SKILL.md` is a
`skill` edit, `ToolRegistry(...)` is `client_tool`, pure prose is `prompt`),
and the per-edit ledger is :class:`~agentdescent.algorithms.rrsi.history.History` (the pinned
durable JSONL) when `history_path` is given, with an in-memory twin for scratch
runs pinned to the same summaries by a parity test. The search state (round,
`S*`, incumbent cost, progress) rides the aggregator checkpoint; the per-edit
evidence is the History's own file, so a resumed run reads the same evidence
the loop conditioned on.

| Slot | Policy | What it regularizes |
|---|---|---|
| `Policies(screen=...)` | `RRsiCritic` | Pre-evaluation screen: annealed edit-budget gate (`len(ops) > b_t` rejected) and deterministic leakage screening against the evolve set (`protected` strings), plus an optional model `reviewer`. Runs **before measurement** (issue #221: "screening before evaluation is important"), so a leaking or over-budget card is settled and reported as `critic-rejected` without a held-out sweep. |
| `Policies(proposal=...)` | `RRsiProposalPolicy` | The regularized proposer: injects the annealed budget, the measured per-edit history (`History.render`), pruning targets with the actionable accepted edits (`History.prune_set`) and the stall-exploration note (`History.stall_flag`) into the reflection prompt. |
| `Policies(acceptance=...)` | `RRsiAcceptance` | The pinned `selection.judge` decision — historical-best floor, both cost-rule branches, novelty over `History.incumbent_component_counts` — with `S*` advanced through the pinned `update_best_score`. `S*`, the round counter and the progress trajectory checkpoint/restore through the aggregator's optional acceptance hook, so a resumed run does not lose the floor. |

### Declared deviations (recorded, not silent)

* **One candidate per round.** The engine merges to one candidate per round,
  so the multi-candidate `select_round` "choose highest admissible argmax"
  step is not exercised; each merge's candidate is judged against the
  incumbent and `S*`. This is the "one winner per round" semantics issue #221
  asks to establish first.
* **The round counter advances once per merge.** On the synchronous path that
  is the round barrier; on the barrier-free path there is no round barrier, so
  the annealed schedule anneals per merge instead of per RRSI round. Callers
  who need the paper's round semantics run the synchronous path.
* **`noise_band` is a configured parameter** (default `0.0`, the conservative
  floor `score >= S*`). The paper estimates it by repeatedly evaluating the
  unchanged base harness; set `RRsiConfig.noise_band` from that measurement.
  It is a score delta in `[0, 1]`; `spec.algorithm` params outside that range
  fail in `plan`.
* **Cost is `EvidenceCard.cost_tokens`, averaged per rollout** (the proposing
  rollout's usage, when the caller shares a `Usage` with its adapters), **not a
  batch sum** -- summing across a merge's cards would make the cost rule depend
  on how many cards happened to land in the bucket. This is still a proxy:
  Algorithm 2's `dC` is the *candidate* harness's cost relative to the
  incumbent's on the same evaluation, and the proposing rollout runs the
  incumbent, so the cost rule is approximate until candidate-side held-out
  telemetry exists. Missing or zero telemetry yields `dC = 0`, matching the
  pinned upstream behaviour.
* **Protected strings come from the evolve set only** — never the final
  holdout. The integration tests enforce that discipline by construction.

What is still not wired: a full harness-directory domain adapter, multi-round
`select_round` trial aggregation, and a RRSI CLI. AgentDescent's default
evolution behavior is unchanged — a run without the bundle reports zero
`critic_checked` / `critic_rejected` and no `critic-rejected` /
`noise-floor` / `cost-rule` outcomes.

## Local verification

```bash
pytest -q tests/test_rrsi_selection.py tests/test_rrsi_schedule.py tests/test_rrsi_history.py tests/test_rrsi_integration.py
```

Selection goldens cover the historical floor, both cost branches, domain veto,
ties, no winner, missing cost telemetry, rejected/missing candidates and invalid
numeric inputs. Schedule fixtures cover endpoints, monotonic anneal, the float
guard, clamping, degenerate horizons and invalid bounds. History fixtures
cover bundle fan-out, unmeasured records, the prune window and \(B_t\),
`render`'s cap, torn-line recovery, atomic rewrites, stall and exploration, and
`normalize`'s evidence check. Integration fixtures
cover the budget/leakage screen before evaluation, the model reviewer being
asked only when deterministic checks pass, history attribution, stall
exploration, pruning targets, `S*` monotonicity and checkpoint/restore, and
offline end-to-end runs (leak → `critic-rejected`, clean → commit, untouched →
zeros).
