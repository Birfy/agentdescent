---
description: A runnable microport of TextGrad's textual-gradient prompt optimization - backward critique, TGD rewrite, validation revert - as a reflection seam. Mechanism microport on GSM8K, offline-tested.
---

# TextGrad — Textual-gradient prompt optimization

> **Reflection / refinement, by backpropagated critique.** An LLM writes a
> natural-language "gradient" on a prompt variable from how it performed, a
> second call rewrites the variable from that critique, and the rewrite is kept
> only if validation did not get worse (upstream's rule; the engine's gate is stricter, see Boundaries). Runs through the shared
> [`MethodPolicy`](policies.md) runner. Example:
> [`examples/textgrad/textgrad_prompt_optimization.py`](https://github.com/Birfy/agentdescent/blob/main/examples/textgrad/textgrad_prompt_optimization.py).

| | |
|---|---|
| **Paper** | *Optimizing generative AI by backpropagating language model feedback* — Yuksekgonul et al., *Nature* 2025 ([doi:10.1038/s41586-025-08661-4](https://www.nature.com/articles/s41586-025-08661-4)); preprint *TextGrad: Automatic "Differentiation" via Text* ([arXiv:2406.07496](https://arxiv.org/abs/2406.07496)) |
| **Upstream code** | [zou-group/textgrad@75e912e2](https://github.com/zou-group/textgrad/tree/75e912e210864b61999781778cdf756d4468120f) (MIT) |
| **Example** | [`examples/textgrad/textgrad_prompt_optimization.py`](https://github.com/Birfy/agentdescent/blob/main/examples/textgrad/textgrad_prompt_optimization.py) |
| **Domain** | **GSM8K** ([`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k), `main`), 64/64/64 splits |
| **Layer** | L1 (`blast_radius=0.6`, set by the shared runner) |
| **Fidelity** | `mechanism_microport` — [what the classes mean](port-fidelity.md) |

Upstream revision traced: `75e912e210864b61999781778cdf756d4468120f`
(`evaluation/prompt_optimization.py`, `textgrad/optimizer/optimizer.py`,
`textgrad/optimizer/optimizer_prompts.py`,
`textgrad/autograd/llm_backward_prompts.py`). The `textgrad` package is **not**
imported: the example mirrors the update rule, and the engine never depends on
it at install time.

## The mechanism

TextGrad's PyTorch analogy is literal. The prompt is a `tg.Variable`; a loss is
computed on the model's answers; `loss.backward()` asks a *gradient engine* LLM
for feedback on how the variable should change (a textual gradient, explicitly
not a new version of it); `optimizer.step()` (`TextualGradientDescent`) asks an
*optimizer* LLM to rewrite the variable from that feedback and reads the result
between `<IMPROVED_VARIABLE>` tags. The prompt-optimization script then keeps
the rewrite unless validation dropped (`run_validation_revert`: a tie is kept).

TextGrad's loss is often a free-form critique rather than a number. Here the
**gate stays a real metric** — GSM8K's last-number match on a held-out split —
and the loss text handed to the backward pass is the evaluator's reading of the
attempt (what it read, what it wanted), as upstream's `StringBasedFunction`
output is for its QA tasks.

## Where each piece lives

| Upstream mechanism | Where it lives here |
|---|---|
| `loss.backward()` on the prompt Variable | the proposal's first call: `backward_prompt`, the base-case backward prompt under `BACKWARD_SYSTEM_PROMPT` |
| `TextualGradientDescent.step()` | the second call: `tgd_prompt` under `OPTIMIZER_SYSTEM_PROMPT`; `extract_improved` reads the `<IMPROVED_VARIABLE>` span |
| The prompt `Variable` | `TextVariable`, a `ValidatedSlot` whose diff id hashes the text (per author and base version); an unchanged rewrite is a counted no-op |
| `constraints=[...]` | `CONSTRAINTS`: a rewrite may not bake this conversation's numbers into a prompt read against other questions |
| `run_validation_revert` | the engine's held-out acceptance gate (default aggregator), **which is stricter** — see the first boundary; no `aggregator_factory`, no new core contract |
| `tg.sum(losses)` over a minibatch | concurrent workers' rewrites meeting in the shared reflective merge (see below) |

The proposal is two reserved calls (`proposal_calls_per_candidate=2`): gradient,
then step.

## Boundaries

- **The gate is stricter than upstream's revert.** `run_validation_revert`
  reverts only when `val_performance < previous`, so a rewrite that **ties** the
  previous validation score is **kept**. The engine's default gate accepts only
  when the Beta posterior `P(candidate_rate > base_rate)` clears an annealed
  threshold (low at version 1, rising with the version), and a tie resolves by
  rejecting. Ties and marginal gains are refused here that upstream would keep.
  Matching upstream exactly would take a thin
  [`aggregator_factory`](aggregator-factory.md); this port documents the
  difference instead.
- **Minibatch of one per worker.** Upstream sums `batch_size=3` losses before a
  single `backward()`; here a proposal is one rollout, and the sum is replaced by
  the reflective merge of concurrent workers' rewrites. Same intent, not the same
  operator.
- **A correct attempt proposes nothing** (a free skip). Upstream's gradient
  engine is told to give no feedback to a variable that is working.
- **One prompt per call.** The completion seam is a single string, so upstream's
  system prompts are prepended to the user turn.
- **A reply without the tags is an invalid proposal.** Upstream raises
  `IndexError`; here the candidate is spent and counted, and the raw reply is
  never substituted.
- **Out of scope:** multi-module compound systems, gradient memory and momentum,
  and the molecule / code / solution-optimization demos. This ports the prompt
  Variable and the textual-gradient step only.
- No gold answer sits in a worked example: the only place it reaches the model is
  the evaluator's own "it wanted" line, as in the other reflective ports.

## Measured results — GSM8K

**Not yet measured.** This page lands with offline tests only; no live run is
recorded, so there is deliberately no number here. When one is, it goes in
[`bench/results/`](https://github.com/Birfy/agentdescent/tree/main/bench/results)
and in the
[all-ports table](self-evolution-examples.md#measured-results-all-twenty).

## Run it

```bash
python -m examples.textgrad.textgrad_prompt_optimization --dry-run

python -m examples.textgrad.textgrad_prompt_optimization --yes --seed 0 \
    --provider openai --model deepseek-v4-flash \
    --async --async-ratio 1 --workers 8 --budget-rollouts 80 --staleness full
```

`--dry-run` makes **zero network and zero model calls**: it reads the committed
96-row GSM8K slice instead of fetching (so its splits are 32/32/32 rather than
the live 64/64/64, and it says so in its plan), and does not leave the switch
on. A live run loads the real benchmark and never falls back to the slice.

Flags: [the MethodPolicy command line](self-evolution-examples.md#the-methodpolicy-command-line).

Offline tests: `tests/test_textgrad_example.py`, `tests/test_candidate_methods.py`.
