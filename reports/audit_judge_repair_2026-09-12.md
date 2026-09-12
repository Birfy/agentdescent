# Judge repair -- can a prompt fix it, and can you tell? (2026-09-12)

Phase 0 on BBH found the judge had stopped *discriminating* on label-shaped answers, not merely become generous -- a failure `delta_hat` cannot tell from the benign one. The hypothesis is that "ignore formatting, grade the meaning" is degenerate when the reference answer *is* a bare label.

| | |
|---|---|
| model | deepseek-v4-flash |
| arms | control, labelled |
| evaluation | re-scoring **stored** outputs; no rollouts |
| model calls | 394 (0+0 tokens) |
| wall clock | 659s |

## The noise floor

The control arm re-scores the same outputs with the **same** prompt. Its disagreement with the stored scores is the judge disagreeing with itself, and no other arm's improvement means anything below it.

| workload | n | flips vs the stored run | as a rate | `sigma` of the *same* prompt |
|---|---|---|---|---|
| `bbh` | 49 | 1 | 2.0% | 0.4738 -> 0.4657 (**improved**) |
| `hotpot` | 177 | 9 | 5.1% | 0.3812 -> 0.3859 (worsened) |

!!! danger "The control improved on itself"
    On `bbh` the **unchanged prompt** scored a smaller residual than the run that produced the stored scores. Nothing was fixed; the judge answered differently.

    This is not a display artifact, it is the finding. A `sigma` that fell is not evidence on its own when the verifier is stochastic, and the control arm is the one row that can demonstrate that -- it has no floor to be read against, because it *is* the floor.

    Read every other row's `helps` as "cleared this", and read this row as how little that means.

## Every arm, on both workloads

`sigma` is the target. `delta` is reported and never scored: a mean goes to zero when errors cancel.

`helps` reads the residual **and** the noise floor: an arm that moved no more units than re-running the same prompt does has not been shown to do anything.

| arm | workload | n | `sigma` | `delta` | disagree | fixed | broke | moved | floor | helps |
|---|---|---|---|---|---|---|---|---|---|---|
| `control` | `bbh` | 49 | 0.4738 -> **0.4657** | +0.3265 -> +0.3061 | 0.327 -> 0.306 | 1 | 0 | 1 | -- |*is the floor* |
| `control` | `hotpot` | 177 | 0.3812 -> **0.3859** | +0.1751 -> +0.1808 | 0.175 -> 0.181 | 4 | 5 | 9 | -- |*is the floor* |
| `labelled` | `bbh` | 49 | 0.4738 -> **0.4461** | +0.3265 -> +0.2653 | 0.327 -> 0.265 | 3 | 0 | 3 | 1 | yes |
| `labelled` | `hotpot` | 177 | 0.3812 -> **0.3660** | +0.1751 -> +0.1582 | 0.175 -> 0.158 | 6 | 3 | 9 | 9 | **no** |

## Does it still rubber-stamp?

Forgive every formatting difference the judge is *told* to forgive -- compare option labels alone -- and ask whether it still says yes where that says no. A merely generous judge scores near zero here.

| arm | labelled units | lenient-correct | judge says right | rubber-stamped |
|---|---|---|---|---|
| `control` (`bbh`) | 20 | 10 | 16 | **6/20 = 30%** |
| `labelled` (`bbh`) | 20 | 10 | 14 | **4/20 = 20%** |

## The one clause

```
One exception: when the reference answer is a multiple-choice label such as (A) or (C), the candidate is correct only if it chooses that same option -- the label is the answer, not its formatting.
```

One sentence, added to the shipped template, nothing else changed. A prompt is a bundle by nature and a bundle launders whatever is in it, so a rewrite would have produced a number nobody could attribute.

## Reproduce

```bash
python -m scripts.audit_judge_repair --model deepseek-v4-flash
```
