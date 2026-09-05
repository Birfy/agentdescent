# Evolving the engine's `task_sampler`, and what it transfers to

The outer artifact is the `task_sampler` policy — which task each rollout of an
**inner** `evolve()` spends. One outer rollout is one whole inner run: evolving
an instruction against a 20-task slice of GSM-Hard, five sweeps, one worker,
`deepseek-v4-flash` at temperature 0 with thinking disabled. The meta-reward is
the inner run's AUC (mean best-so-far held-out reward). Governance is L1, so
every merge also passes the oracle. Produced by
[`bench/metasearch_slots.py`](../metasearch_slots.py); raw record in
[`metasearch-gsm.json`](metasearch-gsm.json).

Four GSM-Hard windows to evolve on (3 seeds each = 12 outer tasks, 6 held out),
six outer sweeps, 12 rollouts. Validation is paired on 2 fresh seeds per
problem: the same windows, two GSM-Hard windows the run never saw, and two
GSM8K windows.

| group | problems | seed rule | evolved rule | gain | wins/losses |
|---|---|---:|---:|---:|---:|
| **train** | 4 GSM-Hard windows | 0.719 | **0.769** | **+0.050** | 4/1 |
| **unseen** | 2 GSM-Hard windows | 0.713 | 0.725 | +0.013 | 1/1 |
| **other** | 2 GSM8K windows | 1.000 | 0.900 | −0.100 | 0/1 |

Transfer ratio (gain over the train gain): **0.25** within the benchmark,
**−2.00** across it. The outer loop committed twice and was oracle-rejected four
times; 12 proposals, none refused by the gate, none invalid. 1,573 model calls,
855 s wall, with 555 of 1,025 completions served from cache.

The rule it found:

```python
class Policy:
    # UCB-style exploration that balances trying unknown tasks with exploiting
    # known low scores, and avoids re-picking solved tasks when alternatives exist.
    def pick(self, keys, round_index):
        unsolved = [k for k in keys if self.scores.get(k, 0) < 1.0] or keys
        ...
```

That is the mechanism the slot is about: the engine asks for **no proposal from
a rollout that passed**, so a pick landing on an already-solved task buys
nothing, and the sampler that avoids them converts more of a fixed rollout
budget into proposals.

## Read the three rows apart, not down

**The train row is the claim that holds.** +0.050 across four windows, four
wins and one loss on paired seeds, and the inner runs are deterministic (see
below), so the pairing is exact rather than noisy.

**The unseen row is weak and honest: +0.013, one win and one loss.** On this
evidence the evolved rule is closer to a fit to the windows it was evolved on
than to a better sampler in general. A transfer ratio of 0.25 is what that
looks like; it is not a null result and it is not a win.

**The other row cannot be read as transfer at all, and that is a flaw in the
validation set.** The seed rule already scores **1.000** on both GSM8K windows —
a current model solves them, which is what `docs/results.md` says about GSM8K —
so there is no headroom and the only available move is down. The −0.100 is one
of four paired runs regressing a saturated window. **A benchmark the seed
already solves cannot serve as a transfer target**, and the next run of this
should replace GSM8K with something that leaves room.

**A hand-written reference says the search left value on the table.** A
"retry the tasks that failed" sampler written by hand scores **0.784** on the
same four windows against the seed's 0.719 and the evolved rule's 0.769. The
outer loop found roughly three quarters of the available gain in 12 rollouts.

## What had to be fixed before any of this was measurable

Eight live runs. The first seven committed nothing, and each one was a
different defect rather than the same one:

| symptom | cause | fix |
|---|---|---|
| crash on the last line, whole run lost | the payload built its own usage dict with the Anthropic SDK's field names | use `examples._measure.usage_dict`; a test that drives `main()` |
| `{'oracle-rejected': 3}`, nothing commits | **the inner run was not reproducible** — the gate was cached, the rollouts were live calls, and temperature 0 was necessary but not sufficient. Validating the seed rule against *itself* reported a gain of −0.0625 | `cached_completion` memoises `prompt -> text`; every paired gain is now exactly 0.000 when the two rules are identical, and the run got 6x faster |
| a null result that could not say what it tried | proposals were not recorded | record each with whether the gate would take it; `SourceSlot.accepts()` so the reporter asks the gate instead of imitating it |
| all 12 rollouts on one problem | `meta_evolve` built its task list problem-major and `evolve()` splits by position — the search never rolled out on the problem it was judged on | interleave seed-major |
| every proposal killed the inner run with `KeyError` | proposals kept per-task state and answered from memory, not from the shard they were handed; the smoke test used one fixed key list, where that is invisible | the sampler smoke test now walks a changing shard, an all-seen shard, and unsorted lists |
| valid proposals, still nothing commits | the outer gate held out **2 windows**, and the good proposals tied the seed on exactly those two while gaining on the two it never gated on | evolve on four windows, so the gate averages over more of the distribution |

The last row is the one worth carrying to another slot: **an effect that only
appears across problems cannot be committed by a gate that sees one or two of
them**, and under L1 a tie is a veto, so a narrow gate reads as "nothing works".

## Run it

```bash
python -m bench.metasearch_slots --dry-run
python -m bench.metasearch_slots --model deepseek-v4-flash \
    --rounds 6 --workers 2 --seeds 3 --validate-seeds 2 \
    --train-windows 4 --unseen-windows 2 --other-windows 2 \
    --eval-cache .cache/metasearch --yes
```

`--eval-cache` also switches on the completion cache underneath it, which is
what makes an inner run a function of the sampler. Without it a paired gain
measures noise; the run plan says so.
