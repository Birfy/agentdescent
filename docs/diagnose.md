# Diagnose — why did this run learn nothing?

`show` gives the evidence: worst rollouts (`samples`), refused candidates
(`rejected`) and the outcome histogram. A user still has to decide whether the
agent, the reflector, the gate or the scorer is the problem. `diagnose` makes
that decision explicit and names the next step.

```bash
agentdescent diagnose <run_id> [--json]
```

The verdicts are deliberately few and ordered so the strongest evidence wins:

| Verdict | Meaning | First thing to try |
|---|---|---|
| `solved` | the target was reached | review and apply |
| `partial-improvement` | candidates committed, reward still below target | more rounds/workers |
| `running` | the run has not finished | `status` / `watch`, then diagnose again |
| `cancelled` | the run was cancelled | resume if the answer is still wanted |
| `run-failed` | the run reported an error | `log.txt` / `status.json`; include `propose_error` when present |
| `reflector-error` | reflector calls failed and nothing committed | model/endpoint/key (`doctor`) |
| `scoring-mismatch` | untruncated samples' output equals gold but reward is low | inspect the scorer |
| `gate-rejected` | candidates were proposed and refused, nothing committed | read `rejected` diffs and p_improve |
| `agent-failures` | most listed samples answer wrongly | fix the skill/prompt/data |
| `no-improvement` | partial reward, no candidates and no gate rejections | raise rounds/workers |
| `unknown` | not enough evidence | `show --json` |

- `show` lists the *worst* rollouts, so "every sample failed" is not enough to
  call the agent broken. `agent-failures` requires most listed samples to fail;
  rejected candidates and committed-but-partial runs are classified first.
- `scoring-mismatch` only compares samples whose output/gold were not truncated.
- A run that committed a candidate is reported as `partial-improvement`, never
  as "no candidate was adopted".
- Running and cancelled runs are named instead of being diagnosed as finished.
- `diagnose_payload` calls `show_payload(..., diff=False)`, so it does not build
  an artifact diff it would throw away.

```python
from agentdescent import diagnose_payload
report = diagnose_payload(run_id)
print(report.verdict, report.summary, report.recommendations)
```

The same read model is exposed as the MCP `diagnose` tool and the CLI verb, so
the answer is bounded and consistent. It does not re-run the agent or reselect a
model.
