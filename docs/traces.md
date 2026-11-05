# Trajectory evidence — a digest a reflector can use

A run's trajectory is the best evidence the loop has, and the hardest thing to
hand a reflector: a coding-agent run easily produces millions of tokens of
prompts, tool calls, errors and scores. `agentdescent traces` turns that stream
into a **layered, sourced digest** — counts by run and task, normalised failure
signatures, and a token-bounded text block — while keeping the event ids needed
to drill back into the raw trace.

```bash
agentdescent traces digest --events runs/<run_id>/trajectories.jsonl \
  --max-chars 4000 --json --out digest.json

agentdescent traces show --events runs/<run_id>/trajectories.jsonl \
  --signature sig_ab12cd34ef56 --json
```

The same operation is the read-only MCP tool `traces_digest`; pass `signature`,
`run_id` or `task_id` to drill down instead of summarising.

## Input

One JSON object per line (or a JSON array). Only `run_id` is required; the rest
has defaults so a partial log is still usable:

```json
{"run_id": "r1", "task_id": "t1", "step": 0, "kind": "prompt",
 "text": "solve 2+2"}
{"run_id": "r1", "task_id": "t1", "step": 1, "kind": "tool",
 "name": "bash", "text": "timeout after 3s", "payload": {"status": "timeout"}}
{"run_id": "r1", "task_id": "t1", "step": 2, "kind": "error",
 "text": "Traceback ValueError: 42"}
```

- `kind` is free-form; `tool_call`/`tool_result` fold to `tool`, `failed`
  folds to `error`.
- `text` may also be `output`, `message`, `content`, `error`, `prompt`,
  `completion`, `result` or `detail`.
- A top-level `reward` or `score` in `[0, 1]` is carried into the run/task
  layer and the failure threshold; `value` is not treated as a reward.
- `payload` is copied into the event after key-based secret redaction; common
  secret **values** in text (`sk-…`, `ghp_…`, `github_pat_…`, `Bearer …`) are
  masked too.
- A malformed line is an input error with its line number, not a half digest.

## The layers

| layer | what it answers |
|---|---|
| run | how many events/tasks/failures, the best reward and the time span |
| task | outcome (`succeeded` / `partial` / `failed` / `unknown`), reward and event ids |
| signature | one normalised failure mode, its count, the runs/tasks and the source event ids |
| text | a prompt-ready block, truncated to `--max-chars` |

Failure messages are normalised before grouping: URLs, paths, hex ids and
numbers become `<url>`, `<path>`, `<hex>` and `#`, so `timeout after 3s` and
`timeout after 7s` are one signature rather than two. A signature id is stable
for the same `(kind, name, message)`.

```text
trace digest: 2 run(s), 3 task(s), 7 event(s); 2 failure signature(s); top: error x2
run r1: 5 events, 2 tasks, 3 failures, reward=1.0
run r2: 2 events, 1 task, 2 failures, reward=0.0
failure signatures:
- sig_… error x2: traceback valueerror: # | sample: Traceback ValueError: 42
- sig_… tool/bash x2: timeout after # | sample: timeout after 3s
```

## Drill-down

`traces show` returns the raw events behind a selected slice, capped by
`--limit`. `--signature` resolves the signature from the raw trace even when it fell
outside the digest's `--max-signatures` top-N, and `--limit` bounds the events
returned (the signature's `count` may be larger). `--run-id` / `--task-id`
filter the stream.

```bash
agentdescent traces show --events traces.jsonl --signature sig_ab12cd34ef56
```

## What it does not do

- It does not call a model, fetch a URL or write anything except `--out`.
- It is not the sparse audit (`audit`) and does not correct a verifier's bias;
  it only makes the trajectory legible to a reflector.
- It does not hide a truncated sample: the digest reports `truncated: true` and
  keeps source event ids so the caller can expand the exact event.
