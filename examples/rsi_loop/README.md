# five-layer RSI demo

A self-contained, offline run of the five integration layers:

| Layer | What the demo shows |
|---|---|
| `LoopKernel` | three durable generations, decision log, telemetry, `apply` |
| `ComputeController` | per-generation choice between direct / PRM rerank / search |
| `promote` | eval + canary evidence becomes one promotion decision |
| `CapabilityFoundry` | a synthesized tool admitted through review, safety, proof and graph registration |
| `ExperienceEngine` | the loop's generations become replay + a guarded meta-policy update |

```bash
python -m examples.rsi_loop.rsi_loop_demo
```

It uses a temporary directory for the SQLite run store and decision log, makes
no network calls, and prints a JSON summary. The companion test is
[`tests/test_rsi_loop_example.py`](../../tests/test_rsi_loop_example.py).
