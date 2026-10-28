Added
`RSITelemetry`, an OTel-inspired span/event tree for RSI runs
(`run → round → rollout → merge → gate → commit`) with unique IDs, attributes,
events, exception/cancellation status, wall vs span time, escaped Prometheus
samples, and OTel-shaped dictionaries (not OTLP JSON).
