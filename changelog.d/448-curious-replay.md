Added
Failure-weighted replay (`agentdescent/curious_replay.py`): the framework
replays the past it is still failing. `ReplayExample` is one replayed past task
with its rolling scores; `curiosity_weight` is high while the artifact is still
failing it (falls as it improves, floor once mastered); `curious_replay_evolve`
runs the ordinary `evolve()` over one task per replayed example, weighted by a
**snapshot** of the weights taken before any candidate rolls out, with the
mastered and still-curious cohorts interleaved so both land on both sides of
the held-out cut; each outcome is recorded into the examples' rolling scores
for the next call, and a `CuriousReplayReport` shows the weights and the gain
on the still-curious vs mastered past. The Curious Replay (ICLR 2025) idea:
spend the budget on the past not yet conquered, measured not assumed.
