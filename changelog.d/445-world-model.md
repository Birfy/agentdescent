Added
World model (`agentdescent/world_model.py`): the framework learns an internal
simulator and dreams before it acts. `LearnedSimulator` is a pure-Python
nearest-neighbour model fit on observed `(state, reward)` pairs — deterministic,
dependency-free, and unable to extrapolate beyond what it has seen, which is
exactly the honesty a simulator of a measured domain should have;
`simulator_problem` turns the simulator into a `Problem` (a cheap, model-free
rollout); `simulator_score` measures the simulator against the real problem
(mean abs error + rank agreement), so the framework knows when to trust the
dream; and `imagine_then_evolve` spends most of its budget in the dream —
evolving against the simulator via `meta_evolve` — then verifies the value on
the real `val_problems`, reporting the dreamed vs verified rollout counts and
the real gain over the seed (`ImagineReport`). The Dreamer loop applied to the
framework: pay the real cost where it pays, dream where it is cheap.
