Added
Learn-to-think (`agentdescent/learn_to_think.py`): the framework evolves *when*
to spend compute. The route rule is a gated source `Policy.route(difficulty,
budget_left, reward)` returning `direct` / `search` / `reflect` (the gate
refuses imports, dunder access, forbidden calls, a wrong signature, and a
return outside the three routes; a smoke test runs the route on easy /
hard-with-budget / hard-without-budget inputs). `think_problem` is a workload
instance whose outcome depends on the route — each route has a cost model and a
quality model, and the reward is quality-per-cost — and `learn_to_think` runs a
bespoke `evolve()` over those problems (the artifact is the route source, the
propose rewrites it) so the framework learns, from its own measured returns, to
search the hard tasks and answer the easy ones directly, instead of paying a
fixed price everywhere. `LearnToThinkReport` compares the evolved rule with the
default on validation problems and shows the route each chose on an easy and a
hard input. The Search-R1 / ZigZag "learn when to search" idea, applied to the
decision plane.
