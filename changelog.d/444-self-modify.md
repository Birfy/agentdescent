Added
Self-modifying framework (`agentdescent/self_modify.py`): the framework evolves
patches to its own code, gated by its own tests. `patch_source` / `apply_patch` turn a candidate into a
unified diff and back (a patch that cannot apply is refused); `gate_source` is
the same AST walk every other self-edit in the framework runs (no imports, no
dunder, no interpreter calls — a candidate compiles in a namespace of safe
builtins only); `evaluate_source` runs the candidate against callable tests and
a benchmark, where **every test must pass before any benchmark score counts** —
a patch that breaks anything is a zero, never a regression with a number
attached; and `improve_source` runs the ordinary `evolve()` over `(test, seed)`
tasks, so the framework searches its own code under its own gate and reports
the accepted patch, the tests it passed, and the measured gain over the seed
(`SelfModifyReport`).
