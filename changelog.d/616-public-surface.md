Changed
`LoopKernel`, `promote` / `PromotionRules` and `ComputeController` are now
re-exported from the top-level `agentdescent` package, listed in the generated
API reference and documented with the loop, promotion and compute guides. The
modules stay usable through their submodules as before. An explicit
`approved=False` now rejects even when `require_approval` is off, matching the
documented fail-closed rule.
