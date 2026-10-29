Added
`ComputeController` + `ComputeOption`: an action-level Thompson-sampling
allocator over direct / PRM-rerank / MCTS / verifier-ensemble strategies. It
composes `model_router` for the concrete model, respects difficulty bands,
context/tag/cost constraints and an optional call budget, and records the
chosen option, model, expected quality/cost and exploration flag for the
decision log.
