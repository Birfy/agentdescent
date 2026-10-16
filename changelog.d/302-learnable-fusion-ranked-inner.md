Fixed
`LearnableFusionPolicy` no longer trains on an unranked union after its
tournament flag is switched on at runtime (including the engine budget
governor's re-enable) (#302). The label-producing path now always delegates
to an inner policy that actually ranks the candidates, and a merge the inner
did not rank is neither counted as a tournament run nor used as a training
label. The budget-disabled union path and the default behaviour are unchanged.
