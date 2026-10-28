Added
`ReplayLab`, a deterministic counterfactual replay layer over a session trace:
remove/replace/disable interventions, leave-one-out effects, and Shapley
attribution through `credit`. The cache is value/metadata-based (or caller-keyed,
or disabled for unhashable payloads), and multi-step results carry the real
intervention tuple.
