Added
`CapabilityScout`: packages failed `TaskCandidate`s as `ToolSynthesisRequest`
examples, runs the existing `ToolSynthesizer`, and hands the selected candidate
to the caller's `CapabilityRequest` builder before `CapabilityFoundry.admit`.
The report makes `no-candidate` and `no-request` explicit; only an admitted tool
reaches `promoted`.
