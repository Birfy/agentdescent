Added
`CapabilityLoop`: composes `TaskPipeline`, `CapabilityScout` and
`CapabilityArchive` so production traces become a validated, admitted and
archived capability in one call. The report records the pipeline verdict, scout
outcome and archive result; `admitted` is true only when the archive added the
artifact (#677).
