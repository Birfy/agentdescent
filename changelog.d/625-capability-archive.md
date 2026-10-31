Added
`CapabilityArchive`: a quality-diversity archive, artifact graph and semantic
dedup kept in sync. A near-duplicate never occupies a QD cell or the graph; an
elite replacement removes the evicted graph node and fingerprint; `add` rolls
the QD cell back to the previous elite if the graph rejects the artifact;
prepopulated graphs seed the dedup fingerprints. Retrieval goes through the
graph and selection through the QD archive.

Changed
`ArtifactGraph.remove()` removes an artifact and decrements document frequency
once per distinct token. Descendants keep their parent ids and treat the
removed node as external.
