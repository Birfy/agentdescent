Added
Third-party packages can register a named algorithm through the
`agentdescent.algorithms` entry-point group (`Algorithm` or a zero-argument
factory returning one). A plugin that fails to load is recorded in
`agentdescent.algorithms.external_errors()` and does not hide the shipped
registry; a name that duplicates a built-in is an error, not a shadow.
