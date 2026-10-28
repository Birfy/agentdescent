Fixed
`agent_society._is_literal` no longer references deprecated `ast.NameConstant`
(removed in Python 3.14); `True`/`False`/`None` are already `ast.Constant` on
the supported 3.9+ range (#591).
