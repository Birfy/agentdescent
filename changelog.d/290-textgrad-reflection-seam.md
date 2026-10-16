Added
`examples/textgrad/`: a `mechanism_microport` of TextGrad
(zou-group/textgrad@75e912e2) as a reflection / propose seam. A proposal is a
textual-gradient call (the base-case backward prompt) followed by a TGD rewrite
read from `<IMPROVED_VARIABLE>` tags, applied to one prompt slot whose diff id is
content-addressed (an unchanged rewrite is a counted no-op); acceptance is the
default held-out gate, so no core contract changes. Registered in
`bench.candidate_methods`, with `docs/algo-textgrad.md` and an offline
`tests/test_textgrad_example.py`; `--dry-run` needs no network and no API key.
The reflection family no longer lists TextGrad as unimplemented (#290).
