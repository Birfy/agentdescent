Fixed
`agentdescent algorithms run NAME PATH` no longer answers a kind mismatch with
the `init --algorithm` text: the error names `algorithm 'NAME'` (the name is
positional), routes the one-file fix through `algorithms run NAME FILE --kind
text`, and, when `--kind` was not given, says a file is always `text` or that a
directory's kind was guessed and `--kind` overrides it (#702).
