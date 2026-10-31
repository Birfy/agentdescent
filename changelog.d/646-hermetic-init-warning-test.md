Fixed
The `init` starter-reflector warning test no longer fails on a machine that
exports `ARK_API_KEY` or another provider key: it clears every name `init`
reads before pinning the "OPENAI_API_KEY is not set" warning (#646).
