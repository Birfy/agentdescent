Added
`dedupe_texts` with Unicode-aware SimHash and MinHash fingerprints, clustering
near-duplicate artifacts deterministically. CJK runs contribute character
trigrams and punctuation-only text falls back to characters, so distinct rules
do not collapse; it is an archive-hygiene utility, not yet wired into the
population/QD path.
