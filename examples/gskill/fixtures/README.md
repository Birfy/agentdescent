# kvtools

A tiny vendored stand-in for "a repository" in the gskill port. Four house
conventions hold everywhere in it; the tests encode them and the code does not
say them out loud:

* ranges are half-open: `lo` is inside, `hi` is not;
* every invalid user value raises `ConfigError` -- never `ValueError`,
  `KeyError`, or `None`;
* sizes and counts are integers, and a partial page/chunk/bucket counts as one;
* a missing value is `None`, never an exception or an empty string.
