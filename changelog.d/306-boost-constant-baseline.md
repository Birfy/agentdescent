Fixed
`VerifierBoost.retrain()` no longer activates a calibration curve that
only shifts the verifier's mean: the bootstrap gate must now also beat the best
constant predictor (the train slice's mean oracle score) on the validation
slice by `min_gain`, so a verifier with no discriminating power that merely
reads systematically high or low is rejected ("no discrimination beyond a
constant") instead of being "validated" and re-ranking candidates. A curve with
fewer than two distinct fitted levels is rejected too, and `status()` and
`BoostState` expose the new `error_const` baseline (#306).
