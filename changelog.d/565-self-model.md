Added
`SelfModel`, a deterministic **logit-space Newton/IRLS** Platt calibrator with
ECE/Brier reporting and calibrated answer/abstain routing. It refuses to predict
before `fit`, reports anti-correlation instead of hiding it, marks tiny samples
`insufficient`, and converges on saturated 0/1 confidences where fixed-step
gradient descent diverges.
