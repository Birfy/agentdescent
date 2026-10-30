Added
`CalibratedDeferral`: fits `SelfModel` on raw confidence/outcomes, then fits
`DeferralPolicy` on the **calibrated** confidence and risk. `decide` returns
answer / defer / deny with both raw and calibrated confidence and the hard-risk
ceiling still enforced. The report carries the calibration and deferral reports
together.
