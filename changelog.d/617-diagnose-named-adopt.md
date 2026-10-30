Fixed
`diagnose` and the guarded-run decision log treat named-algorithm adopt
categories (`gepa-select`, `skillopt-accept`, `dgm-select`) and per-round
`committed` totals as adoptions, so a successful GEPA/SkillOpt/DGM run is
no longer reported as "no candidate was adopted" (#617).
