Added
`DeferralPolicy`, a learned confidence threshold for `auto-approve / defer /
deny`: it fits from past reviewer outcomes under a target auto-approve risk,
honours a hard-risk deny ceiling, refuses small calibration sets, and makes the
defer-all fallback genuinely defer everything (including confidence 1.0).
