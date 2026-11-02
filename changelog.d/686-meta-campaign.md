Added
`meta_campaign_run` turns continuous meta-learning into a bounded campaign:
`RSIController` stop conditions wrap repeated `continuous_meta_run` sessions
that share one durable `ConsolidationStore`, each in its own fresh ledger repo,
so every session seeds from the adopted rule the last one archived. A failed
session is reported and stops the campaign instead of being promoted
(#686).
