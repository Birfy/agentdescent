Added
Rule tournament (`agentdescent/rule_tournament.py`): candidates play against
each other, ratings decide promotion. `rule_tournament` runs a round-robin
where every pair plays twice on each other's ground — rule A is scored by a
critic on rule B's challenge problem (rebuilt from B's descriptor via
`problem_from_archive`, or a caller-supplied `challenges` map) and rule B on
A's — and `EloRating` (K-factor 32, rating mass conserved, draws split) converts
the matches into ratings. A rule that wins on its rival's ground is a better
rule, and `tournament_promote` re-archives the winner with its measured final
reward (its mean critic score on rivals' ground), so the next session's
`_archived_rules` ranks the tournament's choice first. The self-play /
verifiable-reward idea (DeepSeek-R1, self-playing agents) applied to the
archive; a tournament with fewer than two rules is reported as no tournament.
