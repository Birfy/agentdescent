Added
`GuardPipeline`, one ordered acceptance gate that composes the checks a caller
supplies and names the ring that refused a candidate. It keeps the inner
guard's merge category (falling back to `below-threshold`) instead of inventing
a category the aggregator cannot decode, reports the first refusal even with
`collect_all`, and forwards `bind`/`configure` plus checkpoint, restore and
screen-reject hooks to every guard. It imports no specific guard, so third-party
checks are first-class.
