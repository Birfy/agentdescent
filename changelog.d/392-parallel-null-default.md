Fixed
`plan` again accepts `evolve.parallel: null` (the supported engine default) after #385's object-knob guard rejected every value for that key; null on an object-typed evolve knob means "use the engine default" and no longer clears objects `compose` assembled (`#392`).
