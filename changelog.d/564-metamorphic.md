Added
`MetamorphicSuite`, `MetamorphicRelation` and `invariance` check output
relations under input transforms, so a candidate that scores higher by learning
surface form can be caught by a broken invariance. The suite is an acceptance
probe, not yet wired into the evolution loop; `accept(min_rate=...)` supports a
bounded failure rate.
