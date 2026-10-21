Added
Trajectory tree search (`agentdescent/trajectory_search.py`): the evolution
loop becomes a learned MCTS over states. `SearchTree` is a state tree over
rendered values with UCB selection (`ucb_c`), expand and back-propagation;
`search_evolve` spends a fixed budget expanding the most promising branches —
each leaf is branched by a callable, scored by a critic (`auc` by default, or
an oracle / the self-rewarding judge) over seeds, and the value back-propagated
— and returns the highest-mean-value state with a `SearchReport` carrying the
budget, the node count, the search's own value estimate, and a **freshly
verified** score of the best state side by side, so the search's reasoning is
inspectable against final truth. The LATS / AlphaCodium idea applied to the
state space, using the same UCB bandit the tree policies in `selection` run,
one level up.
