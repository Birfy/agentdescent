"""What a task has to supply for the ERA tree search to run on it.

`futs.search` is indifferent to what it is searching over: it ranks nodes by a
score it never interprets, expands the one PUCT picks, and appends whatever
comes back. Upstream demonstrates that on one task and hard-codes it; this
struct is the seam that keeps the loop honest across two.

Nothing algorithmic lives here. A :class:`Domain` is the four things the search
cannot invent -- the program it starts from, the way a program is scored, the
prompt that rewrites one, and the name of the number being reported -- and
`era_empirical_software.py` holds everything else.

The default, :func:`~examples.era.era_empirical_software.s3e1_domain`, is the
upstream Kaggle task, so a port that names no domain is the port upstream ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Sequence, Tuple


@dataclass(frozen=True)
class Domain:
    """One scientific task, as the ERA search sees it."""

    #: Printed in the run plan and stored in the result file.
    name: str
    #: The function a candidate program must define.
    entrypoint: str
    #: The key this task's metric occupies in a node's `metrics` dict, and the
    #: field name it is reported under. `rmse` for the Kaggle task,
    #: `mean_digits` for the integrals -- reported under its own name so a
    #: result file says which number it is holding.
    metric_key: str
    #: "lower" or "higher". Only the sign of the reported gain depends on it;
    #: node ordering is always by `metrics["score"]`, which each task supplies
    #: already oriented so that larger is better, as `futs.search` requires.
    metric_better: str
    #: The root node's program and the summary shown for it.
    initial_program: str
    initial_summary: str
    #: `(code, shards) -> (valid, metrics, error)`, sandboxed.
    evaluate: Callable[[str, Sequence[int]], Tuple[bool, Dict[str, Any], str]]
    #: `metrics -> [0, 1]`, order-preserving with `metrics["score"]`.
    reward: Callable[[Dict[str, Any]], float]
    #: `(parent_program, recalled) -> prompt`, upstream's
    #: `PlaygroundGenerator.__call__`. ``recalled`` is the tree's recall
    #: snapshot -- `(metrics, score)` pairs for nodes it already holds, best
    #: first -- and it is **empty unless a run asks for it**, which is upstream:
    #: there a mutation sees one parent and one score. A domain that wants that
    #: view ignores the second argument.
    prompt: Callable[..., str]
    #: `shard_index -> task prompt`, for the rollout tasks.
    task_prompt: Callable[[int], str]
    #: The shards the search never sees, scored once at the end.
    test_shards: Tuple[int, ...]
    #: Whatever the run should record about the data it ran on.
    data_summary: Dict[str, Any] = field(default_factory=dict)

    def gain(self, baseline: Any, best: Any) -> Any:
        """The improvement in this task's own metric, signed so that up is good."""
        if baseline is None or best is None:
            return None
        if self.metric_better == "lower":
            return float(baseline) - float(best)
        return float(best) - float(baseline)
