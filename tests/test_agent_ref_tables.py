"""``init``'s agent ref tuples stay in step with ``evolvespec`` (#747).

``init`` / ``algorithms run`` refuse an ``--agent`` outside
``cli._WORKSPACE_REFS + cli._PLAIN_REFS`` before a spec is written (#745), and
use the split to refuse a plain model for a directory kind (#456). A short ref
added to ``SHORT_REFS`` but not to these tuples would be refused by ``init``
although ``plan`` accepts it; one filed under the wrong tuple would get the
wrong kind check.
"""

import importlib

from agentdescent import cli
from agentdescent.agents import WorkspaceAgent
from agentdescent.evolvespec import SHORT_REFS, WORKSPACE_AGENT_REFS

_AGENT_MODULES = ("agentdescent.agents", "agentdescent.host_sampling")


def _factory(ref):
    module, _, attr = SHORT_REFS[ref].partition(":")
    return getattr(importlib.import_module(module), attr)


def test_tuples_cover_exactly_the_agent_short_refs():
    agents = cli._WORKSPACE_REFS + cli._PLAIN_REFS
    assert len(set(agents)) == len(agents)
    from_table = {name for name, target in SHORT_REFS.items()
                  if target.partition(":")[0] in _AGENT_MODULES}
    assert set(agents) == from_table


def test_workspace_tuple_matches_the_plan_phrase():
    assert tuple(WORKSPACE_AGENT_REFS.split(" / ")) == cli._WORKSPACE_REFS


def test_workspace_refs_build_workspace_agents():
    for ref in cli._WORKSPACE_REFS:
        # Construction only: no binary is looked up until the agent is called.
        args = (["true"],) if ref == "cli_agent" else ()
        assert isinstance(_factory(ref)(*args), WorkspaceAgent), ref


def test_plain_offline_ref_is_not_a_workspace_agent():
    # The other plain refs need a key, an SDK or a host session to build.
    assert "echo" in cli._PLAIN_REFS
    assert not isinstance(_factory("echo")(), WorkspaceAgent)
