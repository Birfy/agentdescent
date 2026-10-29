"""Agent society has to be right where the framework evolves a topology.

A society is a graph of agent nodes with delegation edges; the whole graph is
one evolvable artifact; and `society_evolve` searches it on correctness-minus-
cost. The fixtures pin the surfaces that make this honest: a chain routes and
costs correctly, the gate refuses unsafe sources, a two-node delegation beats
the single-agent baseline on a two-step task, and the run reports the gain of
the evolved topology.
"""

import pytest

from agentdescent import (
    AgentNode,
    SocietyGraph,
    SocietyReport,
    society_evolve,
    society_problem,
    society_spec,
)


DOUBLE = "def society():\n    return {'nodes': {'double': {'budget': 1, 'code': 'lambda x: x * 2'}}, 'edges': [], 'root': 'double'}\n"
CHAIN = """def society():
    return {
        'nodes': {
            'double': {'budget': 1, 'code': 'lambda x: x * 2'},
            'plus1': {'budget': 1, 'code': 'lambda x: x + 1'},
        },
        'edges': [('double', 'plus1')],
        'root': 'double',
    }
"""


def _gold(x):
    return x * 2 + 1                     # needs both nodes


TASKS = {"a": 1, "b": 2, "c": 3, "d": 4}
VAL_TASKS = {"x": 5, "y": 6, "z": 7}   # held-out inputs, disjoint from training


def test_a_chain_routes_and_costs():
    spec = society_spec()
    chain = spec.compile(CHAIN)
    out, used = chain.solve(1)
    assert out == 3 and used == 2
    single = spec.compile(DOUBLE)
    out1, used1 = single.solve(1)
    assert out1 == 2 and used1 == 1      # single node: misses the +1 step


def test_is_literal_emits_no_nameconstant_deprecation():
    # #591: naming ast.NameConstant in isinstance fired DeprecationWarning on
    # 3.12/3.13 and will AttributeError on 3.14; True/False/None are Constant.
    import ast
    import warnings

    from agentdescent.agent_society import _is_literal

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert _is_literal(ast.parse("True", mode="eval").body)
        assert _is_literal(ast.parse("False", mode="eval").body)
        assert _is_literal(ast.parse("None", mode="eval").body)
        assert _is_literal(ast.parse("{'ok': True, 'n': None}", mode="eval").body)
    assert not any("NameConstant" in str(w.message) for w in caught)


def test_is_literal_accepts_unary_signed_numbers():
    # #593: -1 / +2 are UnaryOp on 3.9+, and literal_eval accepts them; the
    # gate used to call them "computed code".
    import ast

    from agentdescent.agent_society import _gate_society, _is_literal

    assert _is_literal(ast.parse("-1", mode="eval").body)
    assert _is_literal(ast.parse("+2", mode="eval").body)
    assert _is_literal(ast.parse("-0.5", mode="eval").body)
    assert _is_literal(ast.parse("-1j", mode="eval").body)
    assert _is_literal(ast.parse("{'budget': -1, 'xs': [-2, +3]}", mode="eval").body)
    # Still refuse what literal_eval refuses.
    assert not _is_literal(ast.parse("~1", mode="eval").body)
    assert not _is_literal(ast.parse("--1", mode="eval").body)
    assert not _is_literal(ast.parse("-True", mode="eval").body)
    signed = (
        "def society():\n"
        "    return {'nodes': {'n': {'budget': -1, 'code': 'lambda x: x'}}, "
        "'edges': [], 'root': 'n'}\n"
    )
    _gate_society(signed)  # must not raise "not computed code"


def test_the_gate_refuses_unsafe_societies():
    spec = society_spec()
    with pytest.raises(ValueError):
        spec.compile("import os\ndef society():\n    return {'nodes': {}}\n")
    with pytest.raises(ValueError):
        spec.compile("def society():\n    return eval('1')\n")


def test_the_gate_refuses_a_dunder_in_a_node_s_code():
    # the node code is a string constant invisible to the outer AST walk;
    # the nested gate must refuse it before eval
    spec = society_spec()
    attack = ("def society():\n"
              "    return {'nodes': {'a': {'budget': 1, 'code': "
              "\"lambda x: [c for c in ().__class__.__base__.__subclasses__() "
              "if c.__name__ == '_wrap_close'][0].__init__.__globals__['system']\""
              "}}, 'edges': [], 'root': 'a'}\n")
    with pytest.raises(ValueError):
        spec.compile(attack)
    with pytest.raises(ValueError):
        spec.compile("def society():\n"
                     "    return {'nodes': {'a': {'budget': 1, 'code': "
                     "\"lambda x: x.__class__\"}}, 'edges': [], 'root': 'a'}\n")


def test_the_outer_society_body_is_data_not_code():
    # the society() body used to be exec'd, so a frame-escape inside it could
    # reach the interpreter; it is now required to be a literal
    spec = society_spec()
    escape = ("def society():\n"
              "    box = []\n"
              "    g = (box[0].gi_frame.f_back.f_back.f_globals for _ in [1])\n"
              "    box.append(g)\n"
              "    G = list(g)[0]\n"
              "    os_ = G['__builtins__']['__import__']('os')\n"
              "    os_.system('echo PWNED > /tmp/pwned446')\n"
              "    return {'nodes': {'a': {'budget': 1, 'code': 'lambda x: x'}}, "
              "'edges': [], 'root': 'a'}\n")
    with pytest.raises(ValueError):
        spec.compile(escape)
    with pytest.raises(ValueError):
        spec.compile("def society():\n"
                     "    return eval('{}')\n")
    with pytest.raises(ValueError):
        spec.compile("def society():\n"
                     "    nodes = {}\n"
                     "    return nodes\n")      # not a literal body


def test_a_node_s_budget_limits_its_uses():
    spec = society_spec()
    # budget 0 must not run at all; budget 1 runs once and stops
    zero = spec.compile("def society():\n"
                        "    return {'nodes': {'a': {'budget': 0, "
                        "'code': 'lambda x: x + 1'}}, 'edges': [], 'root': 'a'}\n")
    out, used = zero.solve(1)
    assert (out, used) == (1, 0)
    once = spec.compile("def society():\n"
                        "    return {'nodes': {'a': {'budget': 1, "
                        "'code': 'lambda x: x + 1'}}, 'edges': [], 'root': 'a'}\n")
    out, used = once.solve(1)
    assert (out, used) == (2, 1)


def test_solve_walks_a_chain_and_stops_on_a_cycle():
    spec = society_spec()
    # fan-out edges: only the first successor is walked (documented chain)
    fan = spec.compile("def society():\n"
                       "    return {'nodes': {'a': {'budget': 1, "
                       "'code': 'lambda x: x + 1'}, "
                       "'b': {'budget': 1, 'code': 'lambda x: x * 10'}, "
                       "'c': {'budget': 1, 'code': 'lambda x: x * 100'}}, "
                       "'edges': [('a', 'b'), ('a', 'c')], 'root': 'a'}\n")
    out, used = fan.solve(1)
    assert (out, used) == (20, 2)             # a -> b (first edge); c never runs
    # a cycle stops at the repeated node instead of spinning
    cyc = spec.compile("def society():\n"
                       "    return {'nodes': {'a': {'budget': 5, "
                       "'code': 'lambda x: x + 1'}, "
                       "'b': {'budget': 5, 'code': 'lambda x: x + 1'}}, "
                       "'edges': [('a', 'b'), ('b', 'a')], 'root': 'a'}\n")
    out, used = cyc.solve(0)
    assert used == 2                          # a, b, then back to a -> stop


def test_the_problem_rewards_correctness_minus_cost():
    spec = society_spec()
    chain = spec.compile(CHAIN)
    problem = society_problem(TASKS, gold=_gold, lam=0.05)
    outcome = problem(chain, 0)
    assert outcome.final == pytest.approx(0.9)     # 1 - 0.05*2
    single = spec.compile(DOUBLE)
    assert problem(single, 0).final == 0.0          # wrong answer


def test_a_delegating_society_beats_the_single_agent_baseline():
    spec = society_spec()
    problem = society_problem(TASKS, gold=_gold, lam=0.05)
    chain, single = spec.compile(CHAIN), spec.compile(DOUBLE)
    assert problem(chain, 0).final > problem(single, 0).final


def test_society_evolve_finds_the_delegation():
    train = {"t": society_problem(TASKS, gold=_gold, lam=0.05)}
    val = {"v": society_problem(VAL_TASKS, gold=_gold, lam=0.05)}

    def chain_propose(rendered, task, output, reward):
        return CHAIN

    report, result = society_evolve(
        train, val, propose=chain_propose,
        seeds=(0, 1, 2, 3), rounds=1, n_workers=1, repo_path=None)
    assert isinstance(report, SocietyReport)
    assert report.evolved_score > report.seed_score
    assert report.gain > 0
    assert "plus1" in report.value
    assert result.meta["gain"] == pytest.approx(report.gain)


def test_society_evolve_needs_a_proposer():
    with pytest.raises(ValueError):
        society_evolve({"t": society_problem(TASKS, gold=_gold)},
                       {"v": society_problem(TASKS, gold=_gold)})
