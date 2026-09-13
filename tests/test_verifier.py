"""`shares_eval_counts`: which verifiers the aggregator may reuse rates from.

Getting this wrong is silent in the dangerous direction. A verifier whose
expensive layer really is an independent measurement, read as sharing, simply
never gets called -- and the commit gate falls back to the cheap number it was
installed to cross-check, with no warning and no exception.
"""

# -- a declaration beats the base class's default -----------------------------

def test_a_subclass_can_declare_its_expensive_layer_independent():
    """`ThreeLayerVerifier` sets `full_eval_matches_counts = True` as a
    statement about how *that* class is written, and a subclass inherits it --
    so the escape hatch the attribute's docstring offers, "simply does not
    define it", is not available to a subclass. One that set the pre-0.6
    `oracle_shares_full_set = False` was ignored, its independent expensive
    layer went uncalled, and the gate degraded to the cheap measurement it
    existed to cross-check."""
    from agentdescent.verifier import ThreeLayerVerifier, shares_eval_counts

    class OldName(ThreeLayerVerifier):
        oracle_shares_full_set = False

    class NewName(ThreeLayerVerifier):
        full_eval_matches_counts = False

    class Inherits(ThreeLayerVerifier):
        pass

    assert not shares_eval_counts(OldName.__new__(OldName))
    assert not shares_eval_counts(NewName.__new__(NewName))
    assert shares_eval_counts(Inherits.__new__(Inherits)), (
        "a subclass that declares neither keeps the base class's statement")


def test_a_verifier_that_is_not_a_ThreeLayerVerifier_shares_nothing_by_default():
    from agentdescent.verifier import shares_eval_counts

    class Foreign:
        pass

    assert not shares_eval_counts(Foreign())
