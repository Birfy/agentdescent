"""Near-duplicates must fold; distinct rules must not."""

import pytest

from agentdescent.semantic_dedup import (
    canonical_text,
    dedupe_texts,
    hamming,
    minhash_signature,
    minhash_similarity,
    simhash,
    simhash_similarity,
)


def test_canonical_text_is_case_and_whitespace_insensitive():
    assert canonical_text("  Use   THE tool.\n") == "use the tool."
    with pytest.raises(TypeError):
        canonical_text(3)  # type: ignore[arg-type]


def test_simhash_is_deterministic_and_similarity_is_symmetric():
    a = simhash("Cite the source before the claim.")
    b = simhash("Cite the source before the claim.")
    c = simhash("Use a calculator for arithmetic.")
    assert a == b
    assert hamming(a, b) == 0
    assert simhash_similarity(a, b) == 1.0
    assert simhash_similarity(a, c) == simhash_similarity(c, a)
    assert 0.0 <= simhash_similarity(a, c) <= 1.0


def test_simhash_rejects_impossible_widths():
    with pytest.raises(ValueError):
        simhash("x", bits=4)
    with pytest.raises(ValueError):
        simhash("x", bits=128)
    with pytest.raises(ValueError):
        simhash("x", bits=1024)
    with pytest.raises(ValueError):
        simhash_similarity(1, 2, bits=0)


def test_minhash_signature_is_deterministic_and_bounded():
    a = minhash_signature("one two three", num_perm=16)
    b = minhash_signature("one two three", num_perm=16)
    assert a == b and len(a) == 16
    assert 0.0 <= minhash_similarity(a, b) <= 1.0
    with pytest.raises(ValueError):
        minhash_signature("x", num_perm=0)
    with pytest.raises(ValueError):
        minhash_similarity(a, b[:-1])


def test_identical_texts_fold_to_one_representative():
    report = dedupe_texts(["same", "same", "same"], threshold=0.99)
    assert report.unique == 1
    assert report.removed == 2
    assert report.compression == pytest.approx(2 / 3)
    assert report.kept == ("same",)
    assert len(report.clusters[0].members) == 3


def test_near_duplicates_fold_but_distinct_rules_stay():
    a = "Always cite the source before making a claim about the paper."
    b = "Always cite the source before making a claim about this paper."
    c = "Use a calculator for arithmetic."
    for method in ("simhash", "minhash"):
        report = dedupe_texts([a, b, c], threshold=0.85, method=method)
        assert report.unique == 2
        assert report.removed == 1
        assert report.clusters[0].representative == a
        assert report.clusters[1].representative == c


def test_cjk_and_punctuation_only_texts_do_not_collapse():
    cjk = ["引用来源后再下结论。", "用计算器做算术。"]
    report = dedupe_texts(cjk, threshold=0.7)
    assert report.unique == 2
    assert report.removed == 0
    punct = dedupe_texts(["!!!", "???"], threshold=0.7)
    assert punct.unique == 2
    # Identical punctuation is still a duplicate.
    assert dedupe_texts(["!!!", "!!!"], threshold=0.7).unique == 1


def test_accented_latin_is_not_split_into_ascii_fragments():
    report = dedupe_texts(["résumé", "resume"], threshold=0.9)
    assert report.unique == 2


def test_threshold_is_inclusive_and_validated():
    a = "one two three four five"
    b = "one two three four six"
    score = minhash_similarity(minhash_signature(a), minhash_signature(b))
    report = dedupe_texts([a, b], threshold=score, method="minhash")
    assert report.unique == 1
    with pytest.raises(ValueError):
        dedupe_texts(["x"], threshold=-0.1)
    with pytest.raises(ValueError):
        dedupe_texts(["x"], threshold=1.1)
    with pytest.raises(ValueError):
        dedupe_texts(["x"], method="magic")
    with pytest.raises(ValueError):
        dedupe_texts([], method="magic")


def test_empty_input_and_member_cap():
    report = dedupe_texts([], threshold=0.5)
    assert report.unique == 0 and report.removed == 0
    report = dedupe_texts(["x"] * 5, threshold=0.99, max_members=2)
    assert len(report.clusters[0].members) == 2
    assert report.removed == 4
    with pytest.raises(ValueError):
        dedupe_texts(["x"], max_members=-1)


def test_custom_text_projection_is_used():
    items = [{"body": "Same rule"}, {"body": "Same rule"}, {"body": "Different"}]
    report = dedupe_texts(items, text=lambda row: row["body"], threshold=0.99)
    assert report.unique == 2
    assert report.clusters[0].representative["body"] == "Same rule"
