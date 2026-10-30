"""The five-layer demo must run offline and produce every layer's decision."""

import pytest

from examples.rsi_loop.rsi_loop_demo import main


def test_five_layer_demo_runs_end_to_end_offline():
    result = main()
    loop = result["loop"]
    assert loop["status"] == "completed"
    assert loop["best_reward"] == pytest.approx(0.8)
    assert len(loop["generations"]) == 3
    assert loop["applied"] == [row["artifact_id"] for row in loop["generations"]
                                if row["accepted"]]
    assert result["promotion"]["action"] == "promote"
    assert result["promotion"]["accepted"]
    assert result["foundry"]["status"] == "promoted"
    assert result["foundry"]["registered"]
    experience = result["experience"]
    assert experience["rows"] == 3
    assert experience["replay"] == 3
    assert experience["accepted"]
