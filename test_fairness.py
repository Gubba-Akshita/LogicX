import numpy as np

from evops.fairness import SubgroupEvaluator
from evops.model import CompactMultimodalModel


def test_subgroup_evaluator_with_groups():
    """Verifies subgroup metrics calculation when subgroup metadata is provided."""
    rng = np.random.RandomState(42)
    X_t = rng.randn(40, 16).astype(np.float32)
    X_i = rng.randn(40, 16).astype(np.float32)
    y = rng.choice([0, 1], size=40)
    subgroups = ["group_A"] * 20 + ["group_B"] * 20

    model = CompactMultimodalModel(
        text_in_dim=16,
        img_in_dim=16,
        random_state=42,
    )
    model.fit(X_t, X_i, y, epochs=2)

    res = SubgroupEvaluator.evaluate_subgroups(
        model, X_t, X_i, y, subgroups=subgroups
    )

    assert res["subgroup_status"] == "evaluated"
    assert "group_A" in res["group_accuracies"]
    assert "group_B" in res["group_accuracies"]
    assert 0.0 <= res["worst_group_accuracy"] <= 1.0
    assert res["expected_calibration_error"] >= 0.0


def test_subgroup_evaluator_fallback():
    """Verifies that absence of subgroup metadata returns a clean 'not_available' response."""
    rng = np.random.RandomState(42)
    X_t = rng.randn(20, 16).astype(np.float32)
    X_i = rng.randn(20, 16).astype(np.float32)
    y = rng.choice([0, 1], size=20)

    model = CompactMultimodalModel(
        text_in_dim=16,
        img_in_dim=16,
        random_state=42,
    )

    res = SubgroupEvaluator.evaluate_subgroups(
        model, X_t, X_i, y, subgroups=None
    )

    assert res["subgroup_status"] == "not_available"
    assert "expected_calibration_error" in res
