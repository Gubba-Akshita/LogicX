import numpy as np
from evops.model import CompactMultimodalModel
from evops.robustness import (
    evaluate_m2r_robustness,
    perturb_image_features,
    perturb_text_features,
)


def test_text_perturbation():
    X_text = np.ones((10, 16))
    X_pert = perturb_text_features(X_text, drop_rate=0.5, rng=np.random.RandomState(42))
    assert np.any(X_pert == 0.0), "Text perturbation failed to drop features."


def test_image_perturbation():
    X_img = np.ones((10, 16))
    X_pert = perturb_image_features(X_img, noise_scale=0.1, rng=np.random.RandomState(42))
    assert not np.array_equal(X_img, X_pert), "Image perturbation failed to add noise."


def test_m2r_robustness():
    m = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, random_state=42)
    X_t = np.random.randn(20, 16)
    X_i = np.random.randn(20, 16)
    y = np.random.choice([0, 1], size=20)

    res = evaluate_m2r_robustness(m, X_t, X_i, y, robustness_strength=0.05, random_state=42)
    assert "robustness_score" in res, "M2R evaluation missing robustness_score."
    assert 0.0 <= res["robustness_score"] <= 1.0, f"M2R score {res['robustness_score']} out of bounds."
