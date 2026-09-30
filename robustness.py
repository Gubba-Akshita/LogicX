from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def perturb_text_features(
    X_text: np.ndarray,
    drop_rate: float = 0.10,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Applies feature/token dropout perturbation to text representation."""
    if drop_rate <= 0.0:
        return X_text.copy()
    r = rng if rng is not None else np.random.RandomState(42)
    mask = r.binomial(1, 1.0 - drop_rate, size=X_text.shape)
    return X_text * mask


def perturb_image_features(
    X_img: np.ndarray,
    noise_scale: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Applies Gaussian pixel/descriptor perturbation to image representation."""
    if noise_scale <= 0.0:
        return X_img.copy()
    r = rng if rng is not None else np.random.RandomState(42)
    noise = r.normal(0, noise_scale, size=X_img.shape)
    return X_img + noise


def evaluate_m2r_robustness(
    model: Any,
    X_text: np.ndarray,
    X_img: np.ndarray,
    y: np.ndarray,
    robustness_strength: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Modality-Aware Robustness (M2R) Evaluation:
    Evaluates clean accuracy vs text-perturbed, image-perturbed, and joint-perturbed accuracy.
    Computes per-modality degradations and the composite robustness score.
    """
    if len(y) == 0:
        return {
            "clean_acc": 0.0,
            "text_acc": 0.0,
            "img_acc": 0.0,
            "joint_acc": 0.0,
            "text_degradation": 0.0,
            "img_degradation": 0.0,
            "joint_degradation": 0.0,
            "robustness_score": 0.0,
            "degradation": 0.0,
        }

    clean_preds = model.predict(X_text, X_img)
    clean_acc = float(balanced_accuracy_score(y, clean_preds))

    rng = np.random.RandomState(random_state)
    Xt_pert = perturb_text_features(
        X_text, drop_rate=robustness_strength, rng=rng
    )
    Xi_pert = perturb_image_features(
        X_img, noise_scale=robustness_strength, rng=rng
    )

    text_preds = model.predict(Xt_pert, X_img)
    text_acc = float(balanced_accuracy_score(y, text_preds))

    img_preds = model.predict(X_text, Xi_pert)
    img_acc = float(balanced_accuracy_score(y, img_preds))

    joint_preds = model.predict(Xt_pert, Xi_pert)
    joint_acc = float(balanced_accuracy_score(y, joint_preds))

    text_drop = max(0.0, clean_acc - text_acc)
    img_drop = max(0.0, clean_acc - img_acc)
    joint_drop = max(0.0, clean_acc - joint_acc)
    robustness_score = max(0.0, min(1.0, 1.0 - joint_drop))

    return {
        "clean_acc": clean_acc,
        "text_acc": text_acc,
        "img_acc": img_acc,
        "joint_acc": joint_acc,
        "text_degradation": float(text_drop),
        "img_degradation": float(img_drop),
        "joint_degradation": float(joint_drop),
        "robustness_score": float(robustness_score),
        "degradation": float(joint_drop),
    }
