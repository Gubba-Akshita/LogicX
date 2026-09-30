from typing import Any, Dict, List

import numpy as np


def compute_objective_vector(
    predictive_quality: float,
    ood_score: float,
    robustness_score: float,
    stability_score: float,
    latency: float,
    complexity: float,
    parameter_count: int,
    train_val_gap: float,
    event_variance: float,
    ood_degradation: float,
) -> Dict[str, Any]:
    """
    Constructs the complete 10-metric objective vector for a candidate solution.

    MAXIMIZATION TARGETS (Indices 0..3):
    0. predictive_quality (balanced accuracy on validation split)
    1. ood_score (EHOR balanced accuracy on held-out disaster event)
    2. robustness_score (M2R accuracy retention under text & image perturbations)
    3. stability_score (1.0 - train_val_gap)

    MINIMIZATION TARGETS (Indices 4..6 in Pareto vector):
    4. latency (inference latency per sample in seconds)
    5. complexity (structural architectural penalty)
    6. parameter_count (trainable weights count)

    SUPPLEMENTARY DIAGNOSTIC OBJECTIVES:
    - train_val_gap (in-distribution overfitting indicator)
    - event_variance (in-distribution variance across training disaster events)
    - ood_degradation (generalization penalty on held-out event)
    """
    obj_vector = [
        float(predictive_quality),
        float(ood_score),
        float(robustness_score),
        float(stability_score),
        float(latency),
        float(complexity),
        float(parameter_count),
    ]

    # Directions: True = Maximization, False = Minimization
    max_flags = [True, True, True, True, False, False, False]

    return {
        "predictive_quality": float(predictive_quality),
        "ood_score": float(ood_score),
        "robustness_score": float(robustness_score),
        "stability_score": float(stability_score),
        "latency": float(latency),
        "complexity": float(complexity),
        "parameter_count": int(parameter_count),
        "train_val_gap": float(train_val_gap),
        "event_variance": float(event_variance),
        "ood_degradation": float(ood_degradation),
        "obj_vector": obj_vector,
        "max_flags": max_flags,
    }
