import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from evops.chromosome import chromosome_to_dict
from evops.fairness import SubgroupEvaluator
from evops.model import CompactMultimodalModel
from evops.objectives import compute_objective_vector
from evops.robustness import evaluate_m2r_robustness


def evaluate_chromosome(
    chromosome: List[Any],
    feature_store: Dict[str, Any],
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Fits and evaluates a chromosome against cached feature store splits, returning
    the complete 10-metric objective vector, subgroup fairness metrics, and candidate metadata.
    """
    params = chromosome_to_dict(chromosome)

    X_text_tr = feature_store["train"]["text"]
    X_img_tr = feature_store["train"]["image"]
    y_tr = feature_store["train"]["label"]
    events_tr = feature_store["train"]["events"]
    subgroups_tr = feature_store["train"].get("subgroups")

    X_text_ood = feature_store["ood"]["text"]
    X_img_ood = feature_store["ood"]["image"]
    y_ood = feature_store["ood"]["label"]
    subgroups_ood = feature_store["ood"].get("subgroups")

    # Stratified train/val split within training events
    try:
        (
            Xt_train,
            Xt_val,
            Xi_train,
            Xi_val,
            y_train,
            y_val,
        ) = train_test_split(
            X_text_tr,
            X_img_tr,
            y_tr,
            test_size=0.3,
            random_state=random_state,
            stratify=y_tr,
        )
    except ValueError:
        (
            Xt_train,
            Xt_val,
            Xi_train,
            Xi_val,
            y_train,
            y_val,
        ) = train_test_split(
            X_text_tr,
            X_img_tr,
            y_tr,
            test_size=0.3,
            random_state=random_state,
        )

    # Instantiate and fit multimodal neural model with complete analytical backprop
    model = CompactMultimodalModel(
        text_in_dim=Xt_train.shape[1],
        img_in_dim=Xi_train.shape[1],
        text_feature_dim=params["text_feature_dim"],
        image_feature_dim=params["image_feature_dim"],
        hidden_dim=params["hidden_dim"],
        fusion_type=params["fusion_type"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        modality_weight=params["modality_weight"],
        early_stopping_patience=params["early_stopping_patience"],
        random_state=random_state,
    )

    # Train with validation early stopping
    model.fit(
        Xt_train,
        Xi_train,
        y_train,
        X_val_text=Xt_val,
        X_val_img=Xi_val,
        y_val=y_val,
        epochs=15,
    )

    # Clean predictive quality (balanced accuracy on held-out validation)
    val_preds = model.predict(Xt_val, Xi_val)
    val_acc = float(balanced_accuracy_score(y_val, val_preds))

    train_preds = model.predict(Xt_train, Xi_train)
    train_acc = float(balanced_accuracy_score(y_train, train_preds))

    train_val_gap = abs(train_acc - val_acc)
    stability_score = max(0.0, min(1.0, 1.0 - train_val_gap))

    # EHOR: Out-of-Distribution score on held-out disaster event
    if len(y_ood) > 0:
        ood_preds = model.predict(X_text_ood, X_img_ood)
        ood_score = float(balanced_accuracy_score(y_ood, ood_preds))
    else:
        ood_score = val_acc

    ood_degradation = max(0.0, val_acc - ood_score)

    # In-Distribution Event Variance across training events
    unique_events = sorted(list(set(events_tr)))
    event_accs = []
    for ev in unique_events:
        idx = [i for i, e in enumerate(events_tr) if e == ev]
        if len(idx) > 0:
            ev_y = y_tr[idx]
            if len(set(ev_y)) > 1:
                ev_preds = model.predict(X_text_tr[idx], X_img_tr[idx])
                event_accs.append(
                    float(balanced_accuracy_score(ev_y, ev_preds))
                )
    event_variance = float(np.var(event_accs)) if len(event_accs) > 1 else 0.0

    # M2R Modality-Aware Robustness evaluation
    m2r_res = evaluate_m2r_robustness(
        model,
        Xt_val,
        Xi_val,
        y_val,
        robustness_strength=params["robustness_strength"],
        random_state=random_state,
    )
    robustness_score = m2r_res["robustness_score"]

    # Analytical computational latency estimate based on forward pass FLOPs / MACs
    macs = (
        Xt_train.shape[1] * params["text_feature_dim"]
        + Xi_train.shape[1] * params["image_feature_dim"]
        + (params["text_feature_dim"] + params["image_feature_dim"]) * params["hidden_dim"]
        + params["hidden_dim"] * 2
    )
    latency_sec = float(macs / 1e8)  # Deterministic architectural latency in seconds

    # Complexity penalty and parameter count
    param_count = model.count_parameters()
    comp_text = params["text_feature_dim"] / 256.0
    comp_img = params["image_feature_dim"] / 256.0
    comp_hidden = params["hidden_dim"] / 128.0
    complexity_penalty = float(
        0.4 * comp_text + 0.4 * comp_img + 0.2 * comp_hidden
    )

    # Construct complete 10-metric objective vector
    objs = compute_objective_vector(
        predictive_quality=val_acc,
        ood_score=ood_score,
        robustness_score=robustness_score,
        stability_score=stability_score,
        latency=latency_sec,
        complexity=complexity_penalty,
        parameter_count=param_count,
        train_val_gap=train_val_gap,
        event_variance=event_variance,
        ood_degradation=ood_degradation,
    )

    # Subgroup Fairness & Calibration Evaluation
    fairness_res = SubgroupEvaluator.evaluate_subgroups(
        model, Xt_val, Xi_val, y_val, subgroups=subgroups_tr
    )

    # Weighted summary score for readable progress logging (never replaces Pareto sorting)
    weighted_summary = (
        0.40 * val_acc
        + 0.30 * ood_score
        + 0.15 * robustness_score
        + 0.15 * stability_score
    )

    res = {
        "chromosome": list(chromosome),
        "params": params,
        "weighted_summary": float(weighted_summary),
        "model": model,
        "m2r_details": m2r_res,
        "fairness_details": fairness_res,
    }
    res.update(objs)
    return res
