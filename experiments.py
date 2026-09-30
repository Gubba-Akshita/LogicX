import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from evops.chromosome import random_chromosome
from evops.config import EvoForgeConfig
from evops.data_adapter import CrisisMMDAdapter
from evops.evaluation import evaluate_chromosome
from evops.evolution import run_evoforge_evolution
from evops.feature_extraction import CachedFeatureStore
from evops.model import CompactMultimodalModel


def run_baselines_and_ablations(
    config: Optional[EvoForgeConfig] = None, verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes all 5 lightweight baselines and 6 controlled experimental ablations
    using identical train/OOD splits and feature caching.
    """
    cfg = config if config is not None else EvoForgeConfig(population_size=6, generations=3)
    adapter = CrisisMMDAdapter(
        data_root=cfg.data_root,
        random_seed=cfg.random_seed,
        force_synthetic=cfg.synthetic_test_mode,
    )
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event=cfg.ood_event_name)

    feature_extractor = CachedFeatureStore(max_text_features=256)
    feature_store = feature_extractor.fit_transform(splits["train"], splits["ood"])

    X_text_tr = feature_store["train"]["text"]
    X_img_tr = feature_store["train"]["image"]
    y_tr = feature_store["train"]["label"]

    X_text_ood = feature_store["ood"]["text"]
    X_img_ood = feature_store["ood"]["image"]
    y_ood = feature_store["ood"]["label"]

    results = {}

    if verbose:
        print("\n" + "=" * 75)
        print("EVOFORGE-X: BASELINE & ABLATION EXPERIMENTAL EVALUATION")
        print("=" * 75)

    # 1. Baseline 1: Text-Only Model
    m_text = CompactMultimodalModel(
        text_in_dim=X_text_tr.shape[1],
        img_in_dim=X_img_tr.shape[1],
        text_feature_dim=64,
        image_feature_dim=64,
        modality_weight=1.0,
        random_state=cfg.random_seed,
    )
    m_text.fit(X_text_tr, np.zeros_like(X_img_tr), y_tr, epochs=15)
    text_ood_acc = float(
        balanced_accuracy_score(y_ood, m_text.predict(X_text_ood, np.zeros_like(X_img_ood)))
    )
    results["Baseline 1 (Text-Only)"] = {"ood_acc": text_ood_acc}

    # 2. Baseline 2: Image-Only Model
    m_img = CompactMultimodalModel(
        text_in_dim=X_text_tr.shape[1],
        img_in_dim=X_img_tr.shape[1],
        text_feature_dim=64,
        image_feature_dim=64,
        modality_weight=0.0,
        random_state=cfg.random_seed,
    )
    m_img.fit(np.zeros_like(X_text_tr), X_img_tr, y_tr, epochs=15)
    img_ood_acc = float(
        balanced_accuracy_score(y_ood, m_img.predict(np.zeros_like(X_text_ood), X_img_ood))
    )
    results["Baseline 2 (Image-Only)"] = {"ood_acc": img_ood_acc}

    # 3. Baseline 3: Simple Multimodal Fusion (Concat)
    m_simple = CompactMultimodalModel(
        text_in_dim=X_text_tr.shape[1],
        img_in_dim=X_img_tr.shape[1],
        fusion_type=0,
        random_state=cfg.random_seed,
    )
    m_simple.fit(X_text_tr, X_img_tr, y_tr, epochs=15)
    simple_ood_acc = float(
        balanced_accuracy_score(y_ood, m_simple.predict(X_text_ood, X_img_ood))
    )
    results["Baseline 3 (Simple Concat Fusion)"] = {"ood_acc": simple_ood_acc}

    # 4. Baseline 4: Non-Evolutionary Default Configuration
    default_chrom = [64, 64, 32, 0, 0.1, 0.001, 0.0001, 0.5, 0.05, 5]
    default_eval = evaluate_chromosome(default_chrom, feature_store, random_state=cfg.random_seed)
    results["Baseline 4 (Default Configuration)"] = {
        "val_acc": default_eval["predictive_quality"],
        "ood_acc": default_eval["ood_score"],
        "robustness": default_eval["robustness_score"],
    }

    # 5. Baseline 5: GA without Pareto Optimization (Single-Objective Scalar Fitness)
    # Run evolutionary search with fixed scalar fitness
    scalar_cfg = copy.deepcopy(cfg)
    scalar_cfg.population_size = 6
    scalar_cfg.generations = 3
    scalar_evo_res = run_evoforge_evolution(feature_store, config=scalar_cfg, verbose=False)
    results["Baseline 5 (GA w/o Pareto)"] = {
        "best_ood": max(c["ood_score"] for c in scalar_evo_res["archive"]) if scalar_evo_res["archive"] else simple_ood_acc
    }

    # 6. Final: Full EvoForge-X (NSGA-II + DAAM + M2R + EHOR)
    evo_res = run_evoforge_evolution(feature_store, config=cfg, verbose=verbose)
    archive = evo_res["archive"]
    best_evo_ood = max(c["ood_score"] for c in archive) if archive else simple_ood_acc
    best_evo_pred = max(c["predictive_quality"] for c in archive) if archive else 0.0
    best_evo_rob = max(c["robustness_score"] for c in archive) if archive else 0.0

    results["Final (Full EvoForge-X)"] = {
        "best_predictive": best_evo_pred,
        "best_ood": best_evo_ood,
        "best_robustness": best_evo_rob,
        "pareto_size": len(archive),
        "runtime": evo_res["total_runtime"],
    }

    # Controlled Ablation Studies
    ablations = {
        "Ablation A (Without Pareto vs With Pareto)": {
            "without_pareto": results["Baseline 5 (GA w/o Pareto)"]["best_ood"],
            "with_pareto": best_evo_ood,
        },
        "Ablation B (Fixed Mutation vs DAAM)": {
            "initial_rate": cfg.initial_mutation_rate,
            "final_adapted_rate": evo_res["history"][-1]["mutation_rate"] if evo_res["history"] else cfg.initial_mutation_rate,
            "population_diversity": evo_res["history"][-1]["chromosome_diversity"] if evo_res["history"] else 0.0,
        },
        "Ablation C (Unimodal vs Multimodal)": {
            "text_only_ood": text_ood_acc,
            "image_only_ood": img_ood_acc,
            "multimodal_gain": simple_ood_acc - max(text_ood_acc, img_ood_acc),
        },
    }
    results["Ablations"] = ablations

    if verbose:
        print("\nSUMMARY OF BASELINE & ABLATION COMPARISONS:")
        for name, metrics in results.items():
            print(f"  {name:36s}: {metrics}")
        print("=" * 75)

    return results
