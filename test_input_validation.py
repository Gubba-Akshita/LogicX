import math
import numpy as np
import pandas as pd

from evops.config import EvoForgeConfig
from evops.data_adapter import CrisisMMDAdapter
from evops.evolution import run_evoforge_evolution
from evops.feature_extraction import CachedFeatureStore
from evops.model import CompactMultimodalModel
from evops.objectives import compute_objective_vector
from evops.validation import validate_input_data


def test_population_size():
    cfg = EvoForgeConfig(population_size=6, generations=2)
    adapter = CrisisMMDAdapter(random_seed=42)
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event="hurricane_irma")
    fs = CachedFeatureStore().fit_transform(splits["train"], splits["ood"])
    res = run_evoforge_evolution(fs, config=cfg, verbose=False)
    assert res["history"][0]["population_size"] == 6


def test_elitism():
    cfg = EvoForgeConfig(population_size=4, generations=2, elite_count=1)
    adapter = CrisisMMDAdapter(random_seed=42)
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event="hurricane_irma")
    fs = CachedFeatureStore().fit_transform(splits["train"], splits["ood"])
    res = run_evoforge_evolution(fs, config=cfg, verbose=False)
    assert len(res["archive"]) >= 1, "Elitism failed to preserve top solution in Pareto archive."


def test_finite_objectives():
    objs = compute_objective_vector(0.8, 0.7, 0.9, 0.95, 0.001, 0.2, 1000, 0.1, 0.01, 0.1)
    for val in objs["obj_vector"]:
        assert math.isfinite(val), f"Objective value {val} is not finite."


def test_multimodal_fusion():
    for fusion_type in [0, 1, 2]:
        m = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, fusion_type=fusion_type, random_state=42)
        preds = m.predict(np.random.randn(5, 16), np.random.randn(5, 16))
        assert len(preds) == 5, f"Fusion type {fusion_type} failed forward prediction."


def test_missing_images():
    train_df = pd.DataFrame({"text": ["hello", "world"], "label": [0, 1], "event_name": ["e1", "e1"]})
    ood_df = pd.DataFrame({"text": ["test"], "label": [1], "event_name": ["e2"]})
    fs = CachedFeatureStore().fit_transform(train_df, ood_df)
    assert fs["train"]["image"].shape == (2, 16), "Missing image fallback failed."


def test_malformed_data():
    try:
        validate_input_data([], [])
        assert False, "Expected ValueError on empty data."
    except ValueError:
        pass


def test_class_imbalance():
    try:
        validate_input_data(np.ones((10, 2)), np.zeros(10))
        assert False, "Expected ValueError on single-class data."
    except ValueError:
        pass


def test_runtime_budget_handling():
    cfg = EvoForgeConfig(runtime_budget_sec=0.001, generations=50)
    adapter = CrisisMMDAdapter(random_seed=42)
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event="hurricane_irma")
    fs = CachedFeatureStore().fit_transform(splits["train"], splits["ood"])
    res = run_evoforge_evolution(fs, config=cfg, verbose=False)
    assert len(res["history"]) < 50, "Runtime budget failed to interrupt evolution."


def test_early_stopping():
    cfg = EvoForgeConfig(generations=2)
    adapter = CrisisMMDAdapter(random_seed=42)
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event="hurricane_irma")
    fs = CachedFeatureStore().fit_transform(splits["train"], splits["ood"])
    res = run_evoforge_evolution(fs, config=cfg, verbose=False)
    assert len(res["history"]) == 2
