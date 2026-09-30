from evops.config import EvoForgeConfig
from evops.data_adapter import CrisisMMDAdapter
from evops.evolution import run_evoforge_evolution
from evops.feature_extraction import CachedFeatureStore


def test_deterministic_execution():
    cfg = EvoForgeConfig(random_seed=42, population_size=4, generations=2)

    adapter1 = CrisisMMDAdapter(random_seed=42)
    df1 = adapter1.load_data()
    splits1 = adapter1.create_event_splits(df1, ood_event="hurricane_irma")
    fs1 = CachedFeatureStore().fit_transform(splits1["train"], splits1["ood"])
    res1 = run_evoforge_evolution(fs1, config=cfg, verbose=False)

    adapter2 = CrisisMMDAdapter(random_seed=42)
    df2 = adapter2.load_data()
    splits2 = adapter2.create_event_splits(df2, ood_event="hurricane_irma")
    fs2 = CachedFeatureStore().fit_transform(splits2["train"], splits2["ood"])
    res2 = run_evoforge_evolution(fs2, config=cfg, verbose=False)

    arch1_chroms = [c["chromosome"] for c in res1["archive"]]
    arch2_chroms = [c["chromosome"] for c in res2["archive"]]

    arch1_set = set(tuple(c) for c in arch1_chroms)
    arch2_set = set(tuple(c) for c in arch2_chroms)

    assert arch1_set == arch2_set, f"Executions produced different candidate sets despite same seed: {arch1_set} vs {arch2_set}"
