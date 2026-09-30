import numpy as np
import pandas as pd

from evops.data_adapter import (
    CrisisMMDAdapter,
    EventRotationEvaluator,
    GenericImageAdapter,
    GenericMultimodalAdapter,
    GenericTabularAdapter,
    GenericTextAdapter,
)


def test_crisis_mmd_synthetic_mode():
    """Verifies that force_synthetic generates deterministic multi-event dataset."""
    adapter = CrisisMMDAdapter(force_synthetic=True, random_seed=42)
    df = adapter.load_data()

    assert adapter.is_synthetic is True
    assert len(df) == 500
    assert len(df["event_name"].unique()) == 5
    assert "text" in df.columns
    assert "label" in df.columns
    assert "image_feat" in df.columns


def test_event_rotation_evaluator():
    """Verifies that EventRotationEvaluator rotates all available disaster events."""
    adapter = CrisisMMDAdapter(force_synthetic=True, random_seed=42)
    df = adapter.load_data()
    evaluator = EventRotationEvaluator(adapter)
    rotation_splits = evaluator.get_rotation_splits(df)

    assert len(rotation_splits) == 5
    ood_events = [s["ood_event_name"] for s in rotation_splits]
    assert len(set(ood_events)) == 5


def test_generic_adapters():
    """Verifies generic tabular, text, image, and multimodal adapter abstractions."""
    # Tabular
    df_tab = pd.DataFrame({"feat1": [1, 2, 3, 4], "feat2": [5, 6, 7, 8], "target": [0, 1, 0, 1]})
    tab_adapter = GenericTabularAdapter(df_tab, target_col="target")
    clean_tab = tab_adapter.load_data()
    assert "text" in clean_tab.columns
    assert "label" in clean_tab.columns

    # Text
    text_adapter = GenericTextAdapter(texts=["sample a", "sample b"], labels=[0, 1])
    clean_text = text_adapter.load_data()
    assert len(clean_text) == 2

    # Image
    img_adapter = GenericImageAdapter(image_features=np.random.randn(4, 16), labels=[0, 1, 0, 1])
    clean_img = img_adapter.load_data()
    assert len(clean_img) == 4
