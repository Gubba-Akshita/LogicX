import numpy as np

from evops.model import CompactMultimodalModel


def test_full_backpropagation_gradient_updates():
    """Verifies that all parameter matrices (projections, fusion, hidden, output) update during training."""
    rng = np.random.RandomState(42)
    X_t = rng.randn(40, 32).astype(np.float32)
    X_i = rng.randn(40, 16).astype(np.float32)
    y = rng.choice([0, 1], size=40)

    # Test Gated Fusion model
    model = CompactMultimodalModel(
        text_in_dim=32,
        img_in_dim=16,
        text_feature_dim=16,
        image_feature_dim=16,
        hidden_dim=16,
        fusion_type=1,  # gated
        dropout=0.1,
        learning_rate=0.01,
        weight_decay=0.0001,
        random_state=42,
    )

    init_W_text = model.W_text.copy()
    init_W_img = model.W_img.copy()
    init_W_hidden = model.W_hidden.copy()
    init_W_out = model.W_out.copy()
    init_W_gate = model.W_gate.copy()
    init_W_text_align = model.W_text_align.copy()

    model.fit(X_t, X_i, y, epochs=5, batch_size=20)

    assert not np.allclose(model.W_text, init_W_text), "W_text failed to update during backprop!"
    assert not np.allclose(model.W_img, init_W_img), "W_img failed to update during backprop!"
    assert not np.allclose(model.W_hidden, init_W_hidden), "W_hidden failed to update during backprop!"
    assert not np.allclose(model.W_out, init_W_out), "W_out failed to update during backprop!"
    assert not np.allclose(model.W_gate, init_W_gate), "W_gate failed to update during backprop!"
    assert not np.allclose(model.W_text_align, init_W_text_align), "W_text_align failed to update during backprop!"


def test_weighted_late_fusion_backpropagation():
    """Verifies backprop through weighted late fusion alignment matrices."""
    rng = np.random.RandomState(42)
    X_t = rng.randn(30, 20).astype(np.float32)
    X_i = rng.randn(30, 10).astype(np.float32)
    y = rng.choice([0, 1], size=30)

    model = CompactMultimodalModel(
        text_in_dim=20,
        img_in_dim=10,
        text_feature_dim=12,
        image_feature_dim=12,
        hidden_dim=8,
        fusion_type=2,  # weighted late
        modality_weight=0.75,
        learning_rate=0.01,
        random_state=42,
    )

    init_W_img_align = model.W_img_align.copy()
    model.fit(X_t, X_i, y, epochs=5, batch_size=15)

    assert not np.allclose(model.W_img_align, init_W_img_align), "W_img_align failed to update during late fusion backprop!"
