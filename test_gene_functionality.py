import numpy as np

from evops.model import CompactMultimodalModel


def test_dropout_gene_changes_training_behavior():
    """Verifies that dropout parameter produces stochastic training activations."""
    rng = np.random.RandomState(42)
    X_t = rng.randn(20, 16).astype(np.float32)
    X_i = rng.randn(20, 16).astype(np.float32)

    model_no_drop = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, dropout=0.0, random_state=42)
    model_high_drop = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, dropout=0.3, random_state=42)

    cache1 = model_high_drop.forward(X_t, X_i, is_training=True, rng=np.random.RandomState(1))
    cache2 = model_high_drop.forward(X_t, X_i, is_training=True, rng=np.random.RandomState(2))

    # With training dropout, activations across different random masks must differ
    assert not np.allclose(cache1["h_hidden_drop"], cache2["h_hidden_drop"])


def test_fusion_type_changes_parameter_count():
    """Verifies that fusion_type gene genuinely changes architectural structure."""
    m_concat = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, fusion_type=0, random_state=42)
    m_gated = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, fusion_type=1, random_state=42)
    m_late = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, fusion_type=2, random_state=42)

    assert m_concat.count_parameters() != m_gated.count_parameters()
    assert m_gated.count_parameters() != m_late.count_parameters()


def test_learning_rate_affects_step_size():
    """Verifies that learning_rate directly governs parameter update magnitude."""
    rng = np.random.RandomState(42)
    X_t = rng.randn(20, 16).astype(np.float32)
    X_i = rng.randn(20, 16).astype(np.float32)
    y = rng.choice([0, 1], size=20)

    m_small = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, learning_rate=0.0001, random_state=42)
    m_large = CompactMultimodalModel(text_in_dim=16, img_in_dim=16, learning_rate=0.01, random_state=42)

    w_init_s = m_small.W_out.copy()
    w_init_l = m_large.W_out.copy()

    m_small.fit(X_t, X_i, y, epochs=1)
    m_large.fit(X_t, X_i, y, epochs=1)

    diff_s = np.linalg.norm(m_small.W_out - w_init_s)
    diff_l = np.linalg.norm(m_large.W_out - w_init_l)

    assert diff_l > diff_s, "Larger learning rate must produce larger parameter delta!"
