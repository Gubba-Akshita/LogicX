import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class CompactMultimodalModel:
    """
    Compact Multimodal Neural Model supporting:
    1. Text Dense Projection + Inverted Dropout
    2. Image Dense Projection + Inverted Dropout
    3. Multimodal Fusion (Concat / Gated / Weighted Late Fusion)
    4. Hidden Dense Layer + ReLU + Inverted Dropout
    5. Output Layer + Softmax Logits
    6. Complete Analytical Backpropagation across ALL trainable parameters
    7. Gradient Clipping, L2 Weight Decay, and Validation Early Stopping
    """

    def __init__(
        self,
        text_in_dim: int,
        img_in_dim: int,
        text_feature_dim: int = 64,
        image_feature_dim: int = 64,
        hidden_dim: int = 32,
        fusion_type: int = 0,  # 0: concat, 1: gated, 2: late
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        modality_weight: float = 0.5,
        early_stopping_patience: int = 5,
        num_classes: int = 2,
        random_state: int = 42,
    ):
        self.text_in_dim = int(text_in_dim)
        self.img_in_dim = int(img_in_dim)
        self.text_feature_dim = int(text_feature_dim)
        self.image_feature_dim = int(image_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.fusion_type = int(fusion_type)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.modality_weight = float(modality_weight)
        self.early_stopping_patience = int(early_stopping_patience)
        self.num_classes = int(num_classes)
        self.random_state = int(random_state)

        rng = np.random.RandomState(random_state)

        # Text projection
        scale_t = np.sqrt(2.0 / max(1, self.text_in_dim))
        self.W_text = rng.randn(self.text_in_dim, self.text_feature_dim).astype(np.float32) * scale_t
        self.b_text = np.zeros(self.text_feature_dim, dtype=np.float32)

        # Image projection
        scale_i = np.sqrt(2.0 / max(1, self.img_in_dim))
        self.W_img = rng.randn(self.img_in_dim, self.image_feature_dim).astype(np.float32) * scale_i
        self.b_img = np.zeros(self.image_feature_dim, dtype=np.float32)

        # Fusion configuration
        self.W_text_align: Optional[np.ndarray] = None
        self.W_img_align: Optional[np.ndarray] = None
        self.W_gate: Optional[np.ndarray] = None
        self.b_gate: Optional[np.ndarray] = None

        if self.fusion_type == 0:  # Concat
            fused_dim = self.text_feature_dim + self.image_feature_dim
        elif self.fusion_type == 1:  # Gated
            self.common_dim = min(self.text_feature_dim, self.image_feature_dim)
            self.W_text_align = rng.randn(self.text_feature_dim, self.common_dim).astype(np.float32) * 0.1
            self.W_img_align = rng.randn(self.image_feature_dim, self.common_dim).astype(np.float32) * 0.1
            self.W_gate = rng.randn(self.common_dim * 2, 1).astype(np.float32) * 0.1
            self.b_gate = np.zeros(1, dtype=np.float32)
            fused_dim = self.common_dim
        else:  # Weighted Late Fusion
            self.common_dim = min(self.text_feature_dim, self.image_feature_dim)
            self.W_text_align = rng.randn(self.text_feature_dim, self.common_dim).astype(np.float32) * 0.1
            self.W_img_align = rng.randn(self.image_feature_dim, self.common_dim).astype(np.float32) * 0.1
            fused_dim = self.common_dim

        # Hidden layer
        scale_h = np.sqrt(2.0 / max(1, fused_dim))
        self.W_hidden = rng.randn(fused_dim, self.hidden_dim).astype(np.float32) * scale_h
        self.b_hidden = np.zeros(self.hidden_dim, dtype=np.float32)

        # Output classification layer
        scale_out = np.sqrt(2.0 / max(1, self.hidden_dim))
        self.W_out = rng.randn(self.hidden_dim, self.num_classes).astype(np.float32) * scale_out
        self.b_out = np.zeros(self.num_classes, dtype=np.float32)

    def count_parameters(self) -> int:
        """Returns total trainable parameter count."""
        params = (
            self.W_text.size
            + self.b_text.size
            + self.W_img.size
            + self.b_img.size
            + self.W_hidden.size
            + self.b_hidden.size
            + self.W_out.size
            + self.b_out.size
        )
        if self.W_text_align is not None:
            params += self.W_text_align.size
        if self.W_img_align is not None:
            params += self.W_img_align.size
        if self.W_gate is not None:
            params += self.W_gate.size
        if self.b_gate is not None:
            params += self.b_gate.size
        return int(params)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15.0, 15.0)))

    def _apply_dropout(
        self, x: np.ndarray, is_training: bool, rng: np.random.RandomState
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not is_training or self.dropout <= 0.0:
            return x, None
        keep_prob = 1.0 - self.dropout
        mask = (rng.rand(*x.shape) < keep_prob).astype(np.float32) / keep_prob
        return x * mask, mask

    def forward(
        self,
        X_text: np.ndarray,
        X_img: np.ndarray,
        is_training: bool = False,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, Any]:
        """
        Computes forward pass through all multimodal layers, returning intermediate
        activations for exact gradient backpropagation.
        """
        r = rng if rng is not None else np.random.RandomState(self.random_state)

        # 1. Text Projection
        z_text = np.dot(X_text, self.W_text) + self.b_text
        h_text = self._relu(z_text)
        h_text_drop, mask_t = self._apply_dropout(h_text, is_training, r)

        # 2. Image Projection
        z_img = np.dot(X_img, self.W_img) + self.b_img
        h_img = self._relu(z_img)
        h_img_drop, mask_i = self._apply_dropout(h_img, is_training, r)

        # 3. Multimodal Fusion
        h_text_align = None
        h_img_align = None
        gate_val = None
        gate_in = None

        if self.fusion_type == 0:  # Concat
            fused = np.hstack([h_text_drop, h_img_drop])
        elif self.fusion_type == 1:  # Gated Fusion
            h_text_align = np.dot(h_text_drop, self.W_text_align)
            h_img_align = np.dot(h_img_drop, self.W_img_align)
            gate_in = np.hstack([h_text_align, h_img_align])
            gate_val = self._sigmoid(np.dot(gate_in, self.W_gate) + self.b_gate)
            fused = gate_val * h_text_align + (1.0 - gate_val) * h_img_align
        else:  # Weighted Late Fusion
            h_text_align = np.dot(h_text_drop, self.W_text_align)
            h_img_align = np.dot(h_img_drop, self.W_img_align)
            w = self.modality_weight
            fused = w * h_text_align + (1.0 - w) * h_img_align

        # 4. Hidden Layer
        z_hidden = np.dot(fused, self.W_hidden) + self.b_hidden
        h_hidden = self._relu(z_hidden)
        h_hidden_drop, mask_h = self._apply_dropout(h_hidden, is_training, r)

        # 5. Output Layer Logits
        logits = np.dot(h_hidden_drop, self.W_out) + self.b_out

        return {
            "logits": logits,
            "fused": fused,
            "z_text": z_text,
            "h_text": h_text,
            "h_text_drop": h_text_drop,
            "mask_t": mask_t,
            "z_img": z_img,
            "h_img": h_img,
            "h_img_drop": h_img_drop,
            "mask_i": mask_i,
            "h_text_align": h_text_align,
            "h_img_align": h_img_align,
            "gate_in": gate_in,
            "gate_val": gate_val,
            "z_hidden": z_hidden,
            "h_hidden": h_hidden,
            "h_hidden_drop": h_hidden_drop,
            "mask_h": mask_h,
        }

    def predict_proba(self, X_text: np.ndarray, X_img: np.ndarray) -> np.ndarray:
        """Computes numerically stable softmax probabilities."""
        cache = self.forward(X_text, X_img, is_training=False)
        logits = cache["logits"]
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(np.clip(shifted, -20.0, 20.0))
        return exp_l / np.sum(exp_l, axis=1, keepdims=True)

    def predict(self, X_text: np.ndarray, X_img: np.ndarray) -> np.ndarray:
        """Predicts discrete class labels."""
        probs = self.predict_proba(X_text, X_img)
        return np.argmax(probs, axis=1)

    def fit(
        self,
        X_text: np.ndarray,
        X_img: np.ndarray,
        y: np.ndarray,
        X_val_text: Optional[np.ndarray] = None,
        X_val_img: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 15,
        batch_size: int = 32,
    ) -> "CompactMultimodalModel":
        """
        Fits all trainable parameters using mini-batch SGD with analytical backpropagation,
        gradient clipping, weight decay, and validation-based early stopping.
        """
        n_samples = len(y)
        if n_samples == 0:
            return self

        actual_batch = min(batch_size, n_samples)
        rng = np.random.RandomState(self.random_state)

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        has_val = X_val_text is not None and X_val_img is not None and y_val is not None and len(y_val) > 0

        for epoch in range(epochs):
            indices = rng.permutation(n_samples)

            for i in range(0, n_samples, actual_batch):
                idx = indices[i : i + actual_batch]
                Xt_b = X_text[idx]
                Xi_b = X_img[idx]
                yb_b = y[idx]
                b_size = len(yb_b)

                # Forward pass with training activations cached
                cache = self.forward(Xt_b, Xi_b, is_training=True, rng=rng)
                logits = cache["logits"]

                # Softmax probabilities & cross-entropy gradient
                shifted = logits - np.max(logits, axis=1, keepdims=True)
                exp_l = np.exp(np.clip(shifted, -20.0, 20.0))
                probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

                dlogits = probs.copy()
                dlogits[np.arange(b_size), yb_b] -= 1.0
                dlogits /= float(b_size)

                # Output layer gradients
                h_h_drop = cache["h_hidden_drop"]
                dW_out = np.dot(h_h_drop.T, dlogits) + self.weight_decay * self.W_out
                db_out = np.sum(dlogits, axis=0)

                # Backprop to hidden layer
                dh_h_drop = np.dot(dlogits, self.W_out.T)
                if cache["mask_h"] is not None:
                    dh_h = dh_h_drop * cache["mask_h"]
                else:
                    dh_h = dh_h_drop

                dz_hidden = dh_h * (cache["z_hidden"] > 0).astype(np.float32)
                dW_hidden = np.dot(cache["fused"].T, dz_hidden) + self.weight_decay * self.W_hidden
                db_hidden = np.sum(dz_hidden, axis=0)

                # Backprop to fusion layer
                dfused = np.dot(dz_hidden, self.W_hidden.T)

                dW_text_align = None
                dW_img_align = None
                dW_gate = None
                db_gate = None

                if self.fusion_type == 0:  # Concat
                    dh_t_drop = dfused[:, : self.text_feature_dim]
                    dh_i_drop = dfused[:, self.text_feature_dim :]
                elif self.fusion_type == 1:  # Gated
                    g = cache["gate_val"]
                    h_ta = cache["h_text_align"]
                    h_ia = cache["h_img_align"]

                    dh_ta = dfused * g
                    dh_ia = dfused * (1.0 - g)
                    dg = np.sum(dfused * (h_ta - h_ia), axis=1, keepdims=True)

                    dgate_in_z = dg * g * (1.0 - g)
                    dW_gate = np.dot(cache["gate_in"].T, dgate_in_z) + self.weight_decay * self.W_gate
                    db_gate = np.sum(dgate_in_z, axis=0)

                    dgate_in = np.dot(dgate_in_z, self.W_gate.T)
                    dh_ta += dgate_in[:, : self.common_dim]
                    dh_ia += dgate_in[:, self.common_dim :]

                    dW_text_align = np.dot(cache["h_text_drop"].T, dh_ta) + self.weight_decay * self.W_text_align
                    dW_img_align = np.dot(cache["h_img_drop"].T, dh_ia) + self.weight_decay * self.W_img_align
                    dh_t_drop = np.dot(dh_ta, self.W_text_align.T)
                    dh_i_drop = np.dot(dh_ia, self.W_img_align.T)
                else:  # Weighted Late
                    w = self.modality_weight
                    dh_ta = dfused * w
                    dh_ia = dfused * (1.0 - w)

                    dW_text_align = np.dot(cache["h_text_drop"].T, dh_ta) + self.weight_decay * self.W_text_align
                    dW_img_align = np.dot(cache["h_img_drop"].T, dh_ia) + self.weight_decay * self.W_img_align
                    dh_t_drop = np.dot(dh_ta, self.W_text_align.T)
                    dh_i_drop = np.dot(dh_ia, self.W_img_align.T)

                # Backprop to Text Projection
                if cache["mask_t"] is not None:
                    dh_t = dh_t_drop * cache["mask_t"]
                else:
                    dh_t = dh_t_drop

                dz_t = dh_t * (cache["z_text"] > 0).astype(np.float32)
                dW_text = np.dot(Xt_b.T, dz_t) + self.weight_decay * self.W_text
                db_text = np.sum(dz_t, axis=0)

                # Backprop to Image Projection
                if cache["mask_i"] is not None:
                    dh_i = dh_i_drop * cache["mask_i"]
                else:
                    dh_i = dh_i_drop

                dz_i = dh_i * (cache["z_img"] > 0).astype(np.float32)
                dW_img = np.dot(Xi_b.T, dz_i) + self.weight_decay * self.W_img
                db_img = np.sum(dz_i, axis=0)

                # Gradient Clipping for numerical stability (clip by value [-5.0, 5.0])
                for grad in [dW_out, db_out, dW_hidden, db_hidden, dW_text, db_text, dW_img, db_img]:
                    np.clip(grad, -5.0, 5.0, out=grad)

                if dW_text_align is not None:
                    np.clip(dW_text_align, -5.0, 5.0, out=dW_text_align)
                if dW_img_align is not None:
                    np.clip(dW_img_align, -5.0, 5.0, out=dW_img_align)
                if dW_gate is not None:
                    np.clip(dW_gate, -5.0, 5.0, out=dW_gate)
                if db_gate is not None:
                    np.clip(db_gate, -5.0, 5.0, out=db_gate)

                # Parameter Updates
                self.W_out -= self.learning_rate * dW_out
                self.b_out -= self.learning_rate * db_out
                self.W_hidden -= self.learning_rate * dW_hidden
                self.b_hidden -= self.learning_rate * db_hidden
                self.W_text -= self.learning_rate * dW_text
                self.b_text -= self.learning_rate * db_text
                self.W_img -= self.learning_rate * dW_img
                self.b_img -= self.learning_rate * db_img

                if self.W_text_align is not None and dW_text_align is not None:
                    self.W_text_align -= self.learning_rate * dW_text_align
                if self.W_img_align is not None and dW_img_align is not None:
                    self.W_img_align -= self.learning_rate * dW_img_align
                if self.W_gate is not None and dW_gate is not None:
                    self.W_gate -= self.learning_rate * dW_gate
                if self.b_gate is not None and db_gate is not None:
                    self.b_gate -= self.learning_rate * db_gate

            # Validation tracking for Early Stopping
            if has_val:
                val_probs = self.predict_proba(X_val_text, X_val_img)
                eps = 1e-7
                val_loss = -float(np.mean(np.log(val_probs[np.arange(len(y_val)), y_val] + eps)))

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self._get_state()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if best_state is not None:
                            self._set_state(best_state)
                        break

        return self

    def _get_state(self) -> Dict[str, Any]:
        """Returns deep copy of all parameter weights."""
        return {
            "W_text": self.W_text.copy(),
            "b_text": self.b_text.copy(),
            "W_img": self.W_img.copy(),
            "b_img": self.b_img.copy(),
            "W_hidden": self.W_hidden.copy(),
            "b_hidden": self.b_hidden.copy(),
            "W_out": self.W_out.copy(),
            "b_out": self.b_out.copy(),
            "W_text_align": self.W_text_align.copy() if self.W_text_align is not None else None,
            "W_img_align": self.W_img_align.copy() if self.W_img_align is not None else None,
            "W_gate": self.W_gate.copy() if self.W_gate is not None else None,
            "b_gate": self.b_gate.copy() if self.b_gate is not None else None,
        }

    def _set_state(self, state: Dict[str, Any]) -> None:
        """Restores model parameter weights from snapshot."""
        self.W_text = state["W_text"].copy()
        self.b_text = state["b_text"].copy()
        self.W_img = state["W_img"].copy()
        self.b_img = state["b_img"].copy()
        self.W_hidden = state["W_hidden"].copy()
        self.b_hidden = state["b_hidden"].copy()
        self.W_out = state["W_out"].copy()
        self.b_out = state["b_out"].copy()
        if state["W_text_align"] is not None:
            self.W_text_align = state["W_text_align"].copy()
        if state["W_img_align"] is not None:
            self.W_img_align = state["W_img_align"].copy()
        if state["W_gate"] is not None:
            self.W_gate = state["W_gate"].copy()
        if state["b_gate"] is not None:
            self.b_gate = state["b_gate"].copy()
