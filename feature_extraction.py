import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class CachedFeatureStore:
    """
    Pre-extracts and caches text and image feature matrices prior to evolutionary optimization,
    eliminating redundant feature extraction compute during chromosome evaluations.
    Strictly fits transformations on training data only to prevent data leakage.
    """

    def __init__(self, max_text_features: int = 256):
        self.max_text_features = max_text_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_text_features, stop_words="english"
        )
        self.is_fitted = False
        self.extraction_time_sec = 0.0

    def fit_transform(
        self, train_df: pd.DataFrame, ood_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Extracts and caches text & image feature matrices for train and OOD splits."""
        t0 = time.perf_counter()

        train_texts = train_df["text"].fillna("").astype(str).tolist()
        ood_texts = ood_df["text"].fillna("").astype(str).tolist()

        # Fit TF-IDF ONLY on training split
        X_text_tr = (
            self.vectorizer.fit_transform(train_texts)
            .toarray()
            .astype(np.float32)
        )
        # Transform OOD using fitted train vocabulary
        X_text_ood = (
            self.vectorizer.transform(ood_texts).toarray().astype(np.float32)
        )

        if "image_feat" in train_df.columns:
            X_img_tr = np.array(
                train_df["image_feat"].tolist(), dtype=np.float32
            )
            X_img_ood = np.array(
                ood_df["image_feat"].tolist(), dtype=np.float32
            )
        else:
            rng = np.random.RandomState(42)
            X_img_tr = rng.randn(len(train_df), 16).astype(np.float32)
            X_img_ood = rng.randn(len(ood_df), 16).astype(np.float32)

        y_tr = train_df["label"].to_numpy(dtype=np.int64)
        y_ood = ood_df["label"].to_numpy(dtype=np.int64)
        events_tr = train_df["event_name"].tolist()

        subgroups_tr = (
            train_df["subgroup"].tolist() if "subgroup" in train_df.columns else None
        )
        subgroups_ood = (
            ood_df["subgroup"].tolist() if "subgroup" in ood_df.columns else None
        )

        self.is_fitted = True
        self.extraction_time_sec = time.perf_counter() - t0

        return {
            "train": {
                "text": X_text_tr,
                "image": X_img_tr,
                "label": y_tr,
                "events": events_tr,
                "subgroups": subgroups_tr,
            },
            "ood": {
                "text": X_text_ood,
                "image": X_img_ood,
                "label": y_ood,
                "event_name": (
                    ood_df["event_name"].iloc[0]
                    if len(ood_df) > 0
                    else "unknown"
                ),
                "subgroups": subgroups_ood,
            },
            "extraction_time_sec": self.extraction_time_sec,
        }
