from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def validate_input_data(X: Any, y: Any) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Validates input dataset for empty arrays, dimension mismatches, single-class labels,
    and converts inputs into sanitized DataFrames and arrays.
    """
    if X is None or y is None:
        raise ValueError("Input X and y cannot be None.")

    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X_df = pd.DataFrame(X)

    y_arr = np.array(y).ravel()

    if X_df.empty or y_arr.size == 0:
        raise ValueError("Empty dataset provided.")

    if len(X_df) != len(y_arr):
        raise ValueError(
            f"Length mismatch: X has {len(X_df)} samples, y has {len(y_arr)} samples."
        )

    unique_classes = np.unique(y_arr)
    if len(unique_classes) < 2:
        raise ValueError(
            f"Insufficient class diversity: Target y has only {len(unique_classes)} unique class."
        )

    return X_df, y_arr


def check_event_leakage(
    train_df: pd.DataFrame, ood_df: pd.DataFrame, ood_event_name: str
) -> bool:
    """
    Checks for data leakage between training events and the held-out OOD event split.
    Ensures that no sample from the OOD event exists in the training set.
    """
    train_events = set(train_df["event_name"].unique())
    if ood_event_name in train_events:
        raise ValueError(
            f"Data Leakage Detected! Held-out event '{ood_event_name}' found in training split."
        )

    # Check for sample ID overlaps if ID column present
    if "id" in train_df.columns and "id" in ood_df.columns:
        train_ids = set(train_df["id"])
        ood_ids = set(ood_df["id"])
        overlap = train_ids.intersection(ood_ids)
        if overlap:
            raise ValueError(
                f"Data Leakage Detected! {len(overlap)} overlapping sample IDs found."
            )

    return True
