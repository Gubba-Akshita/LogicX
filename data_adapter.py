import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaseDatasetAdapter(ABC):
    """Abstract base class for all standardized dataset adapters."""

    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Loads and returns standardized DataFrame with features, labels, and event IDs."""
        pass

    @abstractmethod
    def create_event_splits(
        self, df: pd.DataFrame, ood_event: str
    ) -> Dict[str, Any]:
        """Creates train and held-out OOD evaluation splits."""
        pass


class CrisisMMDAdapter(BaseDatasetAdapter):
    """
    Data ingestion adapter for the CrisisMMD development benchmark.
    Supports strict real-data mode and deterministic synthetic test mode.
    """

    def __init__(
        self,
        data_root: str = "data/crisismmd",
        random_seed: int = 42,
        force_synthetic: bool = False,
    ):
        self.data_root = data_root
        self.random_seed = random_seed
        self.force_synthetic = force_synthetic
        self.is_synthetic = False

    def load_data(self) -> pd.DataFrame:
        """
        Loads local CrisisMMD dataset if available and force_synthetic is False.
        Otherwise, generates a deterministic synthetic multimodal dataset.
        """
        csv_path = os.path.join(self.data_root, "crisismmd_datasplit.csv")

        if not self.force_synthetic and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            required = ["text", "event_name", "label"]
            if all(col in df.columns for col in required):
                self.is_synthetic = False
                return df

        # Deterministic synthetic test benchmark
        self.is_synthetic = True
        rng = np.random.RandomState(self.random_seed)
        events = [
            "california_wildfires",
            "hurricane_irma",
            "hurricane_maria",
            "mexico_earthquake",
            "iraq_floods",
        ]

        data = []
        words_informative = [
            "help",
            "rescue",
            "shelter",
            "supplies",
            "food",
            "urgent",
            "evacuate",
            "trapped",
        ]
        words_not_informative = [
            "watching",
            "news",
            "sad",
            "thoughts",
            "prayers",
            "rt",
            "photo",
            "today",
        ]

        for event in events:
            for i in range(100):
                label = int(rng.choice([0, 1]))
                if label == 1:
                    w = rng.choice(words_informative, size=4)
                    text = f"Emergency in {event}: {w[0]} {w[1]} needed at {w[2]} {w[3]}"
                else:
                    w = rng.choice(words_not_informative, size=4)
                    text = f"Discussion about {event}: {w[0]} {w[1]} {w[2]} {w[3]}"

                # Synthetic 16-dim visual descriptor with modality-correlated signal
                img_feat = (
                    rng.randn(16) + (2.0 if label == 1 else -1.0)
                ).tolist()

                data.append(
                    {
                        "id": f"{event}_{i}",
                        "text": text,
                        "event_name": event,
                        "label": label,
                        "image_feat": img_feat,
                    }
                )

        return pd.DataFrame(data)

    def create_event_splits(
        self, df: pd.DataFrame, ood_event: str = "hurricane_irma"
    ) -> Dict[str, Any]:
        """
        Creates event-aware training and held-out OOD splits (EHOR protocol).
        Event identifiers are strictly isolated from predictive features.
        """
        unique_events = list(df["event_name"].unique())
        if ood_event not in unique_events:
            ood_event = str(unique_events[-1])

        train_df = (
            df[df["event_name"] != ood_event].copy().reset_index(drop=True)
        )
        ood_df = (
            df[df["event_name"] == ood_event].copy().reset_index(drop=True)
        )

        return {
            "train": train_df,
            "ood": ood_df,
            "ood_event_name": ood_event,
            "train_event_names": list(train_df["event_name"].unique()),
            "is_synthetic": self.is_synthetic,
        }


class GenericMultimodalAdapter(BaseDatasetAdapter):
    """Adapter for generic multimodal tabular/text/image DataFrames."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        image_col: str = "image_feat",
        label_col: str = "label",
        event_col: Optional[str] = "event_name",
        subgroup_col: Optional[str] = None,
    ):
        self.df = df.copy()
        self.text_col = text_col
        self.image_col = image_col
        self.label_col = label_col
        self.event_col = event_col
        self.subgroup_col = subgroup_col

    def load_data(self) -> pd.DataFrame:
        clean_df = pd.DataFrame()
        clean_df["text"] = self.df[self.text_col].fillna("").astype(str)
        clean_df["label"] = self.df[self.label_col].to_numpy()

        if self.image_col in self.df.columns:
            clean_df["image_feat"] = self.df[self.image_col]
        else:
            clean_df["image_feat"] = [np.zeros(16).tolist() for _ in range(len(self.df))]

        if self.event_col and self.event_col in self.df.columns:
            clean_df["event_name"] = self.df[self.event_col].astype(str)
        else:
            clean_df["event_name"] = "default_event"

        if self.subgroup_col and self.subgroup_col in self.df.columns:
            clean_df["subgroup"] = self.df[self.subgroup_col].astype(str)

        return clean_df

    def create_event_splits(
        self, df: pd.DataFrame, ood_event: str = "default_event"
    ) -> Dict[str, Any]:
        unique_events = list(df["event_name"].unique())
        if ood_event not in unique_events or len(unique_events) <= 1:
            # Fallback to stratified 80/20 train/ood split
            from sklearn.model_selection import train_test_split

            train_df, ood_df = train_test_split(
                df, test_size=0.2, random_state=42, stratify=df["label"]
            )
            return {
                "train": train_df.reset_index(drop=True),
                "ood": ood_df.reset_index(drop=True),
                "ood_event_name": "held_out_random_split",
                "train_event_names": list(train_df["event_name"].unique()),
                "is_synthetic": False,
            }

        train_df = df[df["event_name"] != ood_event].copy().reset_index(drop=True)
        ood_df = df[df["event_name"] == ood_event].copy().reset_index(drop=True)

        return {
            "train": train_df,
            "ood": ood_df,
            "ood_event_name": ood_event,
            "train_event_names": list(train_df["event_name"].unique()),
            "is_synthetic": False,
        }


class GenericTabularAdapter(BaseDatasetAdapter):
    """Adapter for pure tabular datasets converting rows into text/feature representations."""

    def __init__(self, df: pd.DataFrame, target_col: str, event_col: Optional[str] = None):
        self.df = df
        self.target_col = target_col
        self.event_col = event_col

    def load_data(self) -> pd.DataFrame:
        clean = pd.DataFrame()
        feature_cols = [c for c in self.df.columns if c not in [self.target_col, self.event_col]]
        clean["text"] = self.df[feature_cols].apply(
            lambda row: " ".join(f"{c}_{val}" for c, val in row.items()), axis=1
        )
        clean["label"] = self.df[self.target_col].to_numpy()
        clean["event_name"] = self.df[self.event_col] if self.event_col else "default_event"
        clean["image_feat"] = [np.zeros(16).tolist() for _ in range(len(self.df))]
        return clean

    def create_event_splits(self, df: pd.DataFrame, ood_event: str = "default_event") -> Dict[str, Any]:
        adapter = GenericMultimodalAdapter(df)
        return adapter.create_event_splits(df, ood_event=ood_event)


class GenericTextAdapter(BaseDatasetAdapter):
    """Adapter for text classification datasets."""

    def __init__(self, texts: List[str], labels: List[int], event_names: Optional[List[str]] = None):
        self.df = pd.DataFrame({
            "text": texts,
            "label": labels,
            "event_name": event_names if event_names else ["default_event"] * len(texts),
            "image_feat": [np.zeros(16).tolist() for _ in range(len(texts))],
        })

    def load_data(self) -> pd.DataFrame:
        return self.df

    def create_event_splits(self, df: pd.DataFrame, ood_event: str = "default_event") -> Dict[str, Any]:
        adapter = GenericMultimodalAdapter(df)
        return adapter.create_event_splits(df, ood_event=ood_event)


class GenericImageAdapter(BaseDatasetAdapter):
    """Adapter for image descriptor classification datasets."""

    def __init__(self, image_features: np.ndarray, labels: List[int], event_names: Optional[List[str]] = None):
        self.df = pd.DataFrame({
            "text": ["image_sample"] * len(labels),
            "label": labels,
            "event_name": event_names if event_names else ["default_event"] * len(labels),
            "image_feat": image_features.tolist(),
        })

    def load_data(self) -> pd.DataFrame:
        return self.df

    def create_event_splits(self, df: pd.DataFrame, ood_event: str = "default_event") -> Dict[str, Any]:
        adapter = GenericMultimodalAdapter(df)
        return adapter.create_event_splits(df, ood_event=ood_event)


class EventRotationEvaluator:
    """
    Executes cross-event rotation protocol (EHOR Rotation):
    Iterates through each distinct disaster event as held-out OOD,
    evaluating generalization stability across all shifts.
    """

    def __init__(self, adapter: BaseDatasetAdapter):
        self.adapter = adapter

    def get_rotation_splits(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        unique_events = list(df["event_name"].unique())
        if len(unique_events) <= 1:
            return [self.adapter.create_event_splits(df, ood_event=unique_events[0])]

        splits = []
        for event in unique_events:
            split = self.adapter.create_event_splits(df, ood_event=event)
            splits.append(split)
        return splits
