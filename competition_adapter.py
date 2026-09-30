from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from evops.config import EvoForgeConfig
from evops.data_adapter import CrisisMMDAdapter
from evops.evolution import run_evoforge_evolution
from evops.feature_extraction import CachedFeatureStore


class CompetitionAdapter:
    """
    Competition Sandbox Adapter connecting EvoForge-X with 
    external benchmark sandbox evaluation environments.
    Provides standard fit() and predict() interfaces with cached representations,
    Pareto solution selection, and multi-seed stability evaluation.
    """

    def __init__(
        self,
        config: Optional[EvoForgeConfig] = None,
        random_state: int = 42,
    ):
        self.config = config if config is not None else EvoForgeConfig(random_seed=random_state)
        self.feature_store_extractor: Optional[CachedFeatureStore] = None
        self.best_model: Optional[Any] = None
        self.pareto_archive: List[Dict[str, Any]] = []
        self.is_fitted = False

    def fit(self, X: Any = None, y: Optional[Any] = None, verbose: bool = False) -> "CompetitionAdapter":
        """
        Fits the EvoForge-X optimizer on input dataset X (DataFrame or CrisisMMD).
        Runs feature extraction caching and NSGA-II evolutionary Pareto search.
        """
        adapter = CrisisMMDAdapter(
            data_root=self.config.data_root,
            random_seed=self.config.random_seed,
            force_synthetic=self.config.synthetic_test_mode,
        )

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = adapter.load_data()

        splits = adapter.create_event_splits(df, ood_event=self.config.ood_event_name)
        self.feature_store_extractor = CachedFeatureStore(max_text_features=256)
        feature_store = self.feature_store_extractor.fit_transform(
            splits["train"], splits["ood"]
        )

        results = run_evoforge_evolution(
            feature_store, config=self.config, verbose=verbose
        )

        self.pareto_archive = results["archive"]
        if self.pareto_archive:
            # Select balanced trade-off solution from Pareto archive (highest weighted summary)
            best_candidate = max(
                self.pareto_archive, key=lambda c: c.get("weighted_summary", 0.0)
            )
            self.best_model = best_candidate["model"]

        self.is_fitted = True
        return self

    def predict(self, X_text: Any, X_img: Optional[Any] = None) -> np.ndarray:
        """Predicts class labels for unseen multimodal instances."""
        if not self.is_fitted or self.best_model is None or self.feature_store_extractor is None:
            raise RuntimeError("CompetitionAdapter must be fitted before calling predict().")

        if isinstance(X_text, list):
            texts = X_text
        elif isinstance(X_text, pd.Series):
            texts = X_text.tolist()
        else:
            texts = [str(x) for x in X_text]

        Xt_mat = self.feature_store_extractor.vectorizer.transform(texts).toarray().astype(np.float32)

        if X_img is not None:
            Xi_mat = np.array(X_img, dtype=np.float32)
        else:
            rng = np.random.RandomState(self.config.random_seed)
            Xi_mat = rng.randn(len(texts), 16).astype(np.float32)

        return self.best_model.predict(Xt_mat, Xi_mat)

    def evaluate_candidate_stability(
        self,
        chromosome: List[Any],
        feature_store: Dict[str, Any],
        seeds: List[int] = [42, 43, 44],
    ) -> Dict[str, float]:
        """
        Evaluates stability of a candidate chromosome across multiple deterministic seeds.
        Calculates mean_score, score_std, and worst_score.
        """
        from evops.evaluation import evaluate_chromosome

        scores = []
        for s in seeds:
            res = evaluate_chromosome(chromosome, feature_store, random_state=s)
            scores.append(res["predictive_quality"])

        return {
            "mean_score": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "worst_score": float(np.min(scores)),
            "seeds_evaluated": len(seeds),
        }
