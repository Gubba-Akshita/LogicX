from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score


class SubgroupEvaluator:
    """
    Evaluates subgroup fairness and prediction calibration (ECE) across heterogeneous subsets.
    If subgroup metadata is not provided, returns a clean 'not_available' status without fabricating data.
    """

    @staticmethod
    def calculate_expected_calibration_error(
        probs: np.ndarray, y_true: np.ndarray, num_bins: int = 10
    ) -> float:
        """Calculates Expected Calibration Error (ECE) across binned probability confidences."""
        if len(y_true) == 0:
            return 0.0

        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = np.mean(accuracies[in_bin])
                avg_conf_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

        return float(ece)

    @classmethod
    def evaluate_subgroups(
        cls,
        model: Any,
        X_text: np.ndarray,
        X_img: np.ndarray,
        y: np.ndarray,
        subgroups: Optional[List[Any]] = None,
        num_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluates model across demographic/domain subgroups.
        Computes subgroup balanced accuracy, worst-group accuracy, disparity variance, and ECE.
        """
        if len(y) == 0:
            return {"subgroup_status": "not_available", "reason": "empty_evaluation_set"}

        probs = model.predict_proba(X_text, X_img)
        preds = np.argmax(probs, axis=1)
        overall_acc = float(balanced_accuracy_score(y, preds))
        ece = cls.calculate_expected_calibration_error(probs, y, num_bins=num_bins)

        if subgroups is None or len(subgroups) != len(y):
            return {
                "subgroup_status": "not_available",
                "overall_accuracy": overall_acc,
                "expected_calibration_error": ece,
            }

        unique_groups = sorted(list(set(subgroups)))
        if len(unique_groups) <= 1:
            return {
                "subgroup_status": "not_available",
                "reason": "insufficient_subgroups",
                "overall_accuracy": overall_acc,
                "expected_calibration_error": ece,
            }

        group_accuracies = {}
        for g in unique_groups:
            idx = [i for i, group_id in enumerate(subgroups) if group_id == g]
            if len(idx) > 0 and len(set(y[idx])) > 1:
                g_acc = float(balanced_accuracy_score(y[idx], preds[idx]))
                group_accuracies[str(g)] = g_acc

        if not group_accuracies:
            return {
                "subgroup_status": "not_available",
                "overall_accuracy": overall_acc,
                "expected_calibration_error": ece,
            }

        acc_values = list(group_accuracies.values())
        worst_group_acc = float(np.min(acc_values))
        mean_group_acc = float(np.mean(acc_values))
        subgroup_variance = float(np.var(acc_values))

        return {
            "subgroup_status": "evaluated",
            "group_accuracies": group_accuracies,
            "worst_group_accuracy": worst_group_acc,
            "mean_group_accuracy": mean_group_acc,
            "subgroup_variance": subgroup_variance,
            "subgroup_disparity": float(np.max(acc_values) - worst_group_acc),
            "expected_calibration_error": ece,
        }
