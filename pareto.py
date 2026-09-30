from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def evaluate_pareto_dominance(
    obj1: List[float], obj2: List[float], max_flags: List[bool]
) -> bool:
    """
    Returns True if obj1 Pareto-dominates obj2.
    obj1 dominates obj2 if obj1 is at least as good in all objectives
    and strictly better in at least one objective.
    """
    better_or_equal = True
    strictly_better = False

    for v1, v2, is_max in zip(obj1, obj2, max_flags):
        if is_max:
            if v1 < v2:
                better_or_equal = False
            elif v1 > v2:
                strictly_better = True
        else:
            if v1 > v2:
                better_or_equal = False
            elif v1 < v2:
                strictly_better = True

    return better_or_equal and strictly_better


def fast_non_dominated_sort(
    objectives_list: List[List[float]], max_flags: List[bool]
) -> List[List[int]]:
    """Performs NSGA-II fast non-dominated sorting. Returns fronts of indices."""
    num_pop = len(objectives_list)
    if num_pop == 0:
        return []

    S = [[] for _ in range(num_pop)]
    n = [0] * num_pop
    rank = [0] * num_pop
    fronts = [[]]

    for p in range(num_pop):
        for q in range(num_pop):
            if p == q:
                continue
            if evaluate_pareto_dominance(
                objectives_list[p], objectives_list[q], max_flags
            ):
                S[p].append(q)
            elif evaluate_pareto_dominance(
                objectives_list[q], objectives_list[p], max_flags
            ):
                n[p] += 1

        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if len(f) > 0]


def calculate_crowding_distance(
    front_indices: List[int],
    objectives_list: List[List[float]],
    max_flags: List[bool],
) -> Dict[int, float]:
    """
    Calculates NSGA-II crowding distance for a single Pareto front of candidate indices.
    Boundary solutions on each objective receive infinite distance (1e9) to preserve diversity.
    """
    if len(front_indices) == 0:
        return {}
    if len(front_indices) <= 2:
        return {idx: float("inf") for idx in front_indices}

    distances = {idx: 0.0 for idx in front_indices}
    num_objectives = len(objectives_list[0])

    for m in range(num_objectives):
        # Sort front indices by m-th objective value with secondary key for deterministic tie-breaking
        sorted_indices = sorted(
            front_indices, key=lambda idx: (objectives_list[idx][m], idx)
        )

        # Boundary solutions get infinity
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        obj_min = objectives_list[sorted_indices[0]][m]
        obj_max = objectives_list[sorted_indices[-1]][m]
        obj_range = abs(obj_max - obj_min)

        if obj_range > 1e-8:
            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i - 1]
                next_idx = sorted_indices[i + 1]
                dist = abs(objectives_list[next_idx][m] - objectives_list[prev_idx][m]) / obj_range
                if distances[idx] != float("inf"):
                    distances[idx] += dist

    return distances


def nsga2_environmental_selection(
    combined_population: List[List[Any]],
    combined_evaluations: List[Dict[str, Any]],
    target_pop_size: int,
    max_flags: Optional[List[bool]] = None,
) -> Tuple[List[List[Any]], List[Dict[str, Any]]]:
    """
    Performs true NSGA-II environmental selection (P_t U Q_t -> P_{t+1}).
    Fills the next generation front-by-front, using crowding distance truncation
    on the splitting boundary front to strictly preserve population size and diversity.
    """
    if len(combined_population) <= target_pop_size:
        return combined_population, combined_evaluations

    if max_flags is None:
        max_flags = combined_evaluations[0]["max_flags"]

    objs_list = [e["obj_vector"] for e in combined_evaluations]
    fronts = fast_non_dominated_sort(objs_list, max_flags)

    next_population = []
    next_evaluations = []

    for front in fronts:
        if len(next_population) + len(front) <= target_pop_size:
            # Front fits entirely
            for idx in front:
                next_population.append(combined_population[idx])
                next_evaluations.append(combined_evaluations[idx])
        else:
            # Front must be truncated using crowding distance
            remaining_needed = target_pop_size - len(next_population)
            distances = calculate_crowding_distance(front, objs_list, max_flags)

            # Sort front by crowding distance descending (infinity first) with deterministic secondary key
            sorted_front = sorted(
                front,
                key=lambda idx: (
                    -1e12 if distances[idx] == float("inf") else -distances[idx],
                    idx,
                ),
            )

            for idx in sorted_front[:remaining_needed]:
                next_population.append(combined_population[idx])
                next_evaluations.append(combined_evaluations[idx])
            break

    return next_population, next_evaluations


class BoundedParetoArchive:
    """
    Maintains a bounded non-dominated Pareto archive.
    Uses crowding distance sorting to preserve maximal objective-space spread
    and boundary extremes when capacity is exceeded.
    """

    def __init__(
        self,
        max_capacity: int = 10,
        max_flags: Optional[List[bool]] = None,
    ):
        self.max_capacity = max_capacity
        self.max_flags = (
            max_flags
            if max_flags is not None
            else [True, True, True, True, False, False, False]
        )
        self.archive: List[Dict[str, Any]] = []

    def update(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Updates archive with candidates, retaining only non-dominated solutions."""
        all_candidates = self.archive + candidates
        if not all_candidates:
            return []

        # Deduplicate based on chromosome configuration first
        unique_candidates = []
        seen = set()
        for item in all_candidates:
            key = tuple(item["chromosome"])
            if key not in seen:
                seen.add(key)
                unique_candidates.append(item)

        objs = [c["obj_vector"] for c in unique_candidates]
        fronts = fast_non_dominated_sort(objs, self.max_flags)
        if not fronts:
            return self.archive

        # Only retain Rank-0 (non-dominated) solutions
        first_front_indices = fronts[0]
        non_dominated_candidates = [unique_candidates[idx] for idx in first_front_indices]

        if len(non_dominated_candidates) <= self.max_capacity:
            self.archive = non_dominated_candidates
        else:
            # Use crowding distance on the non-dominated set to select diverse archive members
            front_objs = [c["obj_vector"] for c in non_dominated_candidates]
            front_indices = list(range(len(non_dominated_candidates)))
            distances = calculate_crowding_distance(
                front_indices, front_objs, self.max_flags
            )

            sorted_indices = sorted(
                front_indices,
                key=lambda idx: (
                    -1e12 if distances[idx] == float("inf") else -distances[idx],
                    idx,
                ),
            )
            self.archive = [
                non_dominated_candidates[idx] for idx in sorted_indices[: self.max_capacity]
            ]

        return self.archive

    def get_diversity_stats(self) -> Dict[str, float]:
        """Calculates crowding distance spread and span across archive objectives."""
        if len(self.archive) <= 1:
            return {"archive_size": len(self.archive), "mean_crowding_dist": 0.0}

        objs = [c["obj_vector"] for c in self.archive]
        distances = calculate_crowding_distance(
            list(range(len(self.archive))), objs, self.max_flags
        )
        finite_dists = [d for d in distances.values() if d != float("inf")]
        mean_d = float(np.mean(finite_dists)) if finite_dists else 1.0

        return {
            "archive_size": len(self.archive),
            "mean_crowding_dist": float(mean_d),
        }
