import numpy as np

from evops.pareto import (
    calculate_crowding_distance,
    fast_non_dominated_sort,
    nsga2_environmental_selection,
)


def test_crowding_distance_boundary_infinity():
    """Verifies that boundary solutions receive infinite crowding distance to preserve extremes."""
    objs = [
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3],
        [0.6, 0.4],
    ]
    max_flags = [True, True]
    front_indices = [0, 1, 2, 3]

    distances = calculate_crowding_distance(front_indices, objs, max_flags)

    assert distances[0] == float("inf"), "Boundary extreme (max obj 0) must have infinite distance!"
    assert distances[3] == float("inf"), "Boundary extreme (min obj 0) must have infinite distance!"
    assert np.isfinite(distances[1]), "Intermediate solution must have finite crowding distance!"
    assert np.isfinite(distances[2]), "Intermediate solution must have finite crowding distance!"


def test_nsga2_environmental_selection_preserves_pop_size():
    """Verifies (P_t U Q_t) environmental selection precisely maintains target population size."""
    combined_pop = [[i] * 10 for i in range(16)]
    combined_evals = []
    for i in range(16):
        combined_evals.append({
            "obj_vector": [float(16 - i), float(i), 0.8, 0.8, 0.01, 0.5, 100],
            "max_flags": [True, True, True, True, False, False, False],
        })

    next_pop, next_evals = nsga2_environmental_selection(
        combined_population=combined_pop,
        combined_evaluations=combined_evals,
        target_pop_size=8,
    )

    assert len(next_pop) == 8, f"Expected 8 candidates in next generation, got {len(next_pop)}"
    assert len(next_evals) == 8, f"Expected 8 evaluations in next generation, got {len(next_evals)}"
