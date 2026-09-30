from evops.pareto import evaluate_pareto_dominance, fast_non_dominated_sort


def test_pareto_dominance():
    max_flags = [True, True, False]
    # obj1 is better in obj0 and obj1, lower in obj2 (min) -> dominates obj2
    obj1 = [0.9, 0.8, 0.1]
    obj2 = [0.7, 0.6, 0.5]
    assert evaluate_pareto_dominance(obj1, obj2, max_flags), "obj1 should dominate obj2"
    assert not evaluate_pareto_dominance(obj2, obj1, max_flags), "obj2 should not dominate obj1"


def test_non_dominated_sorting():
    max_flags = [True, True]
    objs = [
        [0.9, 0.9],  # Rank 0
        [0.5, 0.5],  # Dominated by 0
        [0.8, 0.8],  # Dominated by 0, dominates 1
    ]
    fronts = fast_non_dominated_sort(objs, max_flags)
    assert fronts[0] == [0], f"Expected front 0 to be [0], got {fronts[0]}"
