import json
import os
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from evops.chromosome import (
    adapt_mutation_rate,
    calculate_objective_diversity,
    calculate_population_diversity,
    calculate_unique_chromosome_ratio,
    crossover,
    is_valid_chromosome,
    mutate,
    random_chromosome,
)
from evops.config import EvoForgeConfig
from evops.evaluation import evaluate_chromosome
from evops.pareto import (
    BoundedParetoArchive,
    fast_non_dominated_sort,
    nsga2_environmental_selection,
)


def run_evoforge_evolution(
    feature_store: Dict[str, Any],
    config: Optional[EvoForgeConfig] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete EvoForge-X NSGA-II Evolutionary Optimization loop.
    Features:
    1. Fast Non-Dominated Sorting & Crowding Distance
    2. True (mu + lambda) NSGA-II Environmental Selection
    3. Diversity-Aware Adaptive Mutation (DAAM)
    4. Diversity-Preserving Bounded Pareto Archive
    5. Configurable Runtime Budget with Graceful Exit
    6. Comprehensive JSON Convergence Trace Logging
    """
    cfg = config if config is not None else EvoForgeConfig()
    rng = random.Random(cfg.random_seed)

    # 1. Initialize and evaluate Population P_0
    population = [
        random_chromosome(rng) for _ in range(cfg.population_size)
    ]
    evaluations = [
        evaluate_chromosome(chrom, feature_store, random_state=cfg.random_seed)
        for chrom in population
    ]

    mutation_rate = cfg.initial_mutation_rate
    archive = BoundedParetoArchive(
        max_capacity=cfg.max_archive_size,
        max_flags=evaluations[0]["max_flags"],
    )
    archive.update(evaluations)

    convergence_history: List[Dict[str, Any]] = []

    if verbose:
        print("=" * 75)
        print("EVOFORGE-X: NSGA-II PARETO EVOLUTIONARY OPTIMIZER STARTING")
        print(
            f"Pop: {cfg.population_size} | Gens: {cfg.generations} | "
            f"OOD Event: {feature_store['ood'].get('event_name', 'N/A')} | "
            f"Budget: {cfg.runtime_budget_sec:.1f}s"
        )
        print("=" * 75)

    start_time = time.perf_counter()

    for gen in range(1, cfg.generations + 1):
        elapsed_sec = time.perf_counter() - start_time
        if elapsed_sec >= cfg.runtime_budget_sec:
            if verbose:
                print(f"[BUDGET REACHED] Elapsed time ({elapsed_sec:.2f}s) exceeded budget ({cfg.runtime_budget_sec:.1f}s). Stopping.")
            break

        # 2. Diversity Metrics (DAAM Controller)
        pop_diversity = calculate_population_diversity(population)
        unique_ratio = calculate_unique_chromosome_ratio(population)
        obj_diversity = calculate_objective_diversity(evaluations)

        mutation_rate = adapt_mutation_rate(
            mutation_rate,
            pop_diversity,
            target_low=cfg.daam_target_low,
            target_high=cfg.daam_target_high,
            scale_up=cfg.mutation_scale_up,
            scale_down=cfg.mutation_scale_down,
            min_rate=cfg.min_mutation_rate,
            max_rate=cfg.max_mutation_rate,
        )

        # 3. Offspring Generation Q_t using binary tournament & Pareto rank
        offspring_pop = []
        objs_list = [e["obj_vector"] for e in evaluations]
        fronts = fast_non_dominated_sort(objs_list, evaluations[0]["max_flags"])

        rank_map = {}
        for rank_idx, front in enumerate(fronts):
            for pop_idx in front:
                rank_map[pop_idx] = rank_idx

        def tournament_select() -> int:
            i1, i2 = rng.sample(range(len(population)), 2)
            if rank_map[i1] < rank_map[i2]:
                return i1
            elif rank_map[i2] < rank_map[i1]:
                return i2
            return i1 if rng.random() < 0.5 else i2

        while len(offspring_pop) < cfg.population_size:
            p1_idx = tournament_select()
            p2_idx = tournament_select()

            p1 = population[p1_idx]
            p2 = population[p2_idx]

            c1, c2 = crossover(p1, p2, rng=rng)
            c1 = mutate(c1, mutation_rate, rng=rng)
            offspring_pop.append(c1)

            if len(offspring_pop) < cfg.population_size:
                c2 = mutate(c2, mutation_rate, rng=rng)
                offspring_pop.append(c2)

        # 4. Evaluate Offspring Q_t
        offspring_evals = [
            evaluate_chromosome(chrom, feature_store, random_state=cfg.random_seed + gen)
            for chrom in offspring_pop
        ]

        # 5. True NSGA-II Environmental Selection: (P_t U Q_t -> P_{t+1})
        combined_pop = population + offspring_pop
        combined_evals = evaluations + offspring_evals

        population, evaluations = nsga2_environmental_selection(
            combined_population=combined_pop,
            combined_evaluations=combined_evals,
            target_pop_size=cfg.population_size,
            max_flags=evaluations[0]["max_flags"],
        )

        # 6. Update Diversity-Preserving Bounded Pareto Archive
        archive.update(evaluations)

        # 7. Extract Convergence Logging Metrics
        best_pred = max(e["predictive_quality"] for e in evaluations)
        best_ood = max(e["ood_score"] for e in evaluations)
        best_robustness = max(e["robustness_score"] for e in evaluations)
        best_stability = max(e["stability_score"] for e in evaluations)
        best_lat = min(e["latency"] for e in evaluations)
        avg_lat = float(np.mean([e["latency"] for e in evaluations]))
        avg_comp = float(np.mean([e["complexity"] for e in evaluations]))
        front_0_size = len(fast_non_dominated_sort([e["obj_vector"] for e in evaluations], evaluations[0]["max_flags"])[0])

        log_entry = {
            "generation": gen,
            "population_size": len(population),
            "pareto_front_size": front_0_size,
            "archive_size": len(archive.archive),
            "unique_chromosome_ratio": float(unique_ratio),
            "chromosome_diversity": float(pop_diversity),
            "objective_diversity": float(obj_diversity),
            "mutation_rate": float(mutation_rate),
            "best_predictive": float(best_pred),
            "best_OOD": float(best_ood),
            "best_robustness": float(best_robustness),
            "best_stability": float(best_stability),
            "best_latency": float(best_lat),
            "average_latency": float(avg_lat),
            "average_complexity": float(avg_comp),
            "total_elapsed_time": float(time.perf_counter() - start_time),
        }
        convergence_history.append(log_entry)

        if verbose:
            print(
                f"Gen {gen:2d} | Archive: {len(archive.archive):2d} | Front0: {front_0_size:2d} | "
                f"Div: {pop_diversity:.3f} | MutRate: {mutation_rate:.3f} | "
                f"Pred: {best_pred:.4f} | OOD: {best_ood:.4f} | Rob: {best_robustness:.4f}"
            )

    total_runtime = time.perf_counter() - start_time

    # Write convergence history to local JSON file for verifiable audit trail
    try:
        with open("convergence_history.json", "w") as f:
            json.dump(convergence_history, f, indent=2)
    except Exception:
        pass

    if verbose:
        print("=" * 75)
        print("OPTIMIZATION COMPLETE")
        print(f"Total Runtime:       {total_runtime:.4f} seconds")
        print(f"Pareto Archive Size: {len(archive.archive)} solutions")
        print("=" * 75)

    return {
        "archive": archive.archive,
        "history": convergence_history,
        "total_runtime": total_runtime,
        "config": cfg,
    }
