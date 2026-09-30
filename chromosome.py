import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from evops.config import FUSION_TYPE_MAP, GENE_NAMES, PARAM_DOMAINS


def random_chromosome(rng: Optional[random.Random] = None) -> List[Any]:
    """Generates a random valid chromosome from bounded hyperparameter domains."""
    r = rng if rng is not None else random
    return [r.choice(domain) for domain in PARAM_DOMAINS]


def is_valid_chromosome(chromosome: List[Any]) -> bool:
    """Verifies chromosome validity against discrete domain constraints."""
    if not isinstance(chromosome, (list, tuple)) or len(chromosome) != len(
        PARAM_DOMAINS
    ):
        return False
    for gene, domain in zip(chromosome, PARAM_DOMAINS):
        if gene not in domain:
            return False
    return True


def chromosome_to_dict(chromosome: List[Any]) -> Dict[str, Any]:
    """Converts a chromosome vector to a named hyperparameter dictionary."""
    d = {name: val for name, val in zip(GENE_NAMES, chromosome)}
    d["fusion_type_str"] = FUSION_TYPE_MAP.get(d["fusion_type"], "concat")
    return d


def crossover(
    parent1: List[Any], parent2: List[Any], rng: Optional[random.Random] = None
) -> Tuple[List[Any], List[Any]]:
    """Performs one-point crossover between two parent chromosomes."""
    r = rng if rng is not None else random
    point = r.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(
    chromosome: List[Any],
    mutation_rate: float,
    rng: Optional[random.Random] = None,
) -> List[Any]:
    """Applies discrete uniform mutation to a chromosome with specified probability."""
    r = rng if rng is not None else random
    mutated = list(chromosome)
    for idx in range(len(mutated)):
        if r.random() < mutation_rate:
            domain = PARAM_DOMAINS[idx]
            options = [val for val in domain if val != mutated[idx]]
            if options:
                mutated[idx] = r.choice(options)
            else:
                mutated[idx] = r.choice(domain)
    return mutated


def calculate_population_diversity(population: List[List[Any]]) -> float:
    """Calculates normalized Hamming distance across population chromosomes."""
    if len(population) <= 1:
        return 0.0
    n = len(population)
    num_genes = len(population[0])
    total_dist = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = sum(
                1 for g1, g2 in zip(population[i], population[j]) if g1 != g2
            )
            total_dist += dist / num_genes
            pairs += 1
    return float(total_dist / pairs) if pairs > 0 else 0.0


def calculate_unique_chromosome_ratio(population: List[List[Any]]) -> float:
    """Calculates ratio of unique chromosomes to total population size."""
    if not population:
        return 0.0
    unique_count = len(set(tuple(c) for c in population))
    return float(unique_count / len(population))


def calculate_objective_diversity(evaluations: List[Dict[str, Any]]) -> float:
    """Calculates average normalized variance across objective space."""
    if len(evaluations) <= 1:
        return 0.0
    objs = np.array([e["obj_vector"] for e in evaluations])
    variances = np.var(objs, axis=0)
    return float(np.mean(variances))


def adapt_mutation_rate(
    current_rate: float,
    diversity: float,
    target_low: float = 0.20,
    target_high: float = 0.50,
    scale_up: float = 1.35,
    scale_down: float = 0.85,
    min_rate: float = 0.05,
    max_rate: float = 0.50,
) -> float:
    """
    Diversity-Aware Adaptive Mutation (DAAM):
    Deterministically adapts mutation rate based on population diversity.
    - Low diversity / early stagnation -> increase exploration (scale_up)
    - High diversity -> reduce mutation for fine exploitation (scale_down)
    """
    if diversity < target_low:
        new_rate = min(max_rate, current_rate * scale_up)
    elif diversity > target_high:
        new_rate = max(min_rate, current_rate * scale_down)
    else:
        new_rate = current_rate
    return float(new_rate)
