from evops.chromosome import (
    adapt_mutation_rate,
    is_valid_chromosome,
    mutate,
    random_chromosome,
)


def test_mutation_validity():
    c = random_chromosome()
    mutated = mutate(c, mutation_rate=1.0)
    assert is_valid_chromosome(mutated), f"Mutated chromosome {mutated} is invalid."


def test_adaptive_mutation_behavior():
    # Low diversity (< 0.20) should increase mutation rate
    r1 = adapt_mutation_rate(current_rate=0.20, diversity=0.10)
    assert r1 > 0.20, f"DAAM failed to boost mutation rate: {r1}"

    # High diversity (> 0.50) should decrease mutation rate
    r2 = adapt_mutation_rate(current_rate=0.20, diversity=0.60)
    assert r2 < 0.20, f"DAAM failed to reduce mutation rate: {r2}"
