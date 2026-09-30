from evops.chromosome import crossover, is_valid_chromosome, random_chromosome


def test_crossover_validity():
    p1 = random_chromosome()
    p2 = random_chromosome()
    ch1, ch2 = crossover(p1, p2)
    assert is_valid_chromosome(ch1) and is_valid_chromosome(ch2), "Crossover created invalid chromosome."
