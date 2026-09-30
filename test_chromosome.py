from evops.chromosome import is_valid_chromosome, random_chromosome


def test_chromosome_validity():
    c = random_chromosome()
    assert is_valid_chromosome(c), f"Generated chromosome {c} is invalid."
    assert not is_valid_chromosome([999, 999, 999]), "Invalid chromosome passed check."
