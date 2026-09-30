
import math
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================
N_ESTIMATORS_DOMAIN: List[int] = [50, 75, 100, 150]
MAX_DEPTH_DOMAIN: List[int] = [3, 5, 7, 10]
MIN_SAMPLES_SPLIT_DOMAIN: List[int] = [2, 4, 6, 10]

PARAM_DOMAINS: List[List[int]] = [
    N_ESTIMATORS_DOMAIN,
    MAX_DEPTH_DOMAIN,
    MIN_SAMPLES_SPLIT_DOMAIN,
]

POPULATION_SIZE: int = 8
GENERATIONS: int = 5
ELITE_COUNT: int = 1
MUTATION_RATE: float = 0.2
TOURNAMENT_SIZE: int = 2
RANDOM_SEED: int = 42

# Multi-Objective Weights
WEIGHT_PREDICTIVE: float = 0.50
WEIGHT_ROBUSTNESS: float = 0.35
WEIGHT_RUNTIME: float = 0.08
WEIGHT_COMPLEXITY: float = 0.07

MAX_ALLOWED_RUNTIME_SEC: float = 2.0
PERTURBATION_NOISE_SCALE: float = 0.05


# =============================================================================
# DATA PREPROCESSING AND VALIDATION
# =============================================================================
class DataPreprocessor:
    """
    Robust data validator and preprocessor for competitive ML sandboxes.
    Handles NaNs, Infs, categorical features, class imbalance, and split generation.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.feature_encoders: Dict[Any, OrdinalEncoder] = {}
        self.is_fitted = False

    def validate_and_preprocess(
        self, X: Any, y: Any, test_size: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Validates and cleans input data, then performs a leakage-free stratified split.
        """
        if X is None or y is None:
            raise ValueError("Input X and y cannot be None.")

        # Convert Pandas objects to DataFrame/Series if applicable
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        elif isinstance(X, list):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = np.array(y).ravel()
        else:
            y_arr = np.array(y).ravel()

        # Check empty inputs
        if X_df.empty or y_arr.size == 0:
            raise ValueError("Empty dataset provided.")

        if len(X_df) != len(y_arr):
            raise ValueError(
                f"Length mismatch: X has {len(X_df)} samples, y has {len(y_arr)} samples."
            )

        # Check class diversity
        unique_classes = np.unique(y_arr)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Insufficient class diversity: Target y has only {len(unique_classes)} unique class."
            )

        # Handle object/string categorical features and convert to numeric
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
                encoded_vals = encoder.fit_transform(X_df[[col]].astype(str)).ravel()
                X_df[col] = pd.Series(encoded_vals, index=X_df.index, dtype=np.float64)
                self.feature_encoders[col] = encoder
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        # Convert to float array and replace Inf / -Inf with NaN
        X_mat = np.array(X_df.to_numpy(dtype=np.float64), copy=True)
        X_mat[np.isinf(X_mat)] = np.nan

        # Leakage-free train/validation split
        try:
            X_tr_raw, X_val_raw, y_train, y_val = train_test_split(
                X_mat,
                y_arr,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y_arr,
            )
        except ValueError:
            # Fallback for extreme imbalance where min class count < 2
            X_tr_raw, X_val_raw, y_train, y_val = train_test_split(
                X_mat,
                y_arr,
                test_size=test_size,
                random_state=self.random_state,
            )

        # Fit imputer and scaler ONLY on training data to prevent leakage
        X_train = self.imputer.fit_transform(X_tr_raw)
        X_train = self.scaler.fit_transform(X_train)

        # Transform validation data using pre-fitted transformers
        X_val = self.imputer.transform(X_val_raw)
        X_val = self.scaler.transform(X_val)

        self.is_fitted = True
        return X_train, y_train, X_val, y_val

    def transform_new_data(self, X: Any) -> np.ndarray:
        """Transforms unseen dataset using pre-fitted encoders, imputer, and scaler."""
        if not self.is_fitted:
            raise RuntimeError("DataPreprocessor must be fitted before transforming new data.")

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        elif isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        elif isinstance(X, list):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X)

        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                if col in self.feature_encoders:
                    encoder = self.feature_encoders[col]
                    encoded_vals = encoder.transform(X_df[[col]].astype(str)).ravel()
                else:
                    encoder = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    encoded_vals = encoder.fit_transform(X_df[[col]].astype(str)).ravel()
                X_df[col] = pd.Series(encoded_vals, index=X_df.index, dtype=np.float64)
            else:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce")

        X_mat = np.array(X_df.to_numpy(dtype=np.float64), copy=True)
        X_mat[np.isinf(X_mat)] = np.nan
        X_clean = self.imputer.transform(X_mat)
        X_scaled = self.scaler.transform(X_clean)
        return X_scaled


# =============================================================================
# CORE GENETIC OPERATORS AND EVALUATION
# =============================================================================
def random_chromosome(rng: Optional[random.Random] = None) -> List[int]:
    """Generates a random valid chromosome from predefined hyperparameter domains."""
    r = rng if rng is not None else random
    return [r.choice(domain) for domain in PARAM_DOMAINS]


def is_valid_chromosome(chromosome: List[int]) -> bool:
    """Verifies that a chromosome adheres to discrete domain constraints."""
    if not isinstance(chromosome, (list, tuple)) or len(chromosome) != 3:
        return False
    return (
        chromosome[0] in N_ESTIMATORS_DOMAIN
        and chromosome[1] in MAX_DEPTH_DOMAIN
        and chromosome[2] in MIN_SAMPLES_SPLIT_DOMAIN
    )


def build_model(
    chromosome: List[int], random_state: Optional[int] = RANDOM_SEED
) -> RandomForestClassifier:
    """Instantiates a RandomForestClassifier configured by the chromosome."""
    return RandomForestClassifier(
        n_estimators=int(chromosome[0]),
        max_depth=int(chromosome[1]),
        min_samples_split=int(chromosome[2]),
        n_jobs=-1,
        random_state=random_state,
    )


def evaluate_robustness(
    model: RandomForestClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    noise_scale: float = PERTURBATION_NOISE_SCALE,
    random_state: Optional[int] = RANDOM_SEED,
) -> float:
    """
    Evaluates model resilience under validation dataset distribution shift / Gaussian noise.
    Returns a normalized robustness score in [0.0, 1.0].
    """
    if X_val.shape[0] == 0:
        return 0.0

    std_devs = np.std(X_val, axis=0)
    std_devs[std_devs == 0] = 1.0  # Avoid zero-variance division issues

    rng = np.random.RandomState(random_state)
    noise = rng.normal(loc=0.0, scale=noise_scale * std_devs, size=X_val.shape)
    X_perturbed = X_val + noise

    clean_preds = model.predict(X_val)
    perturbed_preds = model.predict(X_perturbed)

    clean_acc = float(balanced_accuracy_score(y_val, clean_preds))
    perturbed_acc = float(balanced_accuracy_score(y_val, perturbed_preds))

    # Penalize performance degradation under perturbation
    acc_drop = max(0.0, clean_acc - perturbed_acc)
    robustness_score = 1.0 - acc_drop
    return max(0.0, min(1.0, float(robustness_score)))


def calculate_fitness(
    predictive_quality: float,
    robustness_score: float,
    runtime_penalty: float,
    complexity_penalty: float,
) -> float:
    """
    Computes a normalized multi-objective fitness score:
    fitness = w1*PredictiveQuality + w2*Robustness - w3*RuntimePenalty - w4*ComplexityPenalty
    """
    pq = 0.0 if (math.isnan(predictive_quality) or math.isinf(predictive_quality)) else predictive_quality
    rs = 0.0 if (math.isnan(robustness_score) or math.isinf(robustness_score)) else robustness_score
    rt = 1.0 if (math.isnan(runtime_penalty) or math.isinf(runtime_penalty)) else runtime_penalty
    cx = 1.0 if (math.isnan(complexity_penalty) or math.isinf(complexity_penalty)) else complexity_penalty

    fitness = (
        (WEIGHT_PREDICTIVE * pq)
        + (WEIGHT_ROBUSTNESS * rs)
        - (WEIGHT_RUNTIME * rt)
        - (WEIGHT_COMPLEXITY * cx)
    )
    return max(0.0, min(1.0, float(fitness)))


def evaluate_chromosome(
    chromosome: List[int],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Fits and evaluates a chromosome, returning detailed multi-objective metrics."""
    t0 = time.perf_counter()
    model = build_model(chromosome, random_state=random_state)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = float(balanced_accuracy_score(y_val, val_preds))
    robustness_score = evaluate_robustness(
        model, X_val, y_val, random_state=random_state
    )

    elapsed_time = time.perf_counter() - t0
    runtime_penalty = min(1.0, elapsed_time / MAX_ALLOWED_RUNTIME_SEC)

    # Complexity penalty based on hyperparameter limits
    comp_n_est = chromosome[0] / max(N_ESTIMATORS_DOMAIN)
    comp_depth = chromosome[1] / max(MAX_DEPTH_DOMAIN)
    complexity_penalty = 0.5 * comp_n_est + 0.5 * comp_depth

    fitness = calculate_fitness(
        val_acc, robustness_score, runtime_penalty, complexity_penalty
    )

    return {
        "chromosome": list(chromosome),
        "fitness": fitness,
        "val_acc": val_acc,
        "robustness": robustness_score,
        "runtime": elapsed_time,
        "runtime_penalty": runtime_penalty,
        "complexity_penalty": complexity_penalty,
        "model": model,
    }


def select_parents(
    population: List[List[int]],
    fitness_scores: List[float],
    tournament_size: int = TOURNAMENT_SIZE,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Selects a parent chromosome using Tournament Selection."""
    r = rng if rng is not None else random
    indices = r.sample(
        range(len(population)), k=min(tournament_size, len(population))
    )
    best_idx = max(indices, key=lambda idx: fitness_scores[idx])
    return list(population[best_idx])


def crossover(
    parent1: List[int], parent2: List[int], rng: Optional[random.Random] = None
) -> Tuple[List[int], List[int]]:
    """Performs one-point crossover between two parent chromosomes."""
    r = rng if rng is not None else random
    crossover_point = r.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(
    chromosome: List[int],
    mutation_rate: float = MUTATION_RATE,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """Applies discrete domain mutation to a chromosome with probability mutation_rate."""
    r = rng if rng is not None else random
    mutated = list(chromosome)
    for gene_idx in range(len(mutated)):
        if r.random() < mutation_rate:
            domain = PARAM_DOMAINS[gene_idx]
            options = [val for val in domain if val != mutated[gene_idx]]
            if options:
                mutated[gene_idx] = r.choice(options)
            else:
                mutated[gene_idx] = r.choice(domain)
    return mutated


def create_next_population(
    population: List[List[int]],
    fitness_scores: List[float],
    elite_count: int = ELITE_COUNT,
    mutation_rate: float = MUTATION_RATE,
    rng: Optional[random.Random] = None,
) -> List[List[int]]:
    """Generates the next population incorporating elitism, selection, crossover, and mutation."""
    r = rng if rng is not None else random
    pop_size = len(population)
    next_pop: List[List[int]] = []

    # 1. Elitism: Preserve top elite_count individuals
    sorted_indices = sorted(
        range(pop_size), key=lambda i: fitness_scores[i], reverse=True
    )
    for i in range(min(elite_count, pop_size)):
        next_pop.append(list(population[sorted_indices[i]]))

    # 2. Reproduction loop
    while len(next_pop) < pop_size:
        p1 = select_parents(
            population, fitness_scores, tournament_size=TOURNAMENT_SIZE, rng=r
        )
        p2 = select_parents(
            population, fitness_scores, tournament_size=TOURNAMENT_SIZE, rng=r
        )
        child1, child2 = crossover(p1, p2, rng=r)

        child1 = mutate(child1, mutation_rate=mutation_rate, rng=r)
        next_pop.append(child1)
        if len(next_pop) < pop_size:
            child2 = mutate(child2, mutation_rate=mutation_rate, rng=r)
            next_pop.append(child2)

    return next_pop


# =============================================================================
# EVOLUTIONARY OPTIMIZER ENGINE
# =============================================================================
def run_evolution(
    X: Any,
    y: Any,
    population_size: int = POPULATION_SIZE,
    generations: int = GENERATIONS,
    elite_count: int = ELITE_COUNT,
    mutation_rate: float = MUTATION_RATE,
    random_state: int = RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes the complete evolutionary optimization loop.
    Maintains a monotonic best-so-far solution and logs generation progress.
    """
    rng = random.Random(random_state)
    preprocessor = DataPreprocessor(random_state=random_state)
    X_train, y_train, X_val, y_val = preprocessor.validate_and_preprocess(X, y)

    population = [random_chromosome(rng) for _ in range(population_size)]

    best_so_far_res: Optional[Dict[str, Any]] = None
    convergence_history: List[Dict[str, Any]] = []

    if verbose:
        print("=" * 70)
        print("PA-EMLO: PERTURBATION-AWARE EVOLUTIONARY OPTIMIZATION STARTING")
        print(
            f"Population: {population_size} | Generations: {generations} | Elites: {elite_count}"
        )
        print("=" * 70)

    for gen in range(1, generations + 1):
        gen_evals = []
        gen_fitnesses = []

        for chrom in population:
            res = evaluate_chromosome(
                chrom, X_train, y_train, X_val, y_val, random_state=random_state
            )
            gen_evals.append(res)
            gen_fitnesses.append(res["fitness"])

        best_gen_idx = max(range(len(gen_evals)), key=lambda i: gen_fitnesses[i])
        best_gen_res = gen_evals[best_gen_idx]

        # Enforce monotonic best-so-far solution tracking
        if best_so_far_res is None or best_gen_res["fitness"] > best_so_far_res["fitness"]:
            best_so_far_res = best_gen_res.copy()

        convergence_history.append(
            {
                "generation": gen,
                "gen_best_fitness": best_gen_res["fitness"],
                "best_so_far_fitness": best_so_far_res["fitness"],
                "best_chromosome": list(best_so_far_res["chromosome"]),
                "val_acc": best_so_far_res["val_acc"],
                "robustness": best_so_far_res["robustness"],
                "runtime": best_so_far_res["runtime"],
            }
        )

        if verbose:
            print(
                f"Generation {gen:2d} | Best fitness: {best_so_far_res['fitness']:.4f} "
                f"| Val Acc: {best_so_far_res['val_acc']:.4f} "
                f"| Robustness: {best_so_far_res['robustness']:.4f} "
                f"| Chrom: {best_so_far_res['chromosome']}"
            )

        population = create_next_population(
            population,
            gen_fitnesses,
            elite_count=elite_count,
            mutation_rate=mutation_rate,
            rng=rng,
        )

    if verbose and best_so_far_res is not None:
        print("=" * 70)
        print("OPTIMIZATION COMPLETED")
        print(f"Best Chromosome:    {best_so_far_res['chromosome']}")
        print(f"Validation Score:   {best_so_far_res['val_acc']:.4f}")
        print(f"Robustness Score:   {best_so_far_res['robustness']:.4f}")
        print(f"Runtime (seconds):  {best_so_far_res['runtime']:.4f}")
        print(f"Final Fitness:      {best_so_far_res['fitness']:.4f}")
        print("=" * 70)

    return {
        "best_chromosome": best_so_far_res["chromosome"],
        "best_fitness": best_so_far_res["fitness"],
        "best_val_acc": best_so_far_res["val_acc"],
        "best_robustness": best_so_far_res["robustness"],
        "best_runtime": best_so_far_res["runtime"],
        "best_model": best_so_far_res["model"],
        "preprocessor": preprocessor,
        "history": convergence_history,
    }


# =============================================================================
# COMPETITION SANDBOX ADAPTER
# =============================================================================
class CompetitionAdapter:
    """
    Decoupled interface adapter connecting the core evolutionary optimizer 
    with competition sandbox benchmark environments.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.best_model: Optional[RandomForestClassifier] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.optimization_summary: Optional[Dict[str, Any]] = None

    def fit(self, X: Any, y: Any, verbose: bool = False) -> "CompetitionAdapter":
        """Runs hyperparameter search and fits the best model on sanitized data."""
        results = run_evolution(X, y, random_state=self.random_state, verbose=verbose)
        self.best_model = results["best_model"]
        self.preprocessor = results["preprocessor"]
        self.optimization_summary = results
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels for raw input data X using the best fitted model."""
        if self.best_model is None or self.preprocessor is None:
            raise RuntimeError("Adapter must be fitted before calling predict().")

        X_transformed = self.preprocessor.transform_new_data(X)
        return self.best_model.predict(X_transformed)


# =============================================================================
# INTERNAL TESTING SUITE
# =============================================================================
def run_tests() -> bool:
    """
    Executes internal unit tests verifying algorithm correctness, operator bounds,
    elitism preservation, monotonic convergence, and edge-case data handling.
    """
    print("=" * 70)
    print("RUNNING INTERNAL TEST SUITE (run_tests)")
    print("=" * 70)

    # Test 1: Chromosome Validity
    c = random_chromosome()
    assert is_valid_chromosome(c), f"Test 1 Failed: Chromosome {c} invalid."
    print("[PASS] Test 1: Chromosome Validity")

    # Test 2: Crossover Validity
    p1 = [50, 3, 2]
    p2 = [150, 10, 10]
    ch1, ch2 = crossover(p1, p2)
    assert is_valid_chromosome(ch1) and is_valid_chromosome(ch2), (
        "Test 2 Failed: Crossover output invalid."
    )
    print("[PASS] Test 2: Crossover Producing Valid Chromosomes")

    # Test 3: Mutation Validity
    mutated = mutate(p1, mutation_rate=1.0)
    assert is_valid_chromosome(mutated), f"Test 3 Failed: Mutated chromosome {mutated} invalid."
    print("[PASS] Test 3: Mutation Producing Valid Chromosomes")

    # Test 4: Fitness Value Bounds & Stability
    f_val = calculate_fitness(0.85, 0.90, 0.1, 0.2)
    assert 0.0 <= f_val <= 1.0 and math.isfinite(f_val), (
        f"Test 4 Failed: Fitness {f_val} out of bounds/non-finite."
    )
    f_nan = calculate_fitness(
        float("nan"), float("inf"), float("nan"), float("inf")
    )
    assert 0.0 <= f_nan <= 1.0 and math.isfinite(f_nan), (
        "Test 4 Failed: NaN/Inf fitness handling failed."
    )
    print("[PASS] Test 4: Fitness Returning Finite Values")

    # Test 5: Population Size Constancy
    pop = [random_chromosome() for _ in range(8)]
    scores = [float(i) / 10.0 for i in range(8)]
    next_pop = create_next_population(
        pop, scores, elite_count=1, mutation_rate=0.2
    )
    assert len(next_pop) == len(pop), (
        f"Test 5 Failed: Pop size changed from {len(pop)} to {len(next_pop)}."
    )
    print("[PASS] Test 5: Population Size Remaining Constant")

    # Test 6: Elite Preservation
    best_idx = 5
    scores[best_idx] = 0.999
    expected_elite = list(pop[best_idx])
    next_pop_elite = create_next_population(
        pop, scores, elite_count=1, mutation_rate=0.2
    )
    assert next_pop_elite[0] == expected_elite, "Test 6 Failed: Elite individual not preserved."
    print("[PASS] Test 6: Elite Preservation")

    # Test 7: Convergence / Monotonic Best-So-Far Tracking
    rng = np.random.RandomState(42)
    X_syn = rng.randn(100, 5)
    y_syn = (X_syn[:, 0] + X_syn[:, 1] > 0).astype(int)
    results = run_evolution(
        X_syn, y_syn, population_size=4, generations=3, verbose=False
    )
    history = results["history"]
    fitnesses = [h["best_so_far_fitness"] for h in history]
    for i in range(1, len(fitnesses)):
        assert fitnesses[i] >= fitnesses[i - 1], (
            f"Test 7 Failed: Non-monotonic fitness at gen {i}: {fitnesses}"
        )
    print("[PASS] Test 7: Convergence / Monotonic Best-So-Far Tracking")

    # Test 8: Malformed & Edge-Case Input Handling
    # 8a. Handling NaNs, Infs, and String Features
    X_bad = pd.DataFrame(
        {
            "num1": [1.0, 2.0, np.nan, 4.0, np.inf, 6.0, 7.0, 8.0, 9.0, 10.0],
            "cat1": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    y_bad = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    prep = DataPreprocessor(random_state=42)
    X_tr, y_tr, X_va, y_va = prep.validate_and_preprocess(X_bad, y_bad)
    assert not np.isnan(X_tr).any() and not np.isinf(X_tr).any(), (
        "Test 8a Failed: NaN/Inf preprocessing failed."
    )

    # 8b. Empty Dataset Error Handling
    try:
        prep.validate_and_preprocess([], [])
        assert False, "Test 8b Failed: Empty dataset should raise ValueError."
    except ValueError:
        pass

    # 8c. Single Class Target Error Handling
    try:
        prep.validate_and_preprocess(np.ones((10, 2)), np.zeros(10))
        assert False, "Test 8c Failed: Single class target should raise ValueError."
    except ValueError:
        pass

    print("[PASS] Test 8: Malformed & Insufficient Input Handling")
    print("=" * 70)
    print("ALL 8 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        # Run test suite first to ensure system integrity
        run_tests()

        # Run synthetic benchmark optimization demonstration
        print("\nRUNNING BENCHMARK DATASET DEMONSTRATION...")
        np.random.seed(RANDOM_SEED)
        n_samples, n_features = 300, 10
        X_bench = np.random.randn(n_samples, n_features)
        y_bench = (
            X_bench[:, 0] * 0.5 + X_bench[:, 1] ** 2 - X_bench[:, 2] > 0.5
        ).astype(int)

        adapter = CompetitionAdapter(random_state=RANDOM_SEED)
        adapter.fit(X_bench, y_bench, verbose=True)

        preds = adapter.predict(X_bench[:10])
        print(f"\nSample Predictions (First 10 instances): {preds}")
