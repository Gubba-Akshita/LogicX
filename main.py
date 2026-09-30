import argparse
import sys

from evops.competition_adapter import CompetitionAdapter
from evops.config import EvoForgeConfig
from evops.experiments import run_baselines_and_ablations


def main():
    parser = argparse.ArgumentParser(
        description="EvoForge-X: Event-Aware Pareto Evolutionary Multimodal Optimizer"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.json", help="Path to config JSON"
    )
    parser.add_argument(
        "--data_root", type=str, default="data/crisismmd", help="Path to CrisisMMD dataset"
    )
    parser.add_argument(
        "--ood_event", type=str, default="hurricane_irma", help="Name of held-out OOD disaster event"
    )
    parser.add_argument(
        "--synthetic-test", action="store_true", help="Force synthetic test data mode"
    )
    parser.add_argument(
        "--run_experiments", action="store_true", help="Run baseline and ablation benchmark experiments"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run empirical latency and memory benchmark"
    )
    args = parser.parse_args()

    if args.benchmark:
        from benchmark import run_benchmark
        run_benchmark()
        return

    config = EvoForgeConfig.load_from_json(args.config)
    config.data_root = args.data_root
    config.ood_event_name = args.ood_event
    if args.synthetic_test:
        config.synthetic_test_mode = True

    print("=" * 75)
    print("EvoForge-X: Event-Aware Pareto Evolutionary Multimodal Optimizer")
    print("=" * 75)
    data_mode_str = "SYNTHETIC TEST" if config.synthetic_test_mode else "REAL CRISIS MMD (if available, else fallback)"
    print(f"DATA MODE:       {data_mode_str}")
    print(f"Data Root:       {config.data_root}")
    print(f"OOD Event Split: {config.ood_event_name}")
    print(f"Population:      {config.population_size}")
    print(f"Generations:     {config.generations}")
    print("=" * 75)

    if args.run_experiments:
        run_baselines_and_ablations(config=config, verbose=True)
    else:
        adapter = CompetitionAdapter(config=config)
        adapter.fit(X=None, verbose=True)

        # Output Pareto Trade-off Positions
        print("\n" + "=" * 75)
        print("PARETO TRADE-OFF SOLUTIONS SUMMARY:")
        print("=" * 75)
        for i, item in enumerate(adapter.pareto_archive):
            print(
                f"Candidate {i+1:2d} | Chrom: {item['chromosome']} | "
                f"CleanAcc: {item['predictive_quality']:.4f} | "
                f"OODAcc: {item['ood_score']:.4f} | "
                f"Robustness: {item['robustness_score']:.4f} | "
                f"Latency: {item['latency']*1000:.2f}ms | "
                f"Params: {item['parameter_count']}"
            )
        print("=" * 75)


if __name__ == "__main__":
    main()
