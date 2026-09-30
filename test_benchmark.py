import os

from benchmark import run_benchmark


def test_empirical_benchmark_execution():
    """Verifies that the empirical benchmark runs offline and writes valid JSON metrics."""
    out_file = "test_benchmark_temp.json"
    if os.path.exists(out_file):
        os.remove(out_file)

    res = run_benchmark(
        batch_sizes=[1, 8],
        output_json=out_file,
        verbose=False,
    )

    assert res["status"] == "success"
    assert res["trainable_parameters"] > 0
    assert res["preprocessing_time_sec"] >= 0.0
    assert res["training_time_15_epochs_sec"] >= 0.0
    assert res["single_sample_latency_ms"] >= 0.0
    assert res["peak_memory_mb"] >= 0.0
    assert len(res["batch_profiling"]) == 2
    assert os.path.exists(out_file)

    if os.path.exists(out_file):
        os.remove(out_file)
