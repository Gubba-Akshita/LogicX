"""
===============================================================================
EVOFORGE-X: EMPIRICAL BENCHMARK & PROFILING SUITE
===============================================================================
Measures empirical latency, memory usage, throughput, and computational complexity
across varying batch sizes and feature configurations.

100% Offline | Zero Network Dependencies | Deterministic & Reproducible
===============================================================================
"""

import json
import os
import sys
import time
import tracemalloc
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from evops.config import EvoForgeConfig
from evops.data_adapter import CrisisMMDAdapter
from evops.feature_extraction import CachedFeatureStore
from evops.model import CompactMultimodalModel


def run_benchmark(
    batch_sizes: List[int] = [1, 8, 16, 32, 64],
    random_seed: int = 42,
    output_json: str = "benchmark_results.json",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Runs comprehensive empirical profiling of the EvoForge-X architecture:
    - Preprocessing & TF-IDF vectorization time
    - Model construction and parameter footprint
    - Full backpropagation training time
    - Inference latency per sample (ms) and batch latency across batch sizes
    - Batch throughput (samples/sec)
    - Peak memory allocation via tracemalloc
    """
    if verbose:
        print("=" * 80)
        print("EVOFORGE-X: EMPIRICAL BENCHMARK & PROFILING SUITE")
        print("=" * 80)

    tracemalloc.start()

    # 1. Benchmark Data Ingestion & Preprocessing
    t_prep_start = time.perf_counter()
    adapter = CrisisMMDAdapter(random_seed=random_seed, force_synthetic=True)
    df = adapter.load_data()
    splits = adapter.create_event_splits(df, ood_event="hurricane_irma")

    feature_extractor = CachedFeatureStore(max_text_features=256)
    feature_store = feature_extractor.fit_transform(splits["train"], splits["ood"])
    t_prep_end = time.perf_counter()
    preprocessing_latency_sec = t_prep_end - t_prep_start

    X_text_tr = feature_store["train"]["text"]
    X_img_tr = feature_store["train"]["image"]
    y_tr = feature_store["train"]["label"]

    X_text_ood = feature_store["ood"]["text"]
    X_img_ood = feature_store["ood"]["image"]
    y_ood = feature_store["ood"]["label"]

    # 2. Benchmark Model Construction & Parameter Count
    t_init_start = time.perf_counter()
    model = CompactMultimodalModel(
        text_in_dim=X_text_tr.shape[1],
        img_in_dim=X_img_tr.shape[1],
        text_feature_dim=128,
        image_feature_dim=128,
        hidden_dim=64,
        fusion_type=1,  # Gated fusion (most complex)
        dropout=0.1,
        learning_rate=0.001,
        weight_decay=0.0001,
        modality_weight=0.5,
        early_stopping_patience=5,
        random_state=random_seed,
    )
    t_init_end = time.perf_counter()
    init_latency_sec = t_init_end - t_init_start
    total_params = model.count_parameters()

    # 3. Benchmark Training Time with Analytical Backpropagation
    t_train_start = time.perf_counter()
    model.fit(
        X_text_tr,
        X_img_tr,
        y_tr,
        X_val_text=X_text_ood[:50],
        X_val_img=X_img_ood[:50],
        y_val=y_ood[:50],
        epochs=15,
        batch_size=32,
    )
    t_train_end = time.perf_counter()
    train_latency_sec = t_train_end - t_train_start
    train_latency_per_epoch_ms = (train_latency_sec / 15.0) * 1000.0

    # 4. Benchmark Inference Across Batch Sizes
    batch_results = []
    for bs in batch_sizes:
        # Prepare batch samples (repeating if needed)
        idx = np.arange(bs) % len(X_text_ood)
        batch_text = X_text_ood[idx]
        batch_img = X_img_ood[idx]

        # Warmup
        model.predict(batch_text, batch_img)

        # Timed execution (100 repetitions)
        n_iters = 100
        t0 = time.perf_counter()
        for _ in range(n_iters):
            model.predict(batch_text, batch_img)
        t_batch_total = time.perf_counter() - t0

        avg_batch_latency_ms = (t_batch_total / n_iters) * 1000.0
        latency_per_sample_ms = avg_batch_latency_ms / bs
        throughput_samples_per_sec = (bs * n_iters) / t_batch_total

        batch_results.append({
            "batch_size": bs,
            "avg_batch_latency_ms": round(avg_batch_latency_ms, 4),
            "latency_per_sample_ms": round(latency_per_sample_ms, 4),
            "throughput_samples_per_sec": round(throughput_samples_per_sec, 2),
        })

    # 5. Track Memory Usage
    current_mem_bytes, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem_bytes / (1024.0 * 1024.0)
    current_mem_mb = current_mem_bytes / (1024.0 * 1024.0)

    # 6. Compile Results
    benchmark_data = {
        "status": "success",
        "dataset_samples": len(df),
        "train_samples": len(X_text_tr),
        "ood_samples": len(X_text_ood),
        "trainable_parameters": total_params,
        "preprocessing_time_sec": round(preprocessing_latency_sec, 4),
        "model_init_time_sec": round(init_latency_sec, 6),
        "training_time_15_epochs_sec": round(train_latency_sec, 4),
        "training_time_per_epoch_ms": round(train_latency_per_epoch_ms, 2),
        "single_sample_latency_ms": batch_results[0]["latency_per_sample_ms"],
        "peak_memory_mb": round(peak_mem_mb, 3),
        "current_memory_mb": round(current_mem_mb, 3),
        "batch_profiling": batch_results,
    }

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(benchmark_data, f, indent=2)

    # Print Formatted Console Table
    if verbose:
        print(f"Dataset Size:           {len(df)} samples ({len(X_text_tr)} train / {len(X_text_ood)} OOD)")
        print(f"Trainable Parameters:   {total_params:,} parameters")
        print(f"Preprocessing Time:     {preprocessing_latency_sec * 1000.0:.2f} ms")
        print(f"Training Time (15 ep):  {train_latency_sec * 1000.0:.2f} ms ({train_latency_per_epoch_ms:.2f} ms/epoch)")
        print(f"Peak Memory Allocated:  {peak_mem_mb:.3f} MB")
        print("-" * 80)
        print(f"{'Batch Size':<12} | {'Batch Latency (ms)':<20} | {'Latency / Sample (ms)':<22} | {'Throughput (samples/s)':<22}")
        print("-" * 80)
        for r in batch_results:
            print(
                f"{r['batch_size']:<12} | {r['avg_batch_latency_ms']:<20.4f} | "
                f"{r['latency_per_sample_ms']:<22.4f} | {r['throughput_samples_per_sec']:<22.2f}"
            )
        print("=" * 80)
        print(f"Benchmark results written to: {output_json}\n")

    return benchmark_data


if __name__ == "__main__":
    run_benchmark()
