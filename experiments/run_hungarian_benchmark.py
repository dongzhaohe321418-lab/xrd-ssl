"""
Session 3: Hungarian vs sort-match runtime comparison.

Demonstrates that sort-match achieves identical accuracy to Hungarian
matching but is ~100x faster for n=50 peaks.

This is the "elegance argument" — the theorem isn't just correct, it's fast.

Usage:
    python experiments/run_hungarian_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import sort_match_loss, masked_sort_match_loss, hungarian_reference


RESULTS_DIR = "experiments/results_session3"


def benchmark_runtime():
    """Compare wall-clock time at different n values."""
    results = {}
    rng = np.random.default_rng(42)

    for n in [10, 20, 30, 50, 100]:
        B = 32
        pred = torch.from_numpy(rng.uniform(5, 90, (B, n))).double()
        target = torch.from_numpy(rng.uniform(5, 90, (B, n))).double()

        # Warm up
        sort_match_loss(pred, target, cost="mae")
        hungarian_reference(pred, target, cost="mae")

        # Time sort_match
        n_iter = 200
        t0 = time.perf_counter()
        for _ in range(n_iter):
            sort_match_loss(pred, target, cost="mae")
        t_sm = (time.perf_counter() - t0) / n_iter * 1000  # ms

        # Time hungarian
        n_iter_h = min(n_iter, 50 if n >= 50 else 200)
        t0 = time.perf_counter()
        for _ in range(n_iter_h):
            hungarian_reference(pred, target, cost="mae")
        t_hu = (time.perf_counter() - t0) / n_iter_h * 1000  # ms

        speedup = t_hu / t_sm
        results[n] = {
            "n": n,
            "sort_match_ms": round(t_sm, 4),
            "hungarian_ms": round(t_hu, 4),
            "speedup": round(speedup, 1),
        }

    return results


def verify_equivalence():
    """Verify accuracy is identical at each n."""
    rng = np.random.default_rng(2026)
    results = {}

    for n in [10, 20, 30, 50]:
        max_error = 0.0
        for _ in range(50):
            B = 8
            pred = torch.from_numpy(rng.uniform(5, 90, (B, n))).double()
            target = torch.from_numpy(rng.uniform(5, 90, (B, n))).double()

            sm = sort_match_loss(pred, target, cost="mae").item()
            hu = hungarian_reference(pred, target, cost="mae")
            max_error = max(max_error, abs(sm - hu))

        results[n] = {"n": n, "max_error": max_error}

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Benchmarking runtime...")
    runtime = benchmark_runtime()

    print(f"\n{'n':<6} {'Sort-match':>12} {'Hungarian':>12} {'Speedup':>10}")
    print(f"{'-'*40}")
    for n, r in sorted(runtime.items()):
        print(f"{n:<6} {r['sort_match_ms']:>10.3f}ms {r['hungarian_ms']:>10.3f}ms {r['speedup']:>9.1f}x")

    print("\nVerifying equivalence...")
    equiv = verify_equivalence()
    for n, r in sorted(equiv.items()):
        print(f"  n={n}: max_error = {r['max_error']:.2e}")

    all_results = {"runtime": runtime, "equivalence": equiv}
    with open(os.path.join(RESULTS_DIR, "hungarian_benchmark.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved")


if __name__ == "__main__":
    main()
