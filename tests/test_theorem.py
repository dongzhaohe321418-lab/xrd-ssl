"""
Numerical verification of the 1D sort-match theorem.

Theorem: For 1D values with any convex cost c, sort-match gives the same
optimal matching cost as Hungarian (O(n^3) linear sum assignment).

We verify this by generating random prediction/target pairs and checking
that sort_match_loss == hungarian_reference to machine precision.

Runtime target: <5 seconds for all tests.
"""

import sys
import os
import time

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import sort_match_loss, masked_sort_match_loss, hungarian_reference


# ── Fixtures ─────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
TOLERANCE = 1e-10  # should match to near machine precision for float64


def random_batch(batch_size: int, n_peaks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random pred/target in typical XRD 2theta range [5, 90]."""
    pred = torch.from_numpy(RNG.uniform(5.0, 90.0, (batch_size, n_peaks))).double()
    target = torch.from_numpy(RNG.uniform(5.0, 90.0, (batch_size, n_peaks))).double()
    return pred, target


# ── Test 1: Sort-match == Hungarian for MSE cost ─────────────────────

@pytest.mark.parametrize("n_peaks", [10, 20, 30, 50])
def test_sort_match_equals_hungarian_mse(n_peaks: int):
    """sort_match_loss with MSE cost must equal hungarian_reference."""
    batch_size = 8
    pred, target = random_batch(batch_size, n_peaks)

    sm_loss = sort_match_loss(pred, target, cost="mse").item()
    hu_loss = hungarian_reference(pred, target, cost="mse")

    assert abs(sm_loss - hu_loss) < TOLERANCE, (
        f"MSE mismatch for n={n_peaks}: sort_match={sm_loss:.15e}, "
        f"hungarian={hu_loss:.15e}, diff={abs(sm_loss - hu_loss):.2e}"
    )


# ── Test 2: Sort-match == Hungarian for MAE cost ─────────────────────

@pytest.mark.parametrize("n_peaks", [10, 20, 30, 50])
def test_sort_match_equals_hungarian_mae(n_peaks: int):
    """sort_match_loss with MAE cost must equal hungarian_reference."""
    batch_size = 8
    pred, target = random_batch(batch_size, n_peaks)

    sm_loss = sort_match_loss(pred, target, cost="mae").item()
    hu_loss = hungarian_reference(pred, target, cost="mae")

    assert abs(sm_loss - hu_loss) < TOLERANCE, (
        f"MAE mismatch for n={n_peaks}: sort_match={sm_loss:.15e}, "
        f"hungarian={hu_loss:.15e}, diff={abs(sm_loss - hu_loss):.2e}"
    )


# ── Test 3: Sort-match == Hungarian for Huber cost ───────────────────

@pytest.mark.parametrize("n_peaks", [10, 20, 30, 50])
def test_sort_match_equals_hungarian_huber(n_peaks: int):
    """sort_match_loss with Huber cost must equal hungarian_reference."""
    batch_size = 8
    pred, target = random_batch(batch_size, n_peaks)

    sm_loss = sort_match_loss(pred, target, cost="huber").item()
    hu_loss = hungarian_reference(pred, target, cost="huber")

    assert abs(sm_loss - hu_loss) < TOLERANCE, (
        f"Huber mismatch for n={n_peaks}: sort_match={sm_loss:.15e}, "
        f"hungarian={hu_loss:.15e}, diff={abs(sm_loss - hu_loss):.2e}"
    )


# ── Test 4: Masked sort-match == Hungarian ───────────────────────────

@pytest.mark.parametrize("n_peaks", [10, 30, 50])
def test_masked_sort_match_equals_hungarian(n_peaks: int):
    """masked_sort_match_loss must equal hungarian_reference with same mask."""
    batch_size = 8
    pred, target = random_batch(batch_size, n_peaks)

    # Random mask: each sample has between 5 and n_peaks valid peaks
    mask = torch.zeros(batch_size, n_peaks, dtype=torch.bool)
    for i in range(batch_size):
        n_valid = RNG.integers(5, n_peaks + 1)
        mask[i, :n_valid] = True

    for cost_fn in ["mse", "mae", "huber"]:
        sm_loss = masked_sort_match_loss(pred, target, mask, cost=cost_fn).item()
        hu_loss = hungarian_reference(pred, target, cost=cost_fn, mask=mask)

        assert abs(sm_loss - hu_loss) < TOLERANCE, (
            f"Masked {cost_fn} mismatch for n={n_peaks}: "
            f"sort_match={sm_loss:.15e}, hungarian={hu_loss:.15e}, "
            f"diff={abs(sm_loss - hu_loss):.2e}"
        )


# ── Test 5: Edge cases ───────────────────────────────────────────────

def test_identical_inputs():
    """Loss should be zero when pred == target."""
    pred = torch.randn(4, 20).double()
    target = pred.clone()
    loss = sort_match_loss(pred, target, cost="mae").item()
    assert loss < 1e-15, f"Expected ~0 loss for identical inputs, got {loss}"


def test_single_peak():
    """Should work with n=1 (trivial matching)."""
    pred = torch.tensor([[45.0]]).double()
    target = torch.tensor([[30.0]]).double()
    loss = sort_match_loss(pred, target, cost="mae").item()
    assert abs(loss - 15.0) < 1e-10


def test_empty_mask():
    """All-False mask should return 0 loss without error."""
    pred = torch.randn(4, 20).double()
    target = torch.randn(4, 20).double()
    mask = torch.zeros(4, 20, dtype=torch.bool)
    loss = masked_sort_match_loss(pred, target, mask, cost="mae")
    assert loss.item() == 0.0


# ── Test 6: Gradient flow ────────────────────────────────────────────

def test_gradient_flows():
    """sort_match_loss must be differentiable w.r.t. predictions."""
    pred = torch.randn(4, 50, requires_grad=True, dtype=torch.float32)
    target = torch.randn(4, 50, dtype=torch.float32)

    loss = sort_match_loss(pred, target, cost="mae")
    loss.backward()

    assert pred.grad is not None, "No gradient computed"
    assert not torch.isnan(pred.grad).any(), "NaN in gradients"
    assert pred.grad.abs().sum() > 0, "All-zero gradients"


def test_masked_gradient_flows():
    """masked_sort_match_loss must be differentiable."""
    pred = torch.randn(4, 50, requires_grad=True, dtype=torch.float32)
    target = torch.randn(4, 50, dtype=torch.float32)
    mask = torch.ones(4, 50, dtype=torch.bool)
    mask[:, 40:] = False  # last 10 are padding

    loss = masked_sort_match_loss(pred, target, mask, cost="mae")
    loss.backward()

    assert pred.grad is not None
    # Gradients for masked positions should be zero
    assert (pred.grad[:, 40:] == 0).all(), "Masked positions have nonzero gradient"


# ── Test 7: Many random trials ──────────────────────────────────────

def test_many_random_trials():
    """Run 100 random trials, track max error across all.

    This is the strongest verification: if sort-match != Hungarian for ANY
    trial, the theorem implementation is wrong.
    """
    max_error = 0.0
    n_trials = 100
    rng = np.random.default_rng(2026)

    for trial in range(n_trials):
        n = rng.integers(10, 51)
        b = rng.integers(1, 9)
        pred = torch.from_numpy(rng.uniform(5.0, 90.0, (b, n))).double()
        target = torch.from_numpy(rng.uniform(5.0, 90.0, (b, n))).double()

        for cost_fn in ["mse", "mae", "huber"]:
            sm = sort_match_loss(pred, target, cost=cost_fn).item()
            hu = hungarian_reference(pred, target, cost=cost_fn)
            error = abs(sm - hu)
            max_error = max(max_error, error)

    print(f"\n  [100 random trials] Max error across all costs: {max_error:.2e}")
    assert max_error < TOLERANCE, f"Max error {max_error:.2e} exceeds {TOLERANCE:.2e}"


# ── Benchmark: Runtime comparison ────────────────────────────────────

def test_runtime_comparison():
    """Compare wall-clock time: sort_match vs hungarian_reference.

    Expected: sort_match ~100x faster for n=50.
    """
    n = 50
    b = 32
    pred = torch.from_numpy(RNG.uniform(5.0, 90.0, (b, n))).double()
    target = torch.from_numpy(RNG.uniform(5.0, 90.0, (b, n))).double()

    # Warm up
    sort_match_loss(pred, target, cost="mae")
    hungarian_reference(pred, target, cost="mae")

    # Time sort_match
    t0 = time.perf_counter()
    for _ in range(100):
        sort_match_loss(pred, target, cost="mae")
    t_sm = (time.perf_counter() - t0) / 100

    # Time hungarian
    t0 = time.perf_counter()
    for _ in range(100):
        hungarian_reference(pred, target, cost="mae")
    t_hu = (time.perf_counter() - t0) / 100

    speedup = t_hu / t_sm
    print(f"\n  sort_match: {t_sm*1000:.3f} ms, hungarian: {t_hu*1000:.3f} ms, "
          f"speedup: {speedup:.1f}x")
    assert speedup > 5, f"Expected >5x speedup, got {speedup:.1f}x"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
