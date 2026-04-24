"""
Tests for Set Transformer encoder and contrastive SSL framework.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest
import torch
from src.v2.set_encoder import SetTransformerEncoder
from src.v2.augmentations import PeakSetAugmenter, augment_batch
from src.v2.contrastive_ssl import SortMatchSSL, PhaseClassifier


# ── Set Transformer tests ────────────────────────────────────────────

def test_encoder_output_shape():
    enc = SetTransformerEncoder(d_peak=3, d_model=64, n_heads=4, n_layers=2)
    peaks = torch.randn(4, 50, 3)
    mask = torch.ones(4, 50, dtype=torch.bool)
    out = enc(peaks, mask)
    assert out.shape == (4, 64)


def test_encoder_with_mask():
    enc = SetTransformerEncoder(d_peak=3, d_model=64, n_heads=4, n_layers=2)
    peaks = torch.randn(4, 50, 3)
    mask = torch.ones(4, 50, dtype=torch.bool)
    mask[:, 30:] = False  # last 20 are padding
    out = enc(peaks, mask)
    assert out.shape == (4, 64)
    assert not torch.isnan(out).any()


def test_encoder_permutation_invariant():
    """Output should be the same regardless of peak ordering."""
    enc = SetTransformerEncoder(d_peak=3, d_model=64, n_heads=4, n_layers=2)
    enc.eval()

    peaks = torch.randn(1, 20, 3)
    mask = torch.ones(1, 20, dtype=torch.bool)

    # Permute peaks
    perm = torch.randperm(20)
    peaks_perm = peaks[:, perm, :]

    with torch.no_grad():
        out1 = enc(peaks, mask)
        out2 = enc(peaks_perm, mask)

    # Should be very close (not exact due to numerical precision)
    assert torch.allclose(out1, out2, atol=1e-5), \
        f"Max diff: {(out1 - out2).abs().max():.6f}"


def test_encoder_gradient_flow():
    enc = SetTransformerEncoder(d_peak=3, d_model=64, n_heads=4, n_layers=2)
    peaks = torch.randn(4, 30, 3, requires_grad=True)
    mask = torch.ones(4, 30, dtype=torch.bool)
    out = enc(peaks, mask)
    loss = out.sum()
    loss.backward()
    assert peaks.grad is not None
    assert not torch.isnan(peaks.grad).any()


def test_encoder_param_count():
    enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
    n = enc.count_parameters()
    print(f"\n  Set Transformer: {n:,} params")
    assert 50_000 < n < 1_000_000  # reasonable range


# ── Augmentation tests ───────────────────────────────────────────────

def test_augmenter_preserves_shape():
    aug = PeakSetAugmenter()
    peaks = torch.randn(50, 3)
    peaks[:, 0] = torch.linspace(10, 80, 50)  # 2theta
    peaks[:, 1] = torch.rand(50) * 100  # intensity
    peaks[:, 2] = torch.rand(50) * 0.5 + 0.1  # FWHM
    mask = torch.ones(50, dtype=torch.bool)
    mask[40:] = False

    aug_peaks, aug_mask = aug(peaks, mask)
    assert aug_peaks.shape == peaks.shape
    assert aug_mask.shape == mask.shape


def test_augmenter_produces_different_views():
    aug = PeakSetAugmenter(theta_noise_std=0.1, dropout_prob=0.2)
    peaks = torch.randn(50, 3)
    peaks[:, 0] = torch.linspace(10, 80, 50)
    peaks[:, 1] = torch.rand(50) * 100
    peaks[:, 2] = torch.rand(50) * 0.5
    mask = torch.ones(50, dtype=torch.bool)

    v1, m1 = aug(peaks, mask)
    v2, m2 = aug(peaks, mask)
    # Views should differ (random augmentations)
    assert not torch.equal(v1, v2) or not torch.equal(m1, m2)


def test_augmenter_batch():
    aug = PeakSetAugmenter()
    peaks = torch.randn(8, 50, 3)
    masks = torch.ones(8, 50, dtype=torch.bool)
    aug_p, aug_m = augment_batch(peaks, masks, aug)
    assert aug_p.shape == peaks.shape
    assert aug_m.shape == masks.shape


# ── Contrastive SSL tests ────────────────────────────────────────────

def test_ssl_forward():
    model = SortMatchSSL(d_peak=3, d_model=64, d_proj=32, n_layers=2)
    peaks = torch.randn(8, 50, 3)
    peaks[:, :, 0] = peaks[:, :, 0].abs() * 85 + 5  # 2theta in [5, 90]
    masks = torch.ones(8, 50, dtype=torch.bool)

    result = model(peaks, masks)
    assert "loss" in result
    assert "info_nce" in result
    assert "sort_match" in result
    assert result["loss"].requires_grad
    assert not torch.isnan(result["loss"])


def test_ssl_backward():
    model = SortMatchSSL(d_peak=3, d_model=64, d_proj=32, n_layers=2)
    peaks = torch.randn(8, 50, 3)
    peaks[:, :, 0] = peaks[:, :, 0].abs() * 85 + 5
    masks = torch.ones(8, 50, dtype=torch.bool)

    result = model(peaks, masks)
    result["loss"].backward()

    # Check gradients exist
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_ssl_encode():
    model = SortMatchSSL(d_peak=3, d_model=64, d_proj=32, n_layers=2)
    peaks = torch.randn(4, 30, 3)
    masks = torch.ones(4, 30, dtype=torch.bool)

    repr = model.encode(peaks, masks)
    assert repr.shape == (4, 64)


def test_ssl_param_count():
    model = SortMatchSSL(d_peak=3, d_model=128, d_proj=64, n_layers=3)
    n = model.count_parameters()
    print(f"\n  SortMatchSSL: {n:,} params")
    assert n < 2_000_000


# ── Downstream classifier tests ─────────────────────────────────────

def test_phase_classifier():
    enc = SetTransformerEncoder(d_peak=3, d_model=64, n_heads=4, n_layers=2)
    clf = PhaseClassifier(enc, n_classes=10, freeze_encoder=True)

    peaks = torch.randn(4, 30, 3)
    masks = torch.ones(4, 30, dtype=torch.bool)

    logits = clf(peaks, masks)
    assert logits.shape == (4, 10)

    # Check encoder is frozen
    loss = logits.sum()
    loss.backward()
    for p in clf.encoder.parameters():
        assert p.grad is None or (p.grad == 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
