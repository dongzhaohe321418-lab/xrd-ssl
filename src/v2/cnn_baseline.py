"""
1D CNN baseline operating on binned XRD patterns (not peak sets).

This is the mainstream approach: treat the XRD pattern as a 1D signal
of length N_BINS, apply 1D convolutions. No peak extraction needed.

Purpose: demonstrate that the Set Transformer's peak-set representation
provides an advantage over the naive spectrum-level approach.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


N_BINS = 1700  # 2theta in [5, 90] at 0.05 degree resolution


class CNN1DEncoder(nn.Module):
    """1D CNN for XRD pattern encoding.

    Input: (B, 1, N_BINS) binned intensity pattern
    Output: (B, d_model) representation
    """

    def __init__(self, d_model: int = 128, n_bins: int = N_BINS):
        super().__init__()
        self.d_model = d_model
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, d_model)

    @property
    def output_dim(self):
        return self.d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, N_BINS) or (B, N_BINS)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv(x).squeeze(-1)  # (B, 128)
        return self.fc(h)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN1DClassifier(nn.Module):
    """CNN encoder + classification head."""

    def __init__(self, n_classes: int, d_model: int = 128):
        super().__init__()
        self.encoder = CNN1DEncoder(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def peaks_to_binned_pattern(
    peaks: np.ndarray,
    mask: np.ndarray,
    n_bins: int = N_BINS,
    theta_range: tuple = (5.0, 90.0),
    peak_width: float = 0.15,
) -> np.ndarray:
    """Convert peak set to binned 1D pattern via Gaussian broadening.

    Args:
        peaks: (n_max, 3) peak features [2theta, intensity, FWHM]
        mask: (n_max,) boolean mask
        n_bins: number of output bins
        theta_range: (min, max) 2theta range
        peak_width: Gaussian sigma in degrees (used if FWHM is 0)

    Returns:
        pattern: (n_bins,) binned intensity array
    """
    theta_min, theta_max = theta_range
    bin_edges = np.linspace(theta_min, theta_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pattern = np.zeros(n_bins, dtype=np.float32)

    for i in range(len(peaks)):
        if not mask[i]:
            continue
        pos = peaks[i, 0]
        intensity = peaks[i, 1]
        fwhm = peaks[i, 2] if peaks.shape[1] > 2 and peaks[i, 2] > 0 else peak_width * 2.355
        sigma = max(fwhm / 2.355, 0.02)

        # Add Gaussian peak
        gauss = intensity * np.exp(-0.5 * ((bin_centers - pos) / sigma) ** 2)
        pattern += gauss

    # Normalize
    if pattern.max() > 0:
        pattern = pattern / pattern.max() * 100.0

    return pattern


def batch_peaks_to_patterns(peaks: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Convert batch of peak sets to binned patterns."""
    B = peaks.shape[0]
    patterns = np.zeros((B, N_BINS), dtype=np.float32)
    for i in range(B):
        patterns[i] = peaks_to_binned_pattern(peaks[i], masks[i])
    return patterns
