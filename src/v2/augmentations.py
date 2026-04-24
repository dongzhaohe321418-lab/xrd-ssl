"""
Physics-informed augmentations for XRD peak sets.

Each augmentation simulates a real-world variation that occurs between
measurements of the same crystal phase:

- Gaussian noise on 2theta: instrument calibration drift, sample displacement
- Intensity jitter: preferred orientation, sample thickness variation
- Peak broadening: crystallite size variation (Scherrer effect)
- Peak dropout: weak peaks lost in noise or overlapping with neighbors
- Spurious peaks: detector artifacts, impurity peaks from sample holder
- Background shift: fluorescence, air scatter, amorphous content

Augmentations operate on peak sets (n, 3) with columns [2theta, I, FWHM].
"""

from __future__ import annotations

import torch
import numpy as np
from typing import List, Tuple


class PeakSetAugmenter:
    """Applies random augmentations to XRD peak sets for contrastive learning.

    Usage:
        aug = PeakSetAugmenter()
        view1, mask1 = aug(peaks, mask)
        view2, mask2 = aug(peaks, mask)
        # view1 and view2 are different augmented views of the same pattern
    """

    def __init__(
        self,
        theta_noise_std: float = 0.05,    # degrees, typical instrument error
        intensity_jitter: float = 0.15,   # relative, +/- 15%
        fwhm_jitter: float = 0.2,         # relative, +/- 20%
        dropout_prob: float = 0.15,        # probability of dropping each peak
        spurious_prob: float = 0.05,       # probability of adding a spurious peak
        max_spurious: int = 3,             # max spurious peaks to add
    ):
        self.theta_noise_std = theta_noise_std
        self.intensity_jitter = intensity_jitter
        self.fwhm_jitter = fwhm_jitter
        self.dropout_prob = dropout_prob
        self.spurious_prob = spurious_prob
        self.max_spurious = max_spurious

    def __call__(
        self,
        peaks: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentations to a peak set.

        Args:
            peaks: (n_max, 3) peak features [2theta, intensity, FWHM]
            mask: (n_max,) boolean mask

        Returns:
            aug_peaks: (n_max, 3) augmented peak features
            aug_mask: (n_max,) updated mask
        """
        aug_peaks = peaks.clone()
        aug_mask = mask.clone()
        n_valid = mask.sum().item()

        if n_valid == 0:
            return aug_peaks, aug_mask

        # 1. Gaussian noise on 2theta positions
        if self.theta_noise_std > 0:
            noise = torch.randn(aug_peaks.shape[0]) * self.theta_noise_std
            aug_peaks[:, 0] = aug_peaks[:, 0] + noise * aug_mask.float()
            aug_peaks[:, 0] = aug_peaks[:, 0].clamp(5.0, 90.0)

        # 2. Intensity jitter (multiplicative)
        if self.intensity_jitter > 0:
            jitter = 1.0 + (torch.rand(aug_peaks.shape[0]) * 2 - 1) * self.intensity_jitter
            aug_peaks[:, 1] = aug_peaks[:, 1] * jitter * aug_mask.float()
            aug_peaks[:, 1] = aug_peaks[:, 1].clamp(0.0, 200.0)

        # 3. FWHM jitter (if FWHM column exists and is nonzero)
        if aug_peaks.shape[1] > 2 and self.fwhm_jitter > 0:
            jitter = 1.0 + (torch.rand(aug_peaks.shape[0]) * 2 - 1) * self.fwhm_jitter
            aug_peaks[:, 2] = aug_peaks[:, 2] * jitter * aug_mask.float()
            aug_peaks[:, 2] = aug_peaks[:, 2].clamp(0.01, 5.0)

        # 4. Peak dropout
        if self.dropout_prob > 0 and n_valid > 3:
            # Don't drop below 3 peaks
            drop = torch.rand(aug_peaks.shape[0]) < self.dropout_prob
            drop = drop & aug_mask  # only drop valid peaks
            # Ensure at least 3 peaks remain
            n_drop = drop.sum().item()
            if n_valid - n_drop < 3:
                # Undo some drops
                drop_indices = drop.nonzero(as_tuple=True)[0]
                keep_count = n_drop - (n_valid - 3)
                drop[drop_indices[:keep_count]] = False
            aug_mask = aug_mask & ~drop

        # 5. Spurious peak insertion
        if self.spurious_prob > 0 and torch.rand(1).item() < self.spurious_prob:
            n_spurious = min(self.max_spurious, (aug_peaks.shape[0] - n_valid))
            if n_spurious > 0:
                n_add = torch.randint(1, n_spurious + 1, (1,)).item()
                # Find empty slots
                empty_slots = (~aug_mask).nonzero(as_tuple=True)[0]
                if len(empty_slots) >= n_add:
                    for j in range(n_add):
                        idx = empty_slots[j].item()
                        aug_peaks[idx, 0] = 5.0 + torch.rand(1).item() * 85.0  # random 2theta
                        aug_peaks[idx, 1] = torch.rand(1).item() * 20.0  # low intensity
                        if aug_peaks.shape[1] > 2:
                            aug_peaks[idx, 2] = 0.1 + torch.rand(1).item() * 0.5
                        aug_mask[idx] = True

        return aug_peaks, aug_mask


def augment_batch(
    peaks: torch.Tensor,
    masks: torch.Tensor,
    augmenter: PeakSetAugmenter,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply augmentations to a batch of peak sets.

    Args:
        peaks: (B, n_max, d_peak)
        masks: (B, n_max)

    Returns:
        aug_peaks: (B, n_max, d_peak)
        aug_masks: (B, n_max)
    """
    B = peaks.shape[0]
    aug_peaks = peaks.clone()
    aug_masks = masks.clone()
    for i in range(B):
        aug_peaks[i], aug_masks[i] = augmenter(peaks[i], masks[i])
    return aug_peaks, aug_masks
