"""
2D Sliced-Wasserstein loss for XRD peaks with positions AND intensities.

XRD peaks are 2D: (2theta, intensity). The 1D sort-match theorem guarantees
optimal matching for scalar values, but no such exact result exists for
2D (see docs/theorem_2d.md for the counterexample).

The sliced-Wasserstein distance projects 2D points onto random 1D
directions, applies the exact 1D sort-match on each projection, and
averages over directions. This is a principled approximation to the
true 2D Wasserstein distance.

Two direction sampling strategies:
    1. 'uniform' — directions sampled uniformly on S^1
    2. 'xrd_biased' — directions sampled from von Mises distribution
       centered on the 2theta axis, reflecting the physical fact that
       peak positions carry more information than intensities.

Physics rationale for biased directions:
    In XRD, peak positions are determined exactly by the lattice
    (Bragg's law), while intensities depend on atomic scattering factors,
    thermal motion, texture, and preferred orientation. Errors in
    intensity are physically larger and less diagnostic. Therefore,
    matching along the 2theta axis is more important.
"""

from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


def _sample_directions(
    n_slices: int,
    prior: Literal["uniform", "xrd_biased"] = "uniform",
    kappa: float = 2.0,
    device: torch.device = torch.device("cpu"),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample n_slices unit directions on S^1.

    Args:
        n_slices: number of projection directions.
        prior: 'uniform' samples theta ~ Uniform(0, pi),
               'xrd_biased' samples theta ~ von Mises(0, kappa)
               centered on the 2theta axis (angle 0).
        kappa: concentration parameter for von Mises (only used if
               prior='xrd_biased'). Higher kappa = more concentrated
               around the 2theta axis.
        device: torch device.
        generator: optional torch.Generator for reproducibility.

    Returns:
        directions: (n_slices, 2) unit vectors.
    """
    if prior == "uniform":
        # Uniform on [0, pi) — half circle suffices since direction and
        # its negation give the same sorted order
        angles = torch.linspace(0, math.pi, n_slices + 1, device=device)[:-1]
    elif prior == "xrd_biased":
        # von Mises centered at angle=0 (2theta axis)
        # Sample angles from von Mises(mu=0, kappa=kappa)
        # Use numpy for von Mises sampling, then convert
        rng = np.random.default_rng(42 if generator is None else None)
        angles_np = rng.vonmises(mu=0.0, kappa=kappa, size=n_slices)
        # Map to [0, pi) — fold negative angles
        angles_np = np.abs(angles_np) % math.pi
        angles = torch.from_numpy(angles_np).float().to(device)
    else:
        raise ValueError(f"Unknown prior: {prior}")

    # Convert to unit vectors
    directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return directions  # (n_slices, 2)


def sliced_wasserstein_loss(
    pred_2d: torch.Tensor,
    obs_2d: torch.Tensor,
    n_slices: int = 128,
    direction_prior: Literal["uniform", "xrd_biased"] = "uniform",
    kappa: float = 2.0,
    mask: Optional[torch.Tensor] = None,
    two_theta_range: tuple = (5.0, 90.0),
) -> torch.Tensor:
    """2D sliced-Wasserstein loss for (2theta, intensity) peak sets.

    Normalizes both dimensions to [0, 1] before projection.

    Args:
        pred_2d: (B, n, 2) predicted peaks [2theta, intensity].
        obs_2d: (B, n, 2) observed peaks [2theta, intensity].
        n_slices: number of random projection directions.
        direction_prior: 'uniform' or 'xrd_biased'.
        kappa: von Mises concentration (xrd_biased only).
        mask: (B, n) boolean mask. True = valid peak.
        two_theta_range: (min, max) for 2theta normalization.

    Returns:
        Scalar loss averaged over batch, slices, and valid peaks.
    """
    B, n, d = pred_2d.shape
    assert d == 2, f"Expected 2D input, got {d}D"
    device = pred_2d.device

    # Normalize to [0, 1]
    # Dim 0: 2theta -> [0, 1] over two_theta_range
    # Dim 1: intensity -> [0, 1] (already in [0, 100], divide by 100)
    pred_norm = pred_2d.clone()
    obs_norm = obs_2d.clone()

    theta_min, theta_max = two_theta_range
    pred_norm[:, :, 0] = (pred_2d[:, :, 0] - theta_min) / (theta_max - theta_min)
    pred_norm[:, :, 1] = pred_2d[:, :, 1] / 100.0
    obs_norm[:, :, 0] = (obs_2d[:, :, 0] - theta_min) / (theta_max - theta_min)
    obs_norm[:, :, 1] = obs_2d[:, :, 1] / 100.0

    # Sample projection directions
    directions = _sample_directions(n_slices, direction_prior, kappa, device)  # (n_slices, 2)

    # Project: (B, n, 2) @ (2, n_slices) -> (B, n, n_slices)
    pred_proj = torch.matmul(pred_norm, directions.T)   # (B, n, n_slices)
    obs_proj = torch.matmul(obs_norm, directions.T)      # (B, n, n_slices)

    if mask is None:
        # No masking: sort and match across all peaks
        pred_sorted = torch.sort(pred_proj, dim=1).values  # (B, n, n_slices)
        obs_sorted = torch.sort(obs_proj, dim=1).values
        loss = F.l1_loss(pred_sorted, obs_sorted)
    else:
        # Masked: process each sample individually
        total_loss = torch.tensor(0.0, device=device, dtype=pred_2d.dtype)
        total_count = 0

        for i in range(B):
            m = mask[i]
            n_valid = m.sum().item()
            if n_valid == 0:
                continue

            p = pred_proj[i, m, :]    # (n_valid, n_slices)
            o = obs_proj[i, m, :]     # (n_valid, n_slices)

            p_sorted = torch.sort(p, dim=0).values
            o_sorted = torch.sort(o, dim=0).values

            sample_loss = F.l1_loss(p_sorted, o_sorted, reduction="sum")
            total_loss = total_loss + sample_loss
            total_count += n_valid * n_slices

        if total_count == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        loss = total_loss / total_count

    return loss


def hungarian_2d_reference(
    pred_2d: torch.Tensor,
    obs_2d: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    two_theta_range: tuple = (5.0, 90.0),
) -> float:
    """Exact 2D Hungarian matching for verification.

    Uses L2 cost on normalized (2theta, intensity) pairs.

    Returns average cost per valid peak.
    """
    pred_np = pred_2d.detach().cpu().numpy()
    obs_np = obs_2d.detach().cpu().numpy()
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.ones(pred_np.shape[:2], dtype=bool)

    theta_min, theta_max = two_theta_range

    total_cost = 0.0
    total_count = 0

    for i in range(pred_np.shape[0]):
        m = mask_np[i]
        p = pred_np[i, m].copy()
        o = obs_np[i, m].copy()
        n = len(p)
        if n == 0:
            continue

        # Normalize
        p[:, 0] = (p[:, 0] - theta_min) / (theta_max - theta_min)
        p[:, 1] = p[:, 1] / 100.0
        o[:, 0] = (o[:, 0] - theta_min) / (theta_max - theta_min)
        o[:, 1] = o[:, 1] / 100.0

        # L1 cost matrix
        C = np.sum(np.abs(p[:, None, :] - o[None, :, :]), axis=2)  # (n, n)

        row_ind, col_ind = linear_sum_assignment(C)
        total_cost += C[row_ind, col_ind].sum()
        total_count += n

    return total_cost / total_count if total_count > 0 else 0.0
