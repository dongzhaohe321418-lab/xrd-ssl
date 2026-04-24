"""
Sort-match loss for XRD peak prediction.

Theorem (1D sort-match optimality):
    For n real-valued predictions {p_i} and n targets {t_j}, the assignment
    that minimizes sum_i c(p_sigma(i), t_i) for any convex cost c is the
    one that matches the sorted sequences: sigma = argsort(p) composed with
    argsort(t)^{-1}. In other words, sorting both sequences and pairing
    them element-wise yields the optimal transport plan.

    This reduces O(n^3) Hungarian matching to O(n log n) sorting.

    Reference: nmr-ssl (arXiv 2601.18524), Theorem 1.

Physics context:
    XRD peaks are a set of (2theta, intensity) pairs. In Stage 1, we predict
    only 2theta positions. The sort-match loss lets us train with unassigned
    peak lists -- no need to know which predicted peak corresponds to which
    observed peak.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Literal, Optional


def sort_match_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cost: Literal["mse", "mae", "huber"] = "mae",
) -> torch.Tensor:
    """1D sort-match loss for unmasked fixed-length peak sets.

    Sorts predictions and targets independently, then computes element-wise
    cost. For convex costs, this is provably optimal (equivalent to Hungarian).

    Args:
        pred: (B, n) predicted 2theta values in degrees.
        target: (B, n) ground-truth 2theta values in degrees.
        cost: cost function -- "mse", "mae", or "huber".

    Returns:
        Scalar loss averaged over batch and peaks.
    """
    pred_sorted = torch.sort(pred, dim=-1).values
    target_sorted = torch.sort(target, dim=-1).values

    if cost == "mse":
        return F.mse_loss(pred_sorted, target_sorted)
    elif cost == "mae":
        return F.l1_loss(pred_sorted, target_sorted)
    elif cost == "huber":
        return F.smooth_l1_loss(pred_sorted, target_sorted)
    else:
        raise ValueError(f"Unknown cost: {cost}")


def masked_sort_match_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    cost: Literal["mse", "mae", "huber"] = "mae",
) -> torch.Tensor:
    """1D sort-match loss with masking for variable-length peak sets.

    Handles the case where different structures have different numbers of
    peaks. Padded positions (mask=False) must not contribute to the loss.

    Strategy:
        For each sample in the batch:
        1. Extract valid predictions and targets using the mask.
        2. Both should have the same count (ensured by data pipeline).
        3. Sort each independently.
        4. Compute element-wise cost on the sorted sequences.
        We cannot simply set padding to a large sentinel and sort, because
        that would create spurious matches between real peaks and sentinels.
        Instead, we process each sample individually.

    Args:
        pred: (B, n) predicted 2theta values.
        target: (B, n) ground-truth 2theta values.
        mask: (B, n) boolean mask. True = valid peak, False = padding.
        cost: cost function.

    Returns:
        Scalar loss averaged over all valid peaks across the batch.
    """
    batch_size = pred.shape[0]
    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    total_count = 0

    for i in range(batch_size):
        m = mask[i]  # (n,) bool
        n_valid = m.sum().item()
        if n_valid == 0:
            continue

        p = torch.sort(pred[i, m]).values      # (n_valid,)
        t = torch.sort(target[i, m]).values    # (n_valid,)

        if cost == "mse":
            sample_loss = F.mse_loss(p, t, reduction="sum")
        elif cost == "mae":
            sample_loss = F.l1_loss(p, t, reduction="sum")
        elif cost == "huber":
            sample_loss = F.smooth_l1_loss(p, t, reduction="sum")
        else:
            raise ValueError(f"Unknown cost: {cost}")

        total_loss = total_loss + sample_loss
        total_count += n_valid

    if total_count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return total_loss / total_count


def hungarian_reference(
    pred: torch.Tensor,
    target: torch.Tensor,
    cost: Literal["mse", "mae", "huber"] = "mae",
    mask: Optional[torch.Tensor] = None,
) -> float:
    """Reference implementation using scipy Hungarian matching.

    O(n^3) per sample. Used only for theorem verification, never for training.

    Args:
        pred: (B, n) predicted values.
        target: (B, n) target values.
        cost: cost function.
        mask: optional (B, n) boolean mask.

    Returns:
        Average cost (float, detached).
    """
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
    else:
        mask_np = np.ones_like(pred_np, dtype=bool)

    total_cost = 0.0
    total_count = 0

    for i in range(pred_np.shape[0]):
        m = mask_np[i]
        p = pred_np[i, m]
        t = target_np[i, m]
        n = len(p)
        if n == 0:
            continue

        # Build cost matrix C[j, k] = cost(p[j], t[k])
        diff = p[:, None] - t[None, :]  # (n, n)
        if cost == "mse":
            C = diff ** 2
        elif cost == "mae":
            C = np.abs(diff)
        elif cost == "huber":
            # Smooth L1: 0.5*x^2 if |x|<1, |x|-0.5 otherwise
            abs_diff = np.abs(diff)
            C = np.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
        else:
            raise ValueError(f"Unknown cost: {cost}")

        row_ind, col_ind = linear_sum_assignment(C)
        total_cost += C[row_ind, col_ind].sum()
        total_count += n

    if total_count == 0:
        return 0.0
    return total_cost / total_count
