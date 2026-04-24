"""
Sort-match contrastive SSL for XRD peak sets.

Core idea: two augmented views of the same XRD pattern should produce
similar representations, while views from different patterns should differ.

Unlike standard contrastive learning (SimCLR, Barlow Twins) that operates
on full spectra, we operate on PEAK SETS — unordered collections of
(2theta, I, FWHM) tuples. The sort-match loss provides the natural
alignment between augmented views' peak sets.

Framework:
    1. For each pattern, create two augmented views (different noise,
       dropout, etc.)
    2. Encode both views with the Set Transformer
    3. Project into contrastive space
    4. Loss = InfoNCE between positive pairs + sort-match regularizer
       on the peak-level alignment

The sort-match regularizer ensures the encoder learns representations
that respect peak-level correspondence, not just global pattern similarity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.v2.set_encoder import SetTransformerEncoder
from src.v2.augmentations import PeakSetAugmenter, augment_batch
from src.losses import masked_sort_match_loss


class ProjectionHead(nn.Module):
    """MLP projection head for contrastive learning."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SortMatchSSL(nn.Module):
    """Sort-match contrastive self-supervised learning for XRD.

    Combines:
    1. InfoNCE loss on pattern representations (global similarity)
    2. Sort-match loss on peak sets (peak-level alignment)

    The sort-match component is what differentiates this from SimCLR:
    it forces the encoder to learn representations that capture
    individual peak positions, not just overall pattern shape.

    Args:
        d_peak: peak feature dimension (3 for 2theta, I, FWHM)
        d_model: Set Transformer hidden dimension
        d_proj: projection head output dimension
        n_heads: attention heads in Set Transformer
        n_layers: SAB layers in Set Transformer
        temperature: InfoNCE temperature
        sort_match_weight: weight of sort-match regularizer
    """

    def __init__(
        self,
        d_peak: int = 3,
        d_model: int = 128,
        d_proj: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        temperature: float = 0.1,
        sort_match_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature = temperature
        self.sort_match_weight = sort_match_weight

        self.encoder = SetTransformerEncoder(
            d_peak=d_peak, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        )
        self.projector = ProjectionHead(d_model, d_model, d_proj)
        self.augmenter = PeakSetAugmenter()

        # Peak predictor: from representation, predict sorted peak positions
        # This enables the sort-match regularizer
        self.peak_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 100),  # predict up to 100 peak positions
        )

    def info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """InfoNCE contrastive loss.

        Positive pairs: (z1[i], z2[i]) — two views of the same pattern.
        Negative pairs: (z1[i], z2[j]) for j != i.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        B = z1.shape[0]
        # Similarity matrix (B, B)
        sim = torch.matmul(z1, z2.T) / self.temperature

        # Labels: positive pairs are on the diagonal
        labels = torch.arange(B, device=z1.device)

        # Cross-entropy in both directions
        loss_12 = F.cross_entropy(sim, labels)
        loss_21 = F.cross_entropy(sim.T, labels)

        return (loss_12 + loss_21) / 2

    def forward(
        self,
        peaks: torch.Tensor,
        masks: torch.Tensor,
    ) -> dict:
        """Forward pass for SSL pre-training.

        Args:
            peaks: (B, n_max, d_peak) raw peak sets
            masks: (B, n_max) boolean masks

        Returns:
            dict with 'loss', 'info_nce', 'sort_match', 'repr'
        """
        # Create two augmented views
        view1, mask1 = augment_batch(peaks, masks, self.augmenter)
        view2, mask2 = augment_batch(peaks, masks, self.augmenter)

        # Encode both views
        repr1 = self.encoder(view1, mask1)  # (B, d_model)
        repr2 = self.encoder(view2, mask2)  # (B, d_model)

        # Project for contrastive loss
        proj1 = self.projector(repr1)  # (B, d_proj)
        proj2 = self.projector(repr2)  # (B, d_proj)

        # InfoNCE loss
        nce_loss = self.info_nce_loss(proj1, proj2)

        # Sort-match regularizer: predict peak positions from representation
        # and match against the original peaks
        pred_peaks1 = self.peak_predictor(repr1)  # (B, 100)
        pred_peaks2 = self.peak_predictor(repr2)  # (B, 100)

        # Get 2theta positions from original peaks
        target1 = view1[:, :, 0]  # (B, n_max)
        target2 = view2[:, :, 0]

        # Pad predictions to match target size, or truncate
        n_max = target1.shape[1]
        if pred_peaks1.shape[1] > n_max:
            pred_peaks1 = pred_peaks1[:, :n_max]
            pred_peaks2 = pred_peaks2[:, :n_max]
        elif pred_peaks1.shape[1] < n_max:
            pad = torch.zeros(pred_peaks1.shape[0], n_max - pred_peaks1.shape[1],
                            device=pred_peaks1.device)
            pred_peaks1 = torch.cat([pred_peaks1, pad], dim=1)
            pred_peaks2 = torch.cat([pred_peaks2, pad], dim=1)

        sm_loss1 = masked_sort_match_loss(pred_peaks1, target1, mask1, cost="mae")
        sm_loss2 = masked_sort_match_loss(pred_peaks2, target2, mask2, cost="mae")
        sm_loss = (sm_loss1 + sm_loss2) / 2

        # Total loss
        total_loss = nce_loss + self.sort_match_weight * sm_loss

        return {
            "loss": total_loss,
            "info_nce": nce_loss.item(),
            "sort_match": sm_loss.item(),
            "repr": repr1.detach(),
        }

    def encode(self, peaks: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Encode peak sets into representations (no augmentation)."""
        return self.encoder(peaks, masks)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhaseClassifier(nn.Module):
    """Downstream classifier: frozen encoder + linear head for phase ID."""

    def __init__(self, encoder: SetTransformerEncoder, n_classes: int,
                 freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(encoder.output_dim, n_classes),
        )
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, peaks: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        repr = self.encoder(peaks, masks)
        return self.head(repr)

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
