"""
Set Transformer encoder for XRD peak sets.

Takes a variable-length set of peaks {(2theta, intensity, FWHM)} and
produces a fixed-dimensional representation. Uses multi-head attention
over set elements (no positional encoding — the set is unordered).

Architecture:
    1. Peak embedding: Linear(d_peak -> hidden)
    2. K self-attention blocks (Set Attention Blocks, SAB)
    3. Pooling via PMA (Pooling by Multi-head Attention) with m seed vectors
    4. Output: fixed-dim representation (hidden * m)

Reference: Lee et al., "Set Transformer: A Framework for Attention-based
Permutation-Invariant Input", ICML 2019.

Physics motivation:
    XRD peaks are discrete Bragg reflections — naturally an unordered set.
    Unlike spectrum-level approaches (CNN on binned intensity), the set
    representation respects the physics: peaks can appear in any order,
    and the number of peaks varies across patterns.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        B = Q.shape[0]
        # (B, n, d) -> (B, h, n, d_k)
        q = self.W_q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # mask: (B, n_k) -> (B, 1, 1, n_k)
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # (B, h, n_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.W_o(out)


class SAB(nn.Module):
    """Set Attention Block: self-attention + feedforward with residuals."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        H = self.ln1(X + self.mha(X, X, X, mask))
        return self.ln2(H + self.ff(H))


class PMA(nn.Module):
    """Pooling by Multi-head Attention.

    Uses m learnable seed vectors to aggregate set information into
    m fixed output vectors via cross-attention.
    """

    def __init__(self, d_model: int, n_heads: int, n_seeds: int, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        seeds = self.seeds.expand(X.shape[0], -1, -1)
        return self.ln(seeds + self.mha(seeds, X, X, mask))


class SetTransformerEncoder(nn.Module):
    """Full Set Transformer encoder for XRD peak sets.

    Input: (B, n_peaks, d_peak) with mask (B, n_peaks)
    Output: (B, d_repr) fixed-dimensional representation

    Args:
        d_peak: input peak feature dimension (e.g., 3 for 2theta, I, FWHM)
        d_model: hidden dimension
        n_heads: attention heads
        n_layers: number of SAB blocks
        n_seeds: number of PMA seed vectors (output = d_model * n_seeds)
        dropout: dropout rate
    """

    def __init__(
        self,
        d_peak: int = 3,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        n_seeds: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_seeds = n_seeds

        # Peak embedding
        self.peak_embed = nn.Sequential(
            nn.Linear(d_peak, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Self-attention blocks
        self.sab_blocks = nn.ModuleList([
            SAB(d_model, n_heads, dropout) for _ in range(n_layers)
        ])

        # Pooling
        self.pma = PMA(d_model, n_heads, n_seeds, dropout)

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model * n_seeds, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, peaks: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            peaks: (B, n_peaks, d_peak) peak features
            mask: (B, n_peaks) boolean mask, True = valid peak

        Returns:
            repr: (B, d_model) pattern representation
        """
        # Embed peaks
        x = self.peak_embed(peaks)  # (B, n, d_model)

        # Self-attention with masking
        for sab in self.sab_blocks:
            x = sab(x, mask)

        # Pool to fixed size
        pooled = self.pma(x, mask)  # (B, n_seeds, d_model)
        pooled = pooled.reshape(pooled.shape[0], -1)  # (B, n_seeds * d_model)

        # Project
        return self.output_proj(pooled)  # (B, d_model)

    @property
    def output_dim(self) -> int:
        return self.d_model

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
