"""
4-layer Graph Isomorphism Network (GIN) for XRD peak prediction.

Architecture:
    1. Node embedding: Linear(d_node -> hidden)
    2. Edge embedding: Linear(d_edge -> hidden)
    3. 4 GIN layers with edge-conditioned message passing
    4. Global pooling: mean || sum (concatenated)
    5. Output MLP: hidden*2 -> hidden -> n_peaks (predicted 2theta)

No torch_geometric dependency. Message passing implemented manually.

Target: <500k parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.features import CrystalGraph, NODE_FEAT_DIM, EDGE_FEAT_DIM


class GINLayer(nn.Module):
    """Graph Isomorphism Network layer with edge features.

    Message passing:
        m_j = MLP_edge(edge_feat_ij) * h_j
        h_i' = MLP_node((1 + eps) * h_i + sum_j m_j)

    This is a simplified GIN that incorporates edge information via
    element-wise multiplication, following the nmr-ssl pattern.
    """

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        x: torch.Tensor,          # (N, hidden)
        edge_index: torch.Tensor,  # (2, E)
        edge_feats: torch.Tensor,  # (E, edge_dim)
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        n_nodes = x.shape[0]

        # Edge-conditioned messages
        edge_weights = self.edge_mlp(edge_feats)   # (E, hidden)
        messages = x[src] * edge_weights            # (E, hidden)

        # Aggregate messages to destination nodes
        agg = torch.zeros_like(x)                   # (N, hidden)
        agg.index_add_(0, dst, messages)

        # Update
        out = self.node_mlp((1 + self.eps) * x + agg)
        return out


class XRDPredictor(nn.Module):
    """GIN-based model predicting XRD peak positions from crystal graphs.

    Architecture:
        - Node/edge embedding layers
        - 4 GIN message-passing layers
        - Global pooling (mean || sum)
        - MLP output head producing n_peaks 2theta values

    Args:
        hidden_dim: hidden dimension for GIN layers.
        n_layers: number of GIN layers.
        n_peaks: number of output peaks (fixed at 50).
        dropout: dropout rate in output head.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_peaks: int = 50,
        dropout: float = 0.1,
        node_feat_dim: int = NODE_FEAT_DIM,
        edge_feat_dim: int = EDGE_FEAT_DIM,
    ):
        super().__init__()
        self.n_peaks = n_peaks

        # Input embedding
        self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_embed = nn.Linear(edge_feat_dim, hidden_dim)

        # GIN layers
        self.gin_layers = nn.ModuleList([
            GINLayer(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])

        # Output head: mean||sum -> MLP -> 2theta values
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_peaks),
        )

        # Initialize output bias to spread across [5, 90] range
        with torch.no_grad():
            bias = torch.linspace(10.0, 80.0, n_peaks)
            self.output_head[-1].bias.copy_(bias)

    def forward(
        self,
        graph: CrystalGraph,
        batch_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            graph: batched CrystalGraph.
            batch_index: (N,) tensor mapping atoms to graphs.

        Returns:
            pred_2theta: (B, n_peaks) predicted peak positions in degrees.
        """
        x = self.node_embed(graph.node_feats)        # (N, hidden)
        edge_feats = self.edge_embed(graph.edge_feats)  # (E, hidden)

        # Message passing
        for gin in self.gin_layers:
            x = x + gin(x, graph.edge_index, edge_feats)  # residual

        # Global pooling: mean || sum
        batch_size = batch_index.max().item() + 1
        pool_mean = torch.zeros(batch_size, x.shape[1], device=x.device)
        pool_sum = torch.zeros(batch_size, x.shape[1], device=x.device)

        pool_mean.index_add_(0, batch_index, x)
        pool_sum.index_add_(0, batch_index, x)

        # Compute counts for mean
        counts = torch.zeros(batch_size, 1, device=x.device)
        counts.index_add_(0, batch_index, torch.ones(x.shape[0], 1, device=x.device))
        pool_mean = pool_mean / counts.clamp(min=1)

        # Concatenate mean and sum
        graph_repr = torch.cat([pool_mean, pool_sum], dim=1)  # (B, hidden*2)

        # Predict peaks
        pred_2theta = self.output_head(graph_repr)  # (B, n_peaks)

        # Clamp to physical range [5, 90] degrees
        pred_2theta = pred_2theta.clamp(5.0, 90.0)

        return pred_2theta

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    hidden_dim: int = 128,
    n_layers: int = 4,
    n_peaks: int = 50,
) -> XRDPredictor:
    """Build the XRD predictor model.

    Default config targets <500k parameters.
    """
    model = XRDPredictor(hidden_dim=hidden_dim, n_layers=n_layers, n_peaks=n_peaks)
    n_params = model.count_parameters()
    print(f"Model: {n_params:,} parameters (target <500k)")
    return model
