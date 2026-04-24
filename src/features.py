"""
Crystal graph construction for XRD prediction.

Converts pymatgen Structure objects into graph representations suitable
for GNN processing. No torch_geometric dependency.

Physics:
    Atoms in a unit cell are nodes. Edges connect atoms within a distance
    cutoff (5 A), including periodic images. Edge features encode the
    interatomic distance via radial basis function (RBF) expansion.

    Atom features encode:
    - Element identity (one-hot, Z=1..100)
    - Pauling electronegativity
    - Covalent radius (A)
    - Group and period in periodic table (one-hot)
"""

from __future__ import annotations

import dataclasses
from typing import List, Tuple

import numpy as np
import torch


# ── Periodic table data ──────────────────────────────────────────────

# Pauling electronegativities for Z=1..100 (0 = unknown/not applicable)
_ELECTRONEG = np.array([
    0.00,  # placeholder for Z=0
    2.20, 0.00,  # H, He
    0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0.00,  # Li-Ne
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0.00,  # Na-Ar
    0.82, 1.00,  # K, Ca
    1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65,  # Sc-Zn
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00,  # Ga-Kr
    0.82, 0.95,  # Rb, Sr
    1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69,  # Y-Cd
    1.78, 1.96, 2.05, 2.10, 2.66, 2.60,  # In-Xe
    0.79, 0.89,  # Cs, Ba
    1.10, 1.12, 1.13, 1.14, 1.13, 1.17, 1.20, 1.20, 1.10, 1.22, 1.23, 1.24, 1.25, 1.10,  # La-Yb
    1.27, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00,  # Lu-Hg
    1.62, 2.33, 2.02, 2.00, 2.20, 0.00,  # Tl-Rn
    0.70, 0.90, 1.10, 1.30, 1.50, 1.38, 1.36, 1.28, 1.30, 1.30,  # Fr-Fm (approximate)
    1.30, 1.30,  # Md, No
], dtype=np.float32)

# Covalent radii in Angstroms for Z=1..100
_COVALENT_RADIUS = np.array([
    0.00,
    0.31, 0.28,  # H, He
    1.28, 0.96, 0.84, 0.76, 0.71, 0.66, 0.57, 0.58,  # Li-Ne
    1.66, 1.41, 1.21, 1.11, 1.07, 1.05, 1.02, 1.06,  # Na-Ar
    2.03, 1.76,  # K, Ca
    1.70, 1.60, 1.53, 1.39, 1.39, 1.32, 1.26, 1.24, 1.32, 1.22,  # Sc-Zn
    1.22, 1.20, 1.19, 1.20, 1.20, 1.16,  # Ga-Kr
    2.20, 1.95,  # Rb, Sr
    1.90, 1.75, 1.64, 1.54, 1.47, 1.46, 1.42, 1.39, 1.45, 1.44,  # Y-Cd
    1.42, 1.39, 1.39, 1.38, 1.39, 1.40,  # In-Xe
    2.44, 2.15,  # Cs, Ba
    2.07, 2.04, 2.03, 2.01, 1.99, 1.98, 1.98, 1.96, 1.94, 1.92, 1.92, 1.89, 1.90, 1.87,  # La-Yb
    1.87, 1.75, 1.70, 1.62, 1.51, 1.44, 1.41, 1.36, 1.36, 1.32,  # Lu-Hg
    1.45, 1.46, 1.48, 1.40, 1.50, 1.50,  # Tl-Rn
    2.60, 2.21, 2.15, 2.06, 2.00, 1.96, 1.90, 1.87, 1.80, 1.69,  # Fr-Fm
    1.68, 1.68,  # Md, No
], dtype=np.float32)

# Group (1-18) and Period (1-7) for Z=1..100
_GROUP = np.array([
    0,
    1, 18,
    1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    1, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # Lanthanides -> group 3
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,  # Lu-Rn (Lu=group 3 -> 4 here for Hf start)
    1, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # Actinides -> group 3
], dtype=np.int32)

_PERIOD = np.array([
    0,
    1, 1,
    2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
], dtype=np.int32)

MAX_ATOMIC_NUM = 100
NUM_GROUPS = 18
NUM_PERIODS = 7
CUTOFF = 5.0  # Angstroms
RBF_BINS = 8
RBF_MAX = 5.0


# ── Data structure ───────────────────────────────────────────────────

@dataclasses.dataclass
class CrystalGraph:
    """Graph representation of a crystal structure.

    Attributes:
        node_feats: (n_atoms, d_node) atom feature vectors.
        edge_index: (2, n_edges) source-target pairs.
        edge_feats: (n_edges, d_edge) RBF-expanded distances.
        n_atoms: number of atoms in this graph.
    """
    node_feats: torch.Tensor   # (n_atoms, d_node)
    edge_index: torch.Tensor   # (2, n_edges)
    edge_feats: torch.Tensor   # (n_edges, d_edge)
    n_atoms: int

    @property
    def d_node(self) -> int:
        return self.node_feats.shape[1]

    @property
    def d_edge(self) -> int:
        return self.edge_feats.shape[1]


def atom_features(atomic_number: int) -> np.ndarray:
    """Compute feature vector for a single atom.

    Features (total dim = 100 + 1 + 1 + 18 + 7 = 127):
        - One-hot atomic number (100 dims)
        - Pauling electronegativity (1 dim, normalized)
        - Covalent radius (1 dim, normalized)
        - One-hot group (18 dims)
        - One-hot period (7 dims)
    """
    z = min(atomic_number, MAX_ATOMIC_NUM)

    # One-hot atomic number
    z_onehot = np.zeros(MAX_ATOMIC_NUM, dtype=np.float32)
    z_onehot[z - 1] = 1.0

    # Electronegativity (normalized to ~[0,1])
    en = _ELECTRONEG[z] / 4.0 if z < len(_ELECTRONEG) else 0.0

    # Covalent radius (normalized to ~[0,1])
    cr = _COVALENT_RADIUS[z] / 3.0 if z < len(_COVALENT_RADIUS) else 0.0

    # One-hot group
    g = _GROUP[z] if z < len(_GROUP) else 0
    group_onehot = np.zeros(NUM_GROUPS, dtype=np.float32)
    if 1 <= g <= NUM_GROUPS:
        group_onehot[g - 1] = 1.0

    # One-hot period
    p = _PERIOD[z] if z < len(_PERIOD) else 0
    period_onehot = np.zeros(NUM_PERIODS, dtype=np.float32)
    if 1 <= p <= NUM_PERIODS:
        period_onehot[p - 1] = 1.0

    return np.concatenate([z_onehot, [en, cr], group_onehot, period_onehot])


def rbf_expansion(distances: np.ndarray, n_bins: int = RBF_BINS, max_dist: float = RBF_MAX) -> np.ndarray:
    """Radial basis function expansion of distances.

    Expands scalar distances into n_bins Gaussian basis functions
    evenly spaced from 0 to max_dist.

    Args:
        distances: (n,) array of distances in Angstroms.
        n_bins: number of Gaussian centers.
        max_dist: maximum distance for the last center.

    Returns:
        (n, n_bins) array of RBF features.
    """
    centers = np.linspace(0, max_dist, n_bins, dtype=np.float32)
    gamma = 1.0 / (max_dist / n_bins) ** 2  # width parameter
    # (n, 1) - (1, n_bins) -> (n, n_bins)
    return np.exp(-gamma * (distances[:, None] - centers[None, :]) ** 2).astype(np.float32)


def structure_to_graph(cif_path: str, cutoff: float = CUTOFF) -> CrystalGraph:
    """Convert a CIF file to a CrystalGraph.

    Uses pymatgen's get_all_neighbors to find edges within cutoff,
    including periodic images.

    Args:
        cif_path: path to CIF file.
        cutoff: distance cutoff in Angstroms.

    Returns:
        CrystalGraph with node features, edge index, and edge features.
    """
    from pymatgen.core import Structure

    structure = Structure.from_file(cif_path)
    n_atoms = len(structure)

    # Node features
    node_feat_list = []
    for site in structure:
        z = site.specie.Z
        node_feat_list.append(atom_features(z))
    node_feats = np.stack(node_feat_list)  # (n_atoms, d_node)

    # Edges via neighbor search (includes periodic images)
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)

    src_list = []
    dst_list = []
    dist_list = []

    for i, neighbors in enumerate(all_neighbors):
        for neighbor in neighbors:
            j = neighbor[2]  # index of neighbor atom
            d = neighbor[1]  # distance
            src_list.append(i)
            dst_list.append(j)
            dist_list.append(d)

    if len(src_list) == 0:
        # Fallback: self-loops only (isolated atoms -- shouldn't happen with real crystals)
        src_list = list(range(n_atoms))
        dst_list = list(range(n_atoms))
        dist_list = [0.0] * n_atoms

    edge_index = np.array([src_list, dst_list], dtype=np.int64)  # (2, n_edges)
    distances = np.array(dist_list, dtype=np.float32)             # (n_edges,)
    edge_feats = rbf_expansion(distances)                          # (n_edges, RBF_BINS)

    return CrystalGraph(
        node_feats=torch.from_numpy(node_feats),
        edge_index=torch.from_numpy(edge_index),
        edge_feats=torch.from_numpy(edge_feats),
        n_atoms=n_atoms,
    )


def collate_graphs(graphs: List[CrystalGraph]) -> Tuple[CrystalGraph, torch.Tensor]:
    """Batch multiple CrystalGraphs into a single graph (disjoint union).

    Returns:
        batched_graph: single CrystalGraph with concatenated features
            and shifted edge indices.
        batch_index: (total_atoms,) tensor mapping each atom to its
            graph index in the batch.
    """
    node_feats_list = []
    edge_index_list = []
    edge_feats_list = []
    batch_indices = []
    offset = 0

    for i, g in enumerate(graphs):
        node_feats_list.append(g.node_feats)
        edge_feats_list.append(g.edge_feats)

        # Shift edge indices by cumulative atom count
        shifted_edges = g.edge_index.clone()
        shifted_edges += offset
        edge_index_list.append(shifted_edges)

        batch_indices.append(torch.full((g.n_atoms,), i, dtype=torch.long))
        offset += g.n_atoms

    batched = CrystalGraph(
        node_feats=torch.cat(node_feats_list, dim=0),
        edge_index=torch.cat(edge_index_list, dim=1),
        edge_feats=torch.cat(edge_feats_list, dim=0),
        n_atoms=offset,
    )
    batch_index = torch.cat(batch_indices, dim=0)

    return batched, batch_index


# Feature dimensions for reference
NODE_FEAT_DIM = MAX_ATOMIC_NUM + 2 + NUM_GROUPS + NUM_PERIODS  # 127
EDGE_FEAT_DIM = RBF_BINS  # 8
