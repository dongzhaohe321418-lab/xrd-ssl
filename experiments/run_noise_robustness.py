"""
Noise robustness experiment: address Reviewer 5.1.

Tests sort-match SSL when unlabeled data has noise added to 2theta,
simulating experimental measurement error.

Conditions:
    (i)   Clean baseline (no noise)
    (ii)  Gaussian noise sigma = 0.1 degrees
    (iii) Gaussian noise sigma = 0.3 degrees
    (iv)  Gaussian noise sigma = 1.0 degrees
    (v)   10% spurious peaks inserted

Usage:
    python experiments/run_noise_robustness.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import masked_sort_match_loss
from src.model import build_model
from src.features import structure_to_graph, collate_graphs, CrystalGraph
from src.xrd_data import load_dataset


DATA_PATH = "data/xrd_cache.npz"
CIF_DIR = "data/mp_structures/cifs"
RESULTS_DIR = "experiments/results_noise"
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
LABELED_FRAC = 0.1
SEED = 42
DEVICE = "cpu"


def load_data():
    dataset = load_dataset(DATA_PATH)
    material_ids = list(dataset["material_ids"])
    two_theta = dataset["two_theta"]
    masks_np = dataset["mask"]

    graphs = []
    valid_indices = []
    for i, mp_id in enumerate(tqdm(material_ids, desc="Building graphs")):
        cif_path = os.path.join(CIF_DIR, f"{mp_id}.cif")
        if not os.path.exists(cif_path):
            continue
        try:
            g = structure_to_graph(cif_path)
            graphs.append(g)
            valid_indices.append(i)
        except Exception:
            pass

    valid_indices = np.array(valid_indices)
    targets = torch.from_numpy(two_theta[valid_indices]).float()
    masks = torch.from_numpy(masks_np[valid_indices]).bool()

    for i in range(len(targets)):
        m = masks[i]
        n_valid = m.sum().item()
        if n_valid > 0:
            targets[i, :n_valid] = targets[i, :n_valid].sort().values

    return graphs, targets, masks


def corrupt_targets(targets, masks, noise_sigma, spurious_frac, rng):
    """Add noise to unlabeled targets (simulating experimental error)."""
    corrupted = targets.clone()

    if noise_sigma > 0:
        noise = torch.from_numpy(
            rng.normal(0, noise_sigma, targets.shape).astype(np.float32)
        )
        # Only add noise where mask is True
        corrupted = corrupted + noise * masks.float()
        # Clamp to valid range
        corrupted = corrupted.clamp(5.0, 90.0)

    if spurious_frac > 0:
        # For each sample, replace some valid peaks with random positions
        for i in range(len(corrupted)):
            m = masks[i]
            n_valid = m.sum().item()
            n_spurious = max(1, int(n_valid * spurious_frac))
            # Replace last n_spurious valid peaks with random values
            if n_valid > n_spurious:
                spurious_vals = torch.from_numpy(
                    rng.uniform(5.0, 90.0, n_spurious).astype(np.float32)
                )
                corrupted[i, n_valid - n_spurious:n_valid] = spurious_vals

    return corrupted


def make_batch(graphs, targets, masks, batch_idx, device):
    batch_graphs = [graphs[i] for i in batch_idx]
    batch_targets = targets[batch_idx].to(device)
    batch_masks = masks[batch_idx].to(device)
    batched_graph, batch_index = collate_graphs(batch_graphs)
    batched_graph = CrystalGraph(
        node_feats=batched_graph.node_feats.to(device),
        edge_index=batched_graph.edge_index.to(device),
        edge_feats=batched_graph.edge_feats.to(device),
        n_atoms=batched_graph.n_atoms,
    )
    return batched_graph, batch_index.to(device), batch_targets, batch_masks


def train_and_eval(graphs, clean_targets, noisy_targets, masks, seed):
    """Train sort_match_ssl with noisy unlabeled data, evaluate on clean data."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(graphs)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = min(200, n // 5)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    n_labeled = max(1, int(len(train_idx) * LABELED_FRAC))
    labeled_set = set(train_idx[:n_labeled].tolist())

    model = build_model(hidden_dim=128, n_layers=4, n_peaks=50)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(N_EPOCHS):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        ep_perm = rng.permutation(len(train_idx))

        for start in range(0, len(ep_perm), BATCH_SIZE):
            batch_perm = ep_perm[start:start + BATCH_SIZE]
            batch_idx = train_idx[batch_perm]

            batch_graphs = [graphs[i] for i in batch_idx]
            batched_graph, batch_index = collate_graphs(batch_graphs)
            batched_graph = CrystalGraph(
                node_feats=batched_graph.node_feats.to(DEVICE),
                edge_index=batched_graph.edge_index.to(DEVICE),
                edge_feats=batched_graph.edge_feats.to(DEVICE),
                n_atoms=batched_graph.n_atoms,
            )
            batch_index = batch_index.to(DEVICE)
            batch_masks = masks[batch_idx].to(DEVICE)

            pred = model(batched_graph, batch_index)

            is_labeled = torch.tensor([idx in labeled_set for idx in batch_idx.tolist()], dtype=torch.bool)
            n_l = is_labeled.sum().item()
            n_u = (~is_labeled).sum().item()

            # Labeled: use CLEAN targets
            l_loss = torch.tensor(0.0, device=DEVICE)
            if n_l > 0:
                l_targets = clean_targets[batch_idx[is_labeled]].to(DEVICE)
                l_loss = masked_sort_match_loss(pred[is_labeled], l_targets, batch_masks[is_labeled], cost="mae")

            # Unlabeled: use NOISY targets (simulating experimental data)
            u_loss = torch.tensor(0.0, device=DEVICE)
            if n_u > 0:
                u_targets = noisy_targets[batch_idx[~is_labeled]].to(DEVICE)
                u_loss = masked_sort_match_loss(pred[~is_labeled], u_targets, batch_masks[~is_labeled], cost="mae")

            loss = (n_l * l_loss + n_u * u_loss) / max(n_l + n_u, 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate on CLEAN test data
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(test_idx), BATCH_SIZE):
            bidx = test_idx[start:start + BATCH_SIZE]
            bg, bi, bt, bm = make_batch(graphs, clean_targets, masks, bidx, DEVICE)
            pred = model(bg, bi)
            for i in range(len(bidx)):
                m = bm[i]
                nv = m.sum().item()
                if nv == 0:
                    continue
                p = pred[i, :nv].sort().values
                t = bt[i, :nv].sort().values
                errors.append((p - t).abs().mean().item())
    return np.mean(errors)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    graphs, clean_targets, masks = load_data()
    rng = np.random.default_rng(SEED)

    conditions = {
        "clean": {"sigma": 0.0, "spurious": 0.0},
        "noise_0.1": {"sigma": 0.1, "spurious": 0.0},
        "noise_0.3": {"sigma": 0.3, "spurious": 0.0},
        "noise_1.0": {"sigma": 1.0, "spurious": 0.0},
        "spurious_10pct": {"sigma": 0.0, "spurious": 0.1},
        "combined": {"sigma": 0.3, "spurious": 0.1},
    }

    results = {}
    for name, params in conditions.items():
        print(f"\n  Condition: {name} (sigma={params['sigma']}, spurious={params['spurious']})")
        noisy = corrupt_targets(clean_targets, masks, params["sigma"], params["spurious"], rng)
        mae = train_and_eval(graphs, clean_targets, noisy, masks, SEED)
        results[name] = {"mae": mae, **params}
        print(f"    MAE = {mae:.2f} deg")

    print(f"\n{'='*60}")
    print(f"  NOISE ROBUSTNESS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Condition':<20} {'Sigma':>8} {'Spurious':>10} {'MAE':>8}")
    print(f"{'-'*46}")
    for name, r in results.items():
        print(f"{name:<20} {r['sigma']:>7.1f}° {r['spurious']:>9.0%} {r['mae']:>7.2f}°")

    with open(os.path.join(RESULTS_DIR, "noise_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/noise_results.json")


if __name__ == "__main__":
    main()
