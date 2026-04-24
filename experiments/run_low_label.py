"""
Session 3: Low-label ablation — the most important experiment.

labeled_fraction in {0.02, 0.05, 0.1, 0.2, 0.5}
Methods: supervised, sort_match_ssl
2 seeds each

Expected: SSL advantage grows as labeled_fraction shrinks.

Usage:
    python experiments/run_low_label.py [--data_path data/xrd_cache_5k.npz]
"""

from __future__ import annotations

import argparse
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


RESULTS_DIR = "experiments/results_session3"
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
SEEDS = [42, 2026]
DEVICE = "cpu"
LABEL_FRACS = [0.02, 0.05, 0.1, 0.2, 0.5]


def load_data(data_path, cif_dir):
    dataset = load_dataset(data_path)
    material_ids = list(dataset["material_ids"])
    two_theta = dataset["two_theta"]
    masks_np = dataset["mask"]

    graphs = []
    valid_indices = []
    for i, mp_id in enumerate(tqdm(material_ids, desc="Building graphs")):
        cif_path = os.path.join(cif_dir, f"{mp_id}.cif")
        if not os.path.exists(cif_path):
            continue
        try:
            graphs.append(structure_to_graph(cif_path))
            valid_indices.append(i)
        except Exception:
            pass

    valid_indices = np.array(valid_indices)
    targets = torch.from_numpy(two_theta[valid_indices]).float()
    masks = torch.from_numpy(masks_np[valid_indices]).bool()

    for i in range(len(targets)):
        m = masks[i]
        nv = m.sum().item()
        if nv > 0:
            targets[i, :nv] = targets[i, :nv].sort().values

    return graphs, targets, masks


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


def train_and_eval(graphs, targets, masks, variant, label_frac, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(graphs)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = min(500, n // 5)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    n_labeled = max(1, int(len(train_idx) * label_frac))
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
            bg, bi, bt, bm = make_batch(graphs, targets, masks, batch_idx, DEVICE)
            pred = model(bg, bi)

            is_labeled = torch.tensor([idx in labeled_set for idx in batch_idx.tolist()], dtype=torch.bool)
            n_l = is_labeled.sum().item()
            n_u = (~is_labeled).sum().item()

            if variant == "supervised":
                if n_l == 0:
                    continue
                loss = masked_sort_match_loss(pred[is_labeled], bt[is_labeled], bm[is_labeled], cost="mae")
            elif variant == "sort_match_ssl":
                l_loss = masked_sort_match_loss(pred[is_labeled], bt[is_labeled], bm[is_labeled], cost="mae") if n_l > 0 else torch.tensor(0.0)
                u_loss = masked_sort_match_loss(pred[~is_labeled], bt[~is_labeled], bm[~is_labeled], cost="mae") if n_u > 0 else torch.tensor(0.0)
                loss = (n_l * l_loss + n_u * u_loss) / max(n_l + n_u, 1)
            else:
                raise ValueError(variant)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate
    model.eval()
    errors = []
    with torch.no_grad():
        for start in range(0, len(test_idx), BATCH_SIZE):
            bidx = test_idx[start:start + BATCH_SIZE]
            bg, bi, bt, bm = make_batch(graphs, targets, masks, bidx, DEVICE)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/xrd_cache_5k.npz")
    parser.add_argument("--cif_dir", default="data/mp_structures/cifs")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Fall back to 1k dataset if 5k not available
    if not os.path.exists(args.data_path):
        print(f"{args.data_path} not found, falling back to data/xrd_cache.npz")
        args.data_path = "data/xrd_cache.npz"

    graphs, targets, masks = load_data(args.data_path, args.cif_dir)
    print(f"Loaded {len(graphs)} structures")

    results = {}
    for label_frac in LABEL_FRACS:
        for variant in ["supervised", "sort_match_ssl"]:
            maes = []
            for seed in SEEDS:
                print(f"  {variant}, frac={label_frac}, seed={seed}...")
                mae = train_and_eval(graphs, targets, masks, variant, label_frac, seed)
                maes.append(mae)
                print(f"    MAE = {mae:.2f} deg")
            key = f"{variant}_frac{label_frac}"
            results[key] = {
                "variant": variant,
                "label_frac": label_frac,
                "maes": maes,
                "mean": float(np.mean(maes)),
                "std": float(np.std(maes)),
            }

    # Summary
    print(f"\n{'='*70}")
    print(f"  LOW-LABEL ABLATION")
    print(f"{'='*70}")
    print(f"{'Frac':<8} {'Supervised':>18} {'Sort-match SSL':>18} {'Improvement':>14}")
    print(f"{'-'*58}")
    for frac in LABEL_FRACS:
        sup = results[f"supervised_frac{frac}"]
        ssl = results[f"sort_match_ssl_frac{frac}"]
        imp = (sup["mean"] - ssl["mean"]) / sup["mean"] * 100
        print(f"{frac:<8} {sup['mean']:>8.2f}+/-{sup['std']:.2f}° "
              f"{ssl['mean']:>8.2f}+/-{ssl['std']:.2f}° {imp:>+10.1f}%")

    with open(os.path.join(RESULTS_DIR, "low_label_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved")


if __name__ == "__main__":
    main()
