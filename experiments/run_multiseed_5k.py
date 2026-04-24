"""
Multi-seed experiment on 5K dataset.
Wraps run_multiseed.py logic with 5K data path.
"""
from __future__ import annotations
import json
import os
import sys
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import masked_sort_match_loss
from src.model import build_model
from src.features import structure_to_graph, collate_graphs, CrystalGraph
from src.xrd_data import load_dataset

DATA_PATH = "data/xrd_cache_5k.npz"
CIF_DIR = "data/mp_structures/cifs"
RESULTS_DIR = "experiments/results_5k"
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
LABELED_FRAC = 0.1
SEEDS = [42, 2026, 7]
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


def train_one_run(graphs, targets, masks, variant, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n = len(graphs)
    perm = np.random.default_rng(seed).permutation(n)
    n_test = min(1000, n // 5)
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
            bg, bi, bt, bm = make_batch(graphs, targets, masks, batch_idx, DEVICE)
            pred = model(bg, bi)
            is_labeled = torch.tensor([idx in labeled_set for idx in batch_idx.tolist()], dtype=torch.bool)
            n_l, n_u = is_labeled.sum().item(), (~is_labeled).sum().item()
            if variant == "supervised":
                loss = masked_sort_match_loss(pred, bt, bm, cost="mae")
            elif variant == "supervised_10pct":
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
    os.makedirs(RESULTS_DIR, exist_ok=True)
    graphs, targets, masks = load_data()
    print(f"Loaded {len(graphs)} structures from 5K dataset")

    results = {}
    for variant in ["supervised", "supervised_10pct", "sort_match_ssl"]:
        maes = []
        for seed in SEEDS:
            print(f"  {variant}, seed={seed}...")
            mae = train_one_run(graphs, targets, masks, variant, seed)
            maes.append(mae)
            print(f"    MAE = {mae:.2f} deg")
        results[variant] = {"maes": maes, "mean": float(np.mean(maes)), "std": float(np.std(maes))}

    print(f"\n{'='*60}")
    print(f"  5K MULTI-SEED RESULTS (3 seeds)")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Mean MAE':>10} {'Std':>8}")
    for name, r in results.items():
        print(f"{name:<20} {r['mean']:>9.2f}° {r['std']:>7.2f}°")

    with open(os.path.join(RESULTS_DIR, "multiseed_5k.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/multiseed_5k.json")


if __name__ == "__main__":
    main()
