"""
Session 4: Failure analysis — identify what doesn't work.

Train sort-match SSL, find the 100 worst-predicted structures,
analyze them by crystal system, space group, atom count, peak count.

This is honest science — report what fails.

Usage:
    python experiments/failure_analysis.py [--data_path data/xrd_cache.npz]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import masked_sort_match_loss
from src.model import build_model
from src.features import structure_to_graph, collate_graphs, CrystalGraph
from src.xrd_data import load_dataset


RESULTS_DIR = "experiments/results_session4"
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
SEED = 42
DEVICE = "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/xrd_cache.npz")
    parser.add_argument("--cif_dir", default="data/mp_structures/cifs")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load data
    dataset = load_dataset(args.data_path)
    material_ids = list(dataset["material_ids"])
    two_theta = dataset["two_theta"]
    masks_np = dataset["mask"]

    # Load metadata
    meta_path = os.path.join(os.path.dirname(args.cif_dir), "structures.json")
    with open(meta_path) as f:
        metadata = {e["material_id"]: e for e in json.load(f)}

    # Build graphs
    graphs = []
    valid_indices = []
    for i, mp_id in enumerate(tqdm(material_ids, desc="Building graphs")):
        cif_path = os.path.join(args.cif_dir, f"{mp_id}.cif")
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
    valid_ids = [material_ids[i] for i in valid_indices]

    for i in range(len(targets)):
        m = masks[i]
        nv = m.sum().item()
        if nv > 0:
            targets[i, :nv] = targets[i, :nv].sort().values

    # Split
    n = len(graphs)
    perm = np.random.default_rng(SEED).permutation(n)
    n_test = min(200, n // 5)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    labeled_set = set(train_idx[:max(1, int(len(train_idx) * 0.1))].tolist())

    # Train sort_match_ssl
    print("Training model...")
    model = build_model(hidden_dim=128, n_layers=4, n_peaks=50)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(N_EPOCHS):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
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
            bt = targets[batch_idx].to(DEVICE)
            bm = masks[batch_idx].to(DEVICE)
            pred = model(batched_graph, batch_index.to(DEVICE))

            is_labeled = torch.tensor([idx in labeled_set for idx in batch_idx.tolist()], dtype=torch.bool)
            n_l, n_u = is_labeled.sum().item(), (~is_labeled).sum().item()
            l_loss = masked_sort_match_loss(pred[is_labeled], bt[is_labeled], bm[is_labeled]) if n_l > 0 else torch.tensor(0.0)
            u_loss = masked_sort_match_loss(pred[~is_labeled], bt[~is_labeled], bm[~is_labeled]) if n_u > 0 else torch.tensor(0.0)
            loss = (n_l * l_loss + n_u * u_loss) / max(n_l + n_u, 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate per-structure on TEST set
    model.eval()
    per_struct = []
    with torch.no_grad():
        for start in range(0, len(test_idx), BATCH_SIZE):
            bidx = test_idx[start:start + BATCH_SIZE]
            batch_graphs = [graphs[i] for i in bidx]
            batched_graph, batch_index = collate_graphs(batch_graphs)
            batched_graph = CrystalGraph(
                node_feats=batched_graph.node_feats.to(DEVICE),
                edge_index=batched_graph.edge_index.to(DEVICE),
                edge_feats=batched_graph.edge_feats.to(DEVICE),
                n_atoms=batched_graph.n_atoms,
            )
            bt = targets[bidx].to(DEVICE)
            bm = masks[bidx].to(DEVICE)
            pred = model(batched_graph, batch_index.to(DEVICE))

            for i, idx in enumerate(bidx):
                m = bm[i]
                nv = m.sum().item()
                if nv == 0:
                    continue
                p = pred[i, :nv].sort().values
                t = bt[i, :nv].sort().values
                mae = (p - t).abs().mean().item()
                mp_id = valid_ids[idx]
                meta = metadata.get(mp_id, {})
                per_struct.append({
                    "material_id": mp_id,
                    "mae": mae,
                    "n_peaks": nv,
                    "crystal_system": meta.get("crystal_system", "unknown"),
                    "spacegroup": meta.get("spacegroup", "unknown"),
                    "nsites": meta.get("nsites", 0),
                    "formula": meta.get("formula", "unknown"),
                })

    per_struct.sort(key=lambda x: -x["mae"])

    # Report
    n_worst = min(100, len(per_struct))
    worst = per_struct[:n_worst]
    best = per_struct[-n_worst:]

    print(f"\n{'='*70}")
    print(f"  FAILURE ANALYSIS (top {n_worst} worst)")
    print(f"{'='*70}")

    # Crystal system distribution
    worst_cs = Counter(s["crystal_system"] for s in worst)
    all_cs = Counter(s["crystal_system"] for s in per_struct)

    print(f"\nCrystal system distribution:")
    print(f"{'System':<15} {'Worst %':>10} {'Overall %':>10} {'Enrichment':>12}")
    for cs in sorted(all_cs.keys()):
        w_pct = worst_cs.get(cs, 0) / n_worst * 100
        a_pct = all_cs.get(cs, 0) / len(per_struct) * 100
        enrich = w_pct / a_pct if a_pct > 0 else 0
        flag = " ***" if enrich > 1.5 else ""
        print(f"{cs:<15} {w_pct:>9.1f}% {a_pct:>9.1f}% {enrich:>10.2f}x{flag}")

    # Peak count
    worst_peaks = [s["n_peaks"] for s in worst]
    all_peaks = [s["n_peaks"] for s in per_struct]
    print(f"\nPeak count: worst mean={np.mean(worst_peaks):.1f} vs overall mean={np.mean(all_peaks):.1f}")

    # Atom count
    worst_atoms = [s["nsites"] for s in worst]
    all_atoms = [s["nsites"] for s in per_struct]
    print(f"Atom count: worst mean={np.mean(worst_atoms):.1f} vs overall mean={np.mean(all_atoms):.1f}")

    # Top 10 worst
    print(f"\nTop 10 worst structures:")
    for s in worst[:10]:
        print(f"  {s['material_id']:>12} {s['formula']:>15} {s['crystal_system']:>12} "
              f"{s['nsites']:>3} atoms  {s['n_peaks']:>3} peaks  MAE={s['mae']:.2f}°")

    # Save
    results = {
        "worst_100": worst,
        "crystal_system_enrichment": {
            cs: {
                "worst_pct": worst_cs.get(cs, 0) / n_worst * 100,
                "overall_pct": all_cs.get(cs, 0) / len(per_struct) * 100,
            }
            for cs in all_cs
        },
        "summary": {
            "worst_mean_peaks": float(np.mean(worst_peaks)),
            "overall_mean_peaks": float(np.mean(all_peaks)),
            "worst_mean_atoms": float(np.mean(worst_atoms)),
            "overall_mean_atoms": float(np.mean(all_atoms)),
            "worst_mean_mae": float(np.mean([s["mae"] for s in worst])),
            "overall_mean_mae": float(np.mean([s["mae"] for s in per_struct])),
        },
    }
    with open(os.path.join(RESULTS_DIR, "failure_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/failure_analysis.json")


if __name__ == "__main__":
    main()
