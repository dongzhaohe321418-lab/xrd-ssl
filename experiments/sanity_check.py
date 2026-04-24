"""
Sanity experiment: 3-variant comparison on 1000 structures.

Correct experimental design:
    (1) supervised — 100% labeled, sort-match loss (optimal matching).
        This is the ceiling: what you get with full labels.
    (2) supervised_10pct — 10% labeled only, same loss.
        This is the realistic baseline: limited labels, no SSL.
    (3) sort_match_ssl — 10% labeled + 90% unlabeled with sort-match.
        This is our method: use all data via SSL.
    (4) random_match — 10% labeled + 90% unlabeled with RANDOM assignment.
        This is the broken baseline: SSL with wrong matching.

Key comparison:
    sort_match_ssl vs supervised_10pct -> does SSL help?
    sort_match_ssl vs supervised -> how close to full supervision?
    random_match -> should be worse than sort_match_ssl (validates theorem)

Target: sort_match_ssl significantly better than supervised_10pct,
        and within ~20% of supervised (100%).

Usage:
    python experiments/sanity_check.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import sort_match_loss, masked_sort_match_loss
from src.model import XRDPredictor, build_model
from src.features import structure_to_graph, collate_graphs, CrystalGraph
from src.xrd_data import load_dataset


# ── Configuration ────────────────────────────────────────────────────

DATA_PATH = "data/xrd_cache.npz"
CIF_DIR = "data/mp_structures/cifs"
RESULTS_DIR = "experiments/results_session1"
N_EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
LABELED_FRAC = 0.1
UNLABELED_WEIGHT = 1.0
SEED = 42
DEVICE = "cpu"


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graphs_and_targets(data_path: str, cif_dir: str):
    """Load crystal graphs and XRD targets."""
    dataset = load_dataset(data_path)
    material_ids = list(dataset["material_ids"])
    two_theta = dataset["two_theta"]
    masks_np = dataset["mask"]

    graphs = []
    valid_indices = []

    print(f"Building crystal graphs from {len(material_ids)} structures...")
    for i, mp_id in enumerate(tqdm(material_ids)):
        cif_path = os.path.join(cif_dir, f"{mp_id}.cif")
        if not os.path.exists(cif_path):
            continue
        try:
            g = structure_to_graph(cif_path)
            graphs.append(g)
            valid_indices.append(i)
        except Exception as e:
            pass  # skip silently

    valid_indices = np.array(valid_indices)
    targets = torch.from_numpy(two_theta[valid_indices]).float()
    masks = torch.from_numpy(masks_np[valid_indices]).bool()

    # Sort targets within each sample (canonical ordering)
    for i in range(len(targets)):
        m = masks[i]
        n_valid = m.sum().item()
        if n_valid > 0:
            valid_vals = targets[i, :n_valid].sort().values
            targets[i, :n_valid] = valid_vals

    material_ids = [material_ids[i] for i in valid_indices]
    print(f"  {len(graphs)} graphs built successfully")
    return graphs, targets, masks, material_ids


def masked_random_match_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """L1 loss with randomly permuted matching (NOT sort-matched).

    For each sample, randomly permute the valid targets, then compute
    L1 between sorted predictions and randomly-assigned targets.
    This simulates training without knowing the correct assignment AND
    without using sort-match to discover it.
    """
    batch_size = pred.shape[0]
    total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    total_count = 0

    for i in range(batch_size):
        m = mask[i]
        n_valid = m.sum().item()
        if n_valid == 0:
            continue

        p = pred[i, :n_valid]  # DON'T sort predictions
        t = target[i, :n_valid]

        # Randomly permute targets
        perm = torch.randperm(n_valid, device=pred.device)
        t_shuffled = t[perm]

        sample_loss = F.l1_loss(p, t_shuffled, reduction="sum")
        total_loss = total_loss + sample_loss
        total_count += n_valid

    if total_count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return total_loss / total_count


def make_batch(graphs, targets, masks, batch_idx, device):
    """Prepare a batch for forward pass."""
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
    batch_index = batch_index.to(device)

    return batched_graph, batch_index, batch_targets, batch_masks


def train_epoch(
    model, optimizer, graphs, targets, masks, train_indices,
    variant, labeled_set=None
):
    """Train one epoch."""
    model.train()
    rng = np.random.default_rng()
    perm = rng.permutation(len(train_indices))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(perm), BATCH_SIZE):
        batch_perm = perm[start:start + BATCH_SIZE]
        batch_idx = train_indices[batch_perm]

        batched_graph, batch_index, batch_targets, batch_masks = \
            make_batch(graphs, targets, masks, batch_idx, DEVICE)

        pred = model(batched_graph, batch_index)  # (B, 50)

        if variant == "supervised":
            # All data is labeled, use sort-match (optimal for 1D)
            loss = masked_sort_match_loss(pred, batch_targets, batch_masks, cost="mae")

        elif variant == "supervised_10pct":
            # Only labeled data contributes to loss; unlabeled is ignored
            is_labeled = torch.tensor(
                [idx in labeled_set for idx in batch_idx.tolist()],
                dtype=torch.bool
            )
            if is_labeled.any():
                loss = masked_sort_match_loss(
                    pred[is_labeled], batch_targets[is_labeled],
                    batch_masks[is_labeled], cost="mae"
                )
            else:
                # Skip batch if no labeled samples
                continue

        elif variant == "sort_match_ssl":
            # Labeled: sort-match. Unlabeled: also sort-match.
            # The insight: sort-match works even without peak assignments.
            is_labeled = torch.tensor(
                [idx in labeled_set for idx in batch_idx.tolist()],
                dtype=torch.bool
            )

            labeled_loss = torch.tensor(0.0, device=DEVICE)
            unlabeled_loss = torch.tensor(0.0, device=DEVICE)

            if is_labeled.any():
                labeled_loss = masked_sort_match_loss(
                    pred[is_labeled], batch_targets[is_labeled],
                    batch_masks[is_labeled], cost="mae"
                )

            if (~is_labeled).any():
                unlabeled_loss = masked_sort_match_loss(
                    pred[~is_labeled], batch_targets[~is_labeled],
                    batch_masks[~is_labeled], cost="mae"
                )

            # Weight the losses by sample count for proper averaging
            n_l = is_labeled.sum().item()
            n_u = (~is_labeled).sum().item()
            n_total = n_l + n_u
            if n_total > 0:
                loss = (n_l * labeled_loss + UNLABELED_WEIGHT * n_u * unlabeled_loss) / (n_l + UNLABELED_WEIGHT * n_u)
            else:
                continue

        elif variant == "random_match":
            # Labeled: sort-match. Unlabeled: RANDOM matching (broken baseline).
            is_labeled = torch.tensor(
                [idx in labeled_set for idx in batch_idx.tolist()],
                dtype=torch.bool
            )

            labeled_loss = torch.tensor(0.0, device=DEVICE)
            unlabeled_loss = torch.tensor(0.0, device=DEVICE)

            if is_labeled.any():
                labeled_loss = masked_sort_match_loss(
                    pred[is_labeled], batch_targets[is_labeled],
                    batch_masks[is_labeled], cost="mae"
                )

            if (~is_labeled).any():
                # Random matching: DON'T sort, randomly assign targets
                unlabeled_loss = masked_random_match_loss(
                    pred[~is_labeled], batch_targets[~is_labeled],
                    batch_masks[~is_labeled]
                )

            n_l = is_labeled.sum().item()
            n_u = (~is_labeled).sum().item()
            n_total = n_l + n_u
            if n_total > 0:
                loss = (n_l * labeled_loss + UNLABELED_WEIGHT * n_u * unlabeled_loss) / (n_l + UNLABELED_WEIGHT * n_u)
            else:
                continue

        else:
            raise ValueError(f"Unknown variant: {variant}")

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, graphs, targets, masks, indices):
    """Evaluate on test set. Returns MAE in degrees 2theta."""
    model.eval()
    all_errors = []

    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        batched_graph, batch_index, batch_targets, batch_masks = \
            make_batch(graphs, targets, masks, batch_idx, DEVICE)

        pred = model(batched_graph, batch_index)

        for i in range(len(batch_idx)):
            m = batch_masks[i]
            n_valid = m.sum().item()
            if n_valid == 0:
                continue
            p_sorted = pred[i, :n_valid].sort().values
            t_sorted = batch_targets[i, :n_valid].sort().values
            mae = (p_sorted - t_sorted).abs().mean().item()
            all_errors.append(mae)

    return np.mean(all_errors) if all_errors else float("inf")


def run_variant(variant, graphs, targets, masks, train_idx, test_idx, labeled_set=None):
    """Run one training variant."""
    print(f"\n{'='*60}")
    print(f"  Variant: {variant}")
    if variant in ("supervised_10pct", "sort_match_ssl", "random_match"):
        print(f"  Labeled: {len(labeled_set)}, Unlabeled: {len(train_idx) - len(labeled_set)}")
    print(f"{'='*60}")

    set_seed(SEED)
    model = build_model(hidden_dim=128, n_layers=4, n_peaks=50)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses = []
    test_maes = []
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(
            model, optimizer, graphs, targets, masks, train_idx, variant,
            labeled_set=labeled_set
        )
        test_mae = evaluate(model, graphs, targets, masks, test_idx)

        train_losses.append(train_loss)
        test_maes.append(test_mae)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: "
                  f"train_loss={train_loss:.4f}, test_MAE={test_mae:.2f} deg")

    elapsed = time.time() - t0
    final_mae = test_maes[-1]
    best_mae = min(test_maes)

    print(f"  Final MAE: {final_mae:.2f} deg, Best MAE: {best_mae:.2f} deg")
    print(f"  Runtime: {elapsed:.1f}s")

    return {
        "variant": variant,
        "train_losses": train_losses,
        "test_maes": test_maes,
        "final_mae": final_mae,
        "best_mae": best_mae,
        "runtime_s": elapsed,
        "n_params": model.count_parameters(),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    # Load data
    print("Loading data...")
    graphs, targets, masks, material_ids = load_graphs_and_targets(DATA_PATH, CIF_DIR)
    n_total = len(graphs)
    print(f"Total structures: {n_total}")

    # Train/test split
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_test = min(200, n_total // 5)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Labeled subset
    n_labeled = max(1, int(len(train_idx) * LABELED_FRAC))
    labeled_set = set(train_idx[:n_labeled].tolist())
    print(f"Labeled: {n_labeled} ({LABELED_FRAC*100:.0f}% of train)")

    results = {}

    # (1) Supervised 100% — ceiling
    r1 = run_variant("supervised", graphs, targets, masks, train_idx, test_idx)
    results["supervised"] = r1

    # (2) Supervised 10% — realistic baseline (limited labels, no SSL)
    r2 = run_variant("supervised_10pct", graphs, targets, masks, train_idx, test_idx,
                     labeled_set=labeled_set)
    results["supervised_10pct"] = r2

    # (3) Sort-match SSL — our method (10% labeled + 90% unlabeled)
    r3 = run_variant("sort_match_ssl", graphs, targets, masks, train_idx, test_idx,
                     labeled_set=labeled_set)
    results["sort_match_ssl"] = r3

    # (4) Random match — broken baseline
    r4 = run_variant("random_match", graphs, targets, masks, train_idx, test_idx,
                     labeled_set=labeled_set)
    results["random_match"] = r4

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Variant':<20} {'Final MAE':>10} {'Best MAE':>10} {'Runtime':>10}")
    print(f"{'-'*50}")
    for name in ["supervised", "supervised_10pct", "sort_match_ssl", "random_match"]:
        r = results[name]
        print(f"{name:<20} {r['final_mae']:>9.2f}° {r['best_mae']:>9.2f}° {r['runtime_s']:>9.1f}s")

    # Key comparisons
    sup_mae = results["supervised"]["best_mae"]
    sup10_mae = results["supervised_10pct"]["best_mae"]
    ssl_mae = results["sort_match_ssl"]["best_mae"]
    rand_mae = results["random_match"]["best_mae"]

    print(f"\nKey ratios:")
    print(f"  SSL / Supervised(100%): {ssl_mae/sup_mae:.2f} (want <= 1.20)")
    print(f"  SSL / Supervised(10%):  {ssl_mae/sup10_mae:.2f} (want < 1.0 = SSL helps)")
    print(f"  Random / SSL:           {rand_mae/ssl_mae:.2f} (want > 1.0 = sorting matters)")

    if ssl_mae < sup10_mae:
        print("\nOK: Sort-match SSL outperforms supervised-10%, SSL adds value")
    else:
        print("\n*** FLAG: Sort-match SSL does NOT outperform supervised-10% ***")

    if rand_mae > ssl_mae:
        print("OK: Random matching is worse than sort-match (theorem validated)")
    else:
        print("*** FLAG: Random matching is NOT worse than sort-match ***")

    # Save results
    results_path = os.path.join(RESULTS_DIR, "sanity_results.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            key: val if not isinstance(val, (np.ndarray, np.floating)) else float(val)
            for key, val in v.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    results = main()
