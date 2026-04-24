"""
Session 2 experiment: 2D (position + intensity) sort-match comparison.

4 variants on 1000 structures, 30 epochs, labeled_frac=0.1:
    (1) supervised_2d — all data labeled, 2D sort-match (baseline)
    (2) sort_match_1d — SSL with 1D sort-match (positions only, Session 1)
    (3) sliced_ws_uniform — SSL with sliced-Wasserstein, uniform directions
    (4) sliced_ws_xrd_biased — SSL with sliced-Wasserstein, XRD-biased

Metrics:
    - Position MAE (degrees 2theta)
    - Intensity R^2

Usage:
    python experiments/run_2d.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.losses import masked_sort_match_loss
from src.losses_2d import sliced_wasserstein_loss
from src.model import build_model
from src.features import structure_to_graph, collate_graphs, CrystalGraph
from src.xrd_data import load_dataset


# ── Config ───────────────────────────────────────────────────────────

DATA_PATH = "data/xrd_cache.npz"
CIF_DIR = "data/mp_structures/cifs"
RESULTS_DIR = "experiments/results_session2"
N_EPOCHS = 30
BATCH_SIZE = 32
LR = 1e-3
LABELED_FRAC = 0.1
SEED = 42
DEVICE = "cpu"
N_SLICES = 128


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data():
    """Load graphs, 2theta targets, intensity targets, and masks."""
    dataset = load_dataset(DATA_PATH)
    material_ids = list(dataset["material_ids"])
    two_theta = dataset["two_theta"]
    intensity = dataset["intensity"]
    masks_np = dataset["mask"]

    graphs = []
    valid_indices = []

    print(f"Building crystal graphs from {len(material_ids)} structures...")
    for i, mp_id in enumerate(tqdm(material_ids)):
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
    theta_targets = torch.from_numpy(two_theta[valid_indices]).float()
    intensity_targets = torch.from_numpy(intensity[valid_indices]).float()
    masks = torch.from_numpy(masks_np[valid_indices]).bool()

    # Sort by 2theta within each sample (canonical ordering)
    for i in range(len(theta_targets)):
        m = masks[i]
        n_valid = m.sum().item()
        if n_valid > 0:
            order = theta_targets[i, :n_valid].argsort()
            theta_targets[i, :n_valid] = theta_targets[i, :n_valid][order]
            intensity_targets[i, :n_valid] = intensity_targets[i, :n_valid][order]

    print(f"  {len(graphs)} graphs built")
    return graphs, theta_targets, intensity_targets, masks


def make_batch(graphs, theta, intensity, masks, batch_idx, device):
    batch_graphs = [graphs[i] for i in batch_idx]
    batch_theta = theta[batch_idx].to(device)
    batch_intensity = intensity[batch_idx].to(device)
    batch_masks = masks[batch_idx].to(device)

    batched_graph, batch_index = collate_graphs(batch_graphs)
    batched_graph = CrystalGraph(
        node_feats=batched_graph.node_feats.to(device),
        edge_index=batched_graph.edge_index.to(device),
        edge_feats=batched_graph.edge_feats.to(device),
        n_atoms=batched_graph.n_atoms,
    )
    batch_index = batch_index.to(device)

    # Stack targets into 2D: (B, n_peaks, 2)
    batch_2d = torch.stack([batch_theta, batch_intensity], dim=-1)

    return batched_graph, batch_index, batch_theta, batch_intensity, batch_2d, batch_masks


def train_epoch(model, optimizer, graphs, theta, intensity, masks, train_idx,
                variant, labeled_set, predict_2d):
    model.train()
    rng = np.random.default_rng()
    perm = rng.permutation(len(train_idx))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(perm), BATCH_SIZE):
        batch_perm = perm[start:start + BATCH_SIZE]
        batch_idx = train_idx[batch_perm]

        bg, bi, bt, bint, b2d, bm = make_batch(
            graphs, theta, intensity, masks, batch_idx, DEVICE
        )

        pred = model(bg, bi)  # (B, 50) or (B, 50, 2)

        is_labeled = torch.tensor(
            [idx in labeled_set for idx in batch_idx.tolist()], dtype=torch.bool
        )

        if variant == "supervised_2d":
            # 2D model, all labeled
            if predict_2d:
                loss = sliced_wasserstein_loss(
                    pred, b2d, n_slices=N_SLICES, direction_prior="uniform", mask=bm
                )
            else:
                loss = masked_sort_match_loss(pred, bt, bm, cost="mae")

        elif variant == "sort_match_1d":
            # 1D model (positions only), SSL
            labeled_loss = torch.tensor(0.0, device=DEVICE)
            unlabeled_loss = torch.tensor(0.0, device=DEVICE)
            n_l = is_labeled.sum().item()
            n_u = (~is_labeled).sum().item()

            if n_l > 0:
                labeled_loss = masked_sort_match_loss(
                    pred[is_labeled], bt[is_labeled], bm[is_labeled], cost="mae"
                )
            if n_u > 0:
                unlabeled_loss = masked_sort_match_loss(
                    pred[~is_labeled], bt[~is_labeled], bm[~is_labeled], cost="mae"
                )
            total = n_l + n_u
            loss = (n_l * labeled_loss + n_u * unlabeled_loss) / max(total, 1)

        elif variant in ("sliced_ws_uniform", "sliced_ws_xrd_biased"):
            direction = "uniform" if variant == "sliced_ws_uniform" else "xrd_biased"

            labeled_loss = torch.tensor(0.0, device=DEVICE)
            unlabeled_loss = torch.tensor(0.0, device=DEVICE)
            n_l = is_labeled.sum().item()
            n_u = (~is_labeled).sum().item()

            if n_l > 0:
                labeled_loss = sliced_wasserstein_loss(
                    pred[is_labeled], b2d[is_labeled],
                    n_slices=N_SLICES, direction_prior=direction, mask=bm[is_labeled]
                )
            if n_u > 0:
                unlabeled_loss = sliced_wasserstein_loss(
                    pred[~is_labeled], b2d[~is_labeled],
                    n_slices=N_SLICES, direction_prior=direction, mask=bm[~is_labeled]
                )
            total = n_l + n_u
            loss = (n_l * labeled_loss + n_u * unlabeled_loss) / max(total, 1)

        else:
            raise ValueError(variant)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, graphs, theta, intensity, masks, indices, predict_2d):
    model.eval()
    pos_errors = []
    pred_intensities = []
    true_intensities = []

    for start in range(0, len(indices), BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        bg, bi, bt, bint, b2d, bm = make_batch(
            graphs, theta, intensity, masks, batch_idx, DEVICE
        )

        pred = model(bg, bi)

        for i in range(len(batch_idx)):
            m = bm[i]
            n_valid = m.sum().item()
            if n_valid == 0:
                continue

            if predict_2d:
                p_theta = pred[i, :n_valid, 0].sort().values
                p_int = pred[i, :n_valid, 1]
            else:
                p_theta = pred[i, :n_valid].sort().values
                p_int = None

            t_theta = bt[i, :n_valid].sort().values
            t_int = bint[i, :n_valid]

            pos_errors.append((p_theta - t_theta).abs().mean().item())

            if p_int is not None:
                # Match intensities using the position sorting order
                pred_order = pred[i, :n_valid, 0].argsort()
                true_order = bt[i, :n_valid].argsort()
                pred_intensities.extend(pred[i, :n_valid, 1][pred_order].cpu().numpy())
                true_intensities.extend(bint[i, :n_valid][true_order].cpu().numpy())

    pos_mae = np.mean(pos_errors) if pos_errors else float("inf")

    int_r2 = None
    if pred_intensities:
        pred_arr = np.array(pred_intensities)
        true_arr = np.array(true_intensities)
        ss_res = ((pred_arr - true_arr) ** 2).sum()
        ss_tot = ((true_arr - true_arr.mean()) ** 2).sum()
        int_r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    return pos_mae, int_r2


def run_variant(variant, graphs, theta, intensity, masks, train_idx, test_idx, labeled_set):
    predict_2d = variant in ("supervised_2d", "sliced_ws_uniform", "sliced_ws_xrd_biased")

    print(f"\n{'='*60}")
    print(f"  Variant: {variant} (2D={predict_2d})")
    print(f"{'='*60}")

    set_seed(SEED)
    model = build_model(hidden_dim=128, n_layers=4, n_peaks=50, predict_intensity=predict_2d)
    model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    train_losses = []
    test_pos_maes = []
    test_int_r2s = []
    t0 = time.time()

    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(
            model, optimizer, graphs, theta, intensity, masks, train_idx,
            variant, labeled_set, predict_2d
        )
        pos_mae, int_r2 = evaluate(model, graphs, theta, intensity, masks, test_idx, predict_2d)

        train_losses.append(train_loss)
        test_pos_maes.append(pos_mae)
        test_int_r2s.append(int_r2)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            r2_str = f"{int_r2:.3f}" if int_r2 is not None else "N/A"
            print(f"  Epoch {epoch+1:3d}/{N_EPOCHS}: loss={train_loss:.4f}, "
                  f"pos_MAE={pos_mae:.2f}°, int_R²={r2_str}")

    elapsed = time.time() - t0

    return {
        "variant": variant,
        "predict_2d": predict_2d,
        "train_losses": train_losses,
        "test_pos_maes": test_pos_maes,
        "test_int_r2s": test_int_r2s,
        "final_pos_mae": test_pos_maes[-1],
        "best_pos_mae": min(test_pos_maes),
        "final_int_r2": test_int_r2s[-1],
        "best_int_r2": max([r for r in test_int_r2s if r is not None], default=None),
        "runtime_s": elapsed,
        "n_params": model.count_parameters(),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    print("Loading data...")
    graphs, theta, intensity, masks = load_data()
    n_total = len(graphs)

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_total)
    n_test = min(200, n_total // 5)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    n_labeled = max(1, int(len(train_idx) * LABELED_FRAC))
    labeled_set = set(train_idx[:n_labeled].tolist())
    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}, Labeled: {n_labeled}")

    results = {}

    for variant in ["supervised_2d", "sort_match_1d", "sliced_ws_uniform", "sliced_ws_xrd_biased"]:
        r = run_variant(variant, graphs, theta, intensity, masks,
                       train_idx, test_idx, labeled_set)
        results[variant] = r

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<25} {'Pos MAE':>10} {'Int R²':>10} {'Runtime':>10}")
    print(f"{'-'*55}")
    for name, r in results.items():
        r2_str = f"{r['best_int_r2']:.3f}" if r['best_int_r2'] is not None else "N/A"
        print(f"{name:<25} {r['best_pos_mae']:>9.2f}° {r2_str:>10} {r['runtime_s']:>9.1f}s")

    # Save
    results_path = os.path.join(RESULTS_DIR, "results_2d.json")
    serializable = {}
    for k, v in results.items():
        serializable[k] = {
            key: (val if not isinstance(val, (np.ndarray, np.floating)) else float(val))
            for key, val in v.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    main()
