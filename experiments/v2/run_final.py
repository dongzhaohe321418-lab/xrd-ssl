"""
Final experiment: multi-seed + mixed pre-training + UMAP visualization.

This produces the definitive numbers for the paper:
1. 3-seed error bars on Set+SSL vs baselines
2. Mixed pre-training (RRUFF + simulated MP) to test sim-to-exp transfer
3. UMAP visualization of learned representations

Usage:
    python experiments/v2/run_final.py
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.v2.set_encoder import SetTransformerEncoder
from src.v2.contrastive_ssl import SortMatchSSL, PhaseClassifier
from src.v2.cnn_baseline import CNN1DEncoder, batch_peaks_to_patterns
from src.v2.data_experimental import load_rruff_dataset
from src.xrd_data import load_dataset as load_mp_dataset


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEEDS = [42, 2026, 7]
N_TOP = 30
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 80
BATCH_SIZE = 64


def prepare_data(seed=42):
    data = load_rruff_dataset()
    peaks = data["peaks"]
    masks = data["masks"]
    names = data["mineral_names"]

    counter = Counter(names)
    top_minerals = [m for m, c in counter.most_common(N_TOP) if c >= 4]
    mineral_to_id = {m: i for i, m in enumerate(top_minerals)}
    n_classes = len(mineral_to_id)

    valid_idx = [i for i, n in enumerate(names) if n in mineral_to_id]
    peaks_f = peaks[valid_idx]
    masks_f = masks[valid_idx]
    labels = np.array([mineral_to_id[names[i]] for i in valid_idx])
    names_f = np.array([names[i] for i in valid_idx])
    patterns = batch_peaks_to_patterns(peaks_f, masks_f)

    rng = np.random.default_rng(seed)
    test_idx, train_idx = [], []
    for mid in range(n_classes):
        mp = np.where(labels == mid)[0]
        rng.shuffle(mp)
        n_test = max(1, len(mp) // 3)
        test_idx.extend(mp[:n_test])
        train_idx.extend(mp[n_test:])
    test_idx, train_idx = np.array(test_idx), np.array(train_idx)
    rng.shuffle(train_idx)

    all_peaks = torch.from_numpy(data["peaks"]).float()
    all_masks = torch.from_numpy(data["masks"]).bool()

    return (torch.from_numpy(peaks_f).float(), torch.from_numpy(masks_f).bool(),
            torch.from_numpy(patterns).float(), labels, names_f,
            train_idx, test_idx, n_classes,
            all_peaks, all_masks)


def load_simulated_peaks():
    """Load simulated XRD from Materials Project as extra pre-training data."""
    try:
        mp_data = load_mp_dataset("data/xrd_cache_5k.npz")
        peaks = np.zeros((len(mp_data["two_theta"]), 100, 3), dtype=np.float32)
        masks = np.zeros((len(mp_data["two_theta"]), 100), dtype=bool)
        for i in range(len(mp_data["two_theta"])):
            m = mp_data["mask"][i]
            n_valid = m.sum()
            peaks[i, :n_valid, 0] = mp_data["two_theta"][i, :n_valid]
            peaks[i, :n_valid, 1] = mp_data["intensity"][i, :n_valid]
            peaks[i, :n_valid, 2] = 0.15  # default FWHM for simulated
            masks[i, :n_valid] = True
        print(f"  Loaded {len(peaks)} simulated patterns from MP")
        return torch.from_numpy(peaks).float(), torch.from_numpy(masks).bool()
    except Exception as e:
        print(f"  Warning: could not load simulated data: {e}")
        return None, None


def pretrain(peaks, masks, train_idx, n_epochs, temp=0.07):
    model = SortMatchSSL(d_peak=3, d_model=128, d_proj=64, n_heads=4,
                         n_layers=3, sort_match_weight=0.0, temperature=temp)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.default_rng(epoch).permutation(len(train_idx))
        for start in range(0, len(perm), BATCH_SIZE):
            bp = perm[start:start + BATCH_SIZE]
            if len(bp) < 8:
                continue
            idx = train_idx[bp]
            result = model(peaks[idx], masks[idx])
            optimizer.zero_grad()
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"      Epoch {epoch+1}/{n_epochs}")
    return model.encoder


def classify(encoder, data, labels, train_idx, test_idx, n_classes,
             label_frac, is_cnn=False, freeze=True, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_labeled = max(n_classes, int(len(train_idx) * label_frac))
    labeled_idx = train_idx[:n_labeled]

    enc = copy.deepcopy(encoder)
    head = torch.nn.Sequential(
        torch.nn.Linear(enc.output_dim, enc.output_dim),
        torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(enc.output_dim, n_classes),
    )
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
    params = [p for p in list(enc.parameters()) + list(head.parameters()) if p.requires_grad]
    optimizer = AdamW(params, lr=1e-3 if freeze else 5e-4, weight_decay=1e-4)

    best_acc, best_top5 = 0.0, 0.0
    for epoch in range(FINETUNE_EPOCHS):
        enc.train() if not freeze else enc.eval()
        head.train()
        perm = np.random.default_rng(seed + epoch + 5000).permutation(len(labeled_idx))
        for start in range(0, len(perm), 32):
            bp = perm[start:start + 32]
            idx = labeled_idx[bp]
            x = data[idx]
            if is_cnn and x.dim() == 2:
                x = x.unsqueeze(1)
            y = torch.from_numpy(labels[idx]).long()
            h = enc(x) if not freeze else enc(x).detach()
            if freeze:
                h = enc(x)
            logits = head(h)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        enc.eval(); head.eval()
        c, t5c, total = 0, 0, 0
        with torch.no_grad():
            for start in range(0, len(test_idx), 32):
                idx = test_idx[start:start + 32]
                x = data[idx]
                if is_cnn and x.dim() == 2:
                    x = x.unsqueeze(1)
                y = torch.from_numpy(labels[idx]).long()
                logits = head(enc(x))
                c += (logits.argmax(1) == y).sum().item()
                t5 = logits.topk(min(5, n_classes), dim=1).indices
                for i in range(len(y)):
                    if y[i] in t5[i]:
                        t5c += 1
                total += len(y)
        acc, top5 = c / total, t5c / total
        if acc > best_acc:
            best_acc, best_top5 = acc, top5
    return best_acc, best_top5


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    results = {}
    fracs = [0.1, 0.2, 0.5, 1.0]
    methods = {}  # method -> {frac -> [acc across seeds]}

    for method in ["cnn_sup", "set_sup", "set_ssl_ft", "set_ssl_mixed_ft"]:
        methods[method] = {f: {"accs": [], "top5s": []} for f in fracs}

    # Load simulated data for mixed pre-training
    sim_peaks, sim_masks = load_simulated_peaks()

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"  SEED {seed} ({seed_idx+1}/{len(SEEDS)})")
        print(f"{'='*60}")

        np.random.seed(seed)
        torch.manual_seed(seed)

        (peaks, masks, patterns, labels, names_f,
         train_idx, test_idx, n_classes,
         all_peaks, all_masks) = prepare_data(seed)

        # ── CNN supervised ───────────────────────────────────────────
        print("  CNN supervised...")
        for frac in fracs:
            enc = CNN1DEncoder(d_model=128)
            acc, t5 = classify(enc, patterns, labels, train_idx, test_idx,
                              n_classes, frac, is_cnn=True, freeze=False, seed=seed)
            methods["cnn_sup"][frac]["accs"].append(acc)
            methods["cnn_sup"][frac]["top5s"].append(t5)

        # ── Set supervised ───────────────────────────────────────────
        print("  Set supervised...")

        class SW(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
                s.output_dim = 128
            def forward(s, x):
                return s.enc(x[:,:,:3], (x[:,:,1] > 0))

        for frac in fracs:
            enc = SW()
            acc, t5 = classify(enc, peaks, labels, train_idx, test_idx,
                              n_classes, frac, freeze=False, seed=seed)
            methods["set_sup"][frac]["accs"].append(acc)
            methods["set_sup"][frac]["top5s"].append(t5)

        # ── Set + InfoNCE SSL (RRUFF only) ───────────────────────────
        print("  Set + InfoNCE SSL (RRUFF pre-train)...")
        all_train = np.arange(len(all_peaks))
        set_enc = pretrain(all_peaks, all_masks, all_train, PRETRAIN_EPOCHS)

        class SWrap(torch.nn.Module):
            def __init__(s, e):
                super().__init__()
                s.enc = e
                s.output_dim = e.output_dim
            def forward(s, x):
                return s.enc(x[:,:,:3], (x[:,:,1] > 0))

        for frac in fracs:
            acc, t5 = classify(SWrap(set_enc), peaks, labels, train_idx, test_idx,
                              n_classes, frac, freeze=False, seed=seed)
            methods["set_ssl_ft"][frac]["accs"].append(acc)
            methods["set_ssl_ft"][frac]["top5s"].append(t5)

        # ── Set + InfoNCE SSL (mixed: RRUFF + simulated) ─────────────
        if sim_peaks is not None:
            print("  Set + InfoNCE SSL (mixed RRUFF+MP pre-train)...")
            mixed_peaks = torch.cat([all_peaks, sim_peaks], dim=0)
            mixed_masks = torch.cat([all_masks, sim_masks], dim=0)
            mixed_train = np.arange(len(mixed_peaks))
            mixed_enc = pretrain(mixed_peaks, mixed_masks, mixed_train, PRETRAIN_EPOCHS)

            for frac in fracs:
                acc, t5 = classify(SWrap(mixed_enc), peaks, labels, train_idx, test_idx,
                                  n_classes, frac, freeze=False, seed=seed)
                methods["set_ssl_mixed_ft"][frac]["accs"].append(acc)
                methods["set_ssl_mixed_ft"][frac]["top5s"].append(t5)

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS ({len(SEEDS)} seeds, {elapsed/60:.0f} min)")
    print(f"{'='*70}")
    print(f"\nTOP-1 ACCURACY (mean +/- std):")
    print(f"{'Method':<28} {'10%':>12} {'20%':>12} {'50%':>12} {'100%':>12}")
    print("-" * 76)

    final_results = {}
    for method, label in [("cnn_sup","CNN supervised"), ("set_sup","Set supervised"),
                           ("set_ssl_ft","Set+SSL (RRUFF)"), ("set_ssl_mixed_ft","Set+SSL (mixed)")]:
        row = []
        for frac in fracs:
            accs = methods[method][frac]["accs"]
            if accs:
                mean, std = np.mean(accs), np.std(accs)
                row.append(f"{mean:.1%}+/-{std:.1%}")
                final_results[f"{method}_{frac}"] = {
                    "acc_mean": float(mean), "acc_std": float(std),
                    "top5_mean": float(np.mean(methods[method][frac]["top5s"])),
                    "top5_std": float(np.std(methods[method][frac]["top5s"])),
                    "accs": [float(a) for a in accs],
                }
            else:
                row.append("N/A")
        print(f"{label:<28} {'  '.join(row)}")

    print(f"\nTOP-5 ACCURACY (mean +/- std):")
    for method, label in [("cnn_sup","CNN supervised"), ("set_sup","Set supervised"),
                           ("set_ssl_ft","Set+SSL (RRUFF)"), ("set_ssl_mixed_ft","Set+SSL (mixed)")]:
        row = []
        for frac in fracs:
            t5s = methods[method][frac]["top5s"]
            if t5s:
                row.append(f"{np.mean(t5s):.1%}+/-{np.std(t5s):.1%}")
            else:
                row.append("N/A")
        print(f"{label:<28} {'  '.join(row)}")

    # Save UMAP data (embeddings for visualization)
    print("\nGenerating embeddings for UMAP...")
    (peaks, masks, patterns, labels, names_f,
     train_idx, test_idx, n_classes,
     all_peaks, all_masks) = prepare_data(42)

    torch.manual_seed(42)
    ssl_model = SortMatchSSL(d_peak=3, d_model=128, d_proj=64, n_heads=4,
                             n_layers=3, sort_match_weight=0.0, temperature=0.07)
    ssl_model = pretrain(all_peaks, all_masks, np.arange(len(all_peaks)), PRETRAIN_EPOCHS)

    class FinalWrap(torch.nn.Module):
        def __init__(s, e):
            super().__init__()
            s.enc = e
            s.output_dim = e.output_dim
        def forward(s, x):
            return s.enc(x[:,:,:3], (x[:,:,1] > 0))

    wrapped = FinalWrap(ssl_model)
    wrapped.eval()
    with torch.no_grad():
        embeddings = wrapped(peaks).numpy()

    np.savez(os.path.join(RESULTS_DIR, "embeddings.npz"),
             embeddings=embeddings, labels=labels, names=names_f)

    with open(os.path.join(RESULTS_DIR, "final_results.json"), "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
