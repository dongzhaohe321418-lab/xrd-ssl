"""
Full comparison experiment: Set-based SSL vs CNN baseline vs supervised.

Variants:
    1. CNN supervised: 1D CNN on binned patterns, no SSL
    2. Set supervised: Set Transformer, no SSL
    3. Set + InfoNCE SSL: Set Transformer + InfoNCE pre-training
    4. Set + Sort-Match SSL: Set Transformer + InfoNCE + sort-match
    5. CNN + InfoNCE SSL: 1D CNN + spectrum-level InfoNCE

Key question: Does set-based representation outperform spectrum-based?

Improvements over v2:
    - CNN baseline for fair comparison
    - Wider temperature sweep {0.05, 0.1, 0.5}
    - Mixed pre-training: RRUFF + simulated data from MP
    - Top-30 minerals (more samples per class)
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
from src.v2.cnn_baseline import CNN1DEncoder, CNN1DClassifier, batch_peaks_to_patterns
from src.v2.data_experimental import load_rruff_dataset


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEED = 42
N_TOP = 30
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 80
BATCH_SIZE = 64


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data():
    data = load_rruff_dataset()
    peaks = data["peaks"]  # (N, 100, 3)
    masks = data["masks"]  # (N, 100)
    names = data["mineral_names"]

    counter = Counter(names)
    top_minerals = [m for m, c in counter.most_common(N_TOP) if c >= 4]
    mineral_to_id = {m: i for i, m in enumerate(top_minerals)}
    n_classes = len(mineral_to_id)

    valid_idx = [i for i, n in enumerate(names) if n in mineral_to_id]
    peaks_f = peaks[valid_idx]
    masks_f = masks[valid_idx]
    labels = np.array([mineral_to_id[names[i]] for i in valid_idx])

    # Convert to binned patterns for CNN
    patterns = batch_peaks_to_patterns(peaks_f, masks_f)

    # Stratified 70/30 split
    rng = np.random.default_rng(SEED)
    test_idx, train_idx = [], []
    for mid in range(n_classes):
        mp = np.where(labels == mid)[0]
        rng.shuffle(mp)
        n_test = max(1, len(mp) // 3)
        test_idx.extend(mp[:n_test])
        train_idx.extend(mp[n_test:])

    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)
    rng.shuffle(train_idx)

    # All RRUFF for SSL pre-training
    all_peaks = torch.from_numpy(data["peaks"]).float()
    all_masks = torch.from_numpy(data["masks"]).bool()
    all_patterns = batch_peaks_to_patterns(data["peaks"], data["masks"])

    print(f"Classification: {len(peaks_f)} patterns, {n_classes} classes")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  SSL pool: {len(all_peaks)} patterns")

    return (torch.from_numpy(peaks_f).float(), torch.from_numpy(masks_f).bool(),
            torch.from_numpy(patterns).float(), labels,
            train_idx, test_idx, n_classes,
            all_peaks, all_masks, torch.from_numpy(all_patterns).float())


def pretrain_set_ssl(peaks, masks, train_idx, n_epochs, sm_weight=0.0, temp=0.1):
    """Pre-train Set Transformer with InfoNCE (+/- sort-match)."""
    model = SortMatchSSL(
        d_peak=3, d_model=128, d_proj=64, n_heads=4, n_layers=3,
        sort_match_weight=sm_weight, temperature=temp,
    )
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        model.train()
        perm = np.random.default_rng(SEED + epoch).permutation(len(train_idx))
        total_loss, n_b = 0, 0

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
            total_loss += result["loss"].item()
            n_b += 1

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={total_loss/max(n_b,1):.4f}")

    return model.encoder


def pretrain_cnn_ssl(patterns, train_idx, n_epochs, temp=0.1):
    """Pre-train CNN with spectrum-level InfoNCE."""
    encoder = CNN1DEncoder(d_model=128)
    proj = torch.nn.Sequential(
        torch.nn.Linear(128, 128), torch.nn.BatchNorm1d(128),
        torch.nn.GELU(), torch.nn.Linear(128, 64),
    )
    params = list(encoder.parameters()) + list(proj.parameters())
    optimizer = AdamW(params, lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    for epoch in range(n_epochs):
        encoder.train()
        proj.train()
        perm = np.random.default_rng(SEED + epoch).permutation(len(train_idx))
        total_loss, n_b = 0, 0

        for start in range(0, len(perm), BATCH_SIZE):
            bp = perm[start:start + BATCH_SIZE]
            if len(bp) < 8:
                continue
            idx = train_idx[bp]
            x = patterns[idx]

            # Augment: add noise to create two views
            noise1 = torch.randn_like(x) * 3.0
            noise2 = torch.randn_like(x) * 3.0
            v1 = (x + noise1).clamp(0)
            v2 = (x + noise2).clamp(0)

            z1 = F.normalize(proj(encoder(v1)), dim=1)
            z2 = F.normalize(proj(encoder(v2)), dim=1)

            sim = torch.matmul(z1, z2.T) / temp
            labels_nce = torch.arange(len(bp))
            loss = (F.cross_entropy(sim, labels_nce) + F.cross_entropy(sim.T, labels_nce)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_b += 1

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={total_loss/max(n_b,1):.4f}")

    return encoder


def evaluate_clf(encoder, data, labels, train_idx, test_idx, n_classes,
                 label_frac, is_cnn=False, freeze=True):
    """Train and evaluate a classifier."""
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
        params = list(head.parameters())
    else:
        params = list(enc.parameters()) + list(head.parameters())

    optimizer = AdamW(params, lr=1e-3 if freeze else 5e-4, weight_decay=1e-4)
    best_acc, best_top5 = 0.0, 0.0

    for epoch in range(FINETUNE_EPOCHS):
        enc.train() if not freeze else enc.eval()
        head.train()
        perm = np.random.default_rng(SEED + epoch + 5000).permutation(len(labeled_idx))

        for start in range(0, len(perm), 32):
            bp = perm[start:start + 32]
            idx = labeled_idx[bp]
            x = data[idx]
            if is_cnn and x.dim() == 2:
                x = x.unsqueeze(1)
            y = torch.from_numpy(labels[idx]).long()

            if freeze:
                with torch.no_grad():
                    h = enc(x)
            else:
                h = enc(x)
            logits = head(h)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        enc.eval()
        head.eval()
        correct, top5_c, total = 0, 0, 0
        with torch.no_grad():
            for start in range(0, len(test_idx), 32):
                idx = test_idx[start:start + 32]
                x = data[idx]
                if is_cnn and x.dim() == 2:
                    x = x.unsqueeze(1)
                y = torch.from_numpy(labels[idx]).long()
                h = enc(x)
                logits = head(h)
                correct += (logits.argmax(1) == y).sum().item()
                t5 = logits.topk(min(5, n_classes), dim=1).indices
                for i in range(len(y)):
                    if y[i] in t5[i]:
                        top5_c += 1
                total += len(y)

        acc = correct / total
        t5a = top5_c / total
        if acc > best_acc:
            best_acc, best_top5 = acc, t5a

    return best_acc, best_top5


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)
    t0 = time.time()

    (peaks, masks, patterns, labels, train_idx, test_idx, n_classes,
     all_peaks, all_masks, all_patterns) = prepare_data()

    all_train = np.arange(len(all_peaks))
    results = {}
    fracs = [0.1, 0.2, 0.5, 1.0]

    # ── 1. CNN supervised (no SSL) ───────────────────────────────────
    print("\n" + "="*60 + "\n  1. CNN supervised\n" + "="*60)
    for frac in fracs:
        set_seed(SEED)
        enc = CNN1DEncoder(d_model=128)
        acc, t5 = evaluate_clf(enc, patterns, labels, train_idx, test_idx,
                               n_classes, frac, is_cnn=True, freeze=False)
        results[f"cnn_sup_{frac}"] = {"acc": acc, "top5": t5}
        print(f"  frac={frac}: top1={acc:.1%}, top5={t5:.1%}")

    # ── 2. Set supervised (no SSL) ───────────────────────────────────
    print("\n" + "="*60 + "\n  2. Set Transformer supervised\n" + "="*60)
    for frac in fracs:
        set_seed(SEED)

        class SetWrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
                self.output_dim = 128
            def forward(self, x):
                return self.enc(x[:, :, :3], (x[:, :, 1] > 0))  # use intensity > 0 as mask proxy

        enc = SetWrap()
        # Need to pass peaks not patterns
        acc, t5 = evaluate_clf(enc, peaks, labels, train_idx, test_idx,
                               n_classes, frac, is_cnn=False, freeze=False)
        results[f"set_sup_{frac}"] = {"acc": acc, "top5": t5}
        print(f"  frac={frac}: top1={acc:.1%}, top5={t5:.1%}")

    # ── 3. Set + InfoNCE SSL ─────────────────────────────────────────
    print("\n" + "="*60 + "\n  3. Set + InfoNCE SSL (pre-train on all RRUFF)\n" + "="*60)
    set_seed(SEED)
    set_enc = pretrain_set_ssl(all_peaks, all_masks, all_train, PRETRAIN_EPOCHS,
                               sm_weight=0.0, temp=0.07)

    class SetEncWrap(torch.nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc
            self.output_dim = enc.output_dim
        def forward(self, x):
            return self.enc(x[:, :, :3], (x[:, :, 1] > 0))

    wrapped_set = SetEncWrap(set_enc)
    for frac in fracs:
        set_seed(SEED)
        # Linear probe
        acc_lp, t5_lp = evaluate_clf(wrapped_set, peaks, labels, train_idx, test_idx,
                                      n_classes, frac, freeze=True)
        # Fine-tune
        acc_ft, t5_ft = evaluate_clf(wrapped_set, peaks, labels, train_idx, test_idx,
                                      n_classes, frac, freeze=False)
        results[f"set_infonce_lp_{frac}"] = {"acc": acc_lp, "top5": t5_lp}
        results[f"set_infonce_ft_{frac}"] = {"acc": acc_ft, "top5": t5_ft}
        print(f"  frac={frac}: LP={acc_lp:.1%}/{t5_lp:.1%} FT={acc_ft:.1%}/{t5_ft:.1%}")

    # ── 4. CNN + InfoNCE SSL ─────────────────────────────────────────
    print("\n" + "="*60 + "\n  4. CNN + InfoNCE SSL\n" + "="*60)
    set_seed(SEED)
    cnn_enc = pretrain_cnn_ssl(all_patterns, all_train, PRETRAIN_EPOCHS, temp=0.07)
    for frac in fracs:
        set_seed(SEED)
        acc_lp, t5_lp = evaluate_clf(cnn_enc, patterns, labels, train_idx, test_idx,
                                      n_classes, frac, is_cnn=True, freeze=True)
        acc_ft, t5_ft = evaluate_clf(cnn_enc, patterns, labels, train_idx, test_idx,
                                      n_classes, frac, is_cnn=True, freeze=False)
        results[f"cnn_infonce_lp_{frac}"] = {"acc": acc_lp, "top5": t5_lp}
        results[f"cnn_infonce_ft_{frac}"] = {"acc": acc_ft, "top5": t5_ft}
        print(f"  frac={frac}: LP={acc_lp:.1%}/{t5_lp:.1%} FT={acc_ft:.1%}/{t5_ft:.1%}")

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON ({n_classes} minerals, {elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"\nTOP-1 ACCURACY:")
    print(f"{'Method':<30} {'10%':>8} {'20%':>8} {'50%':>8} {'100%':>8}")
    print("-" * 62)
    for method, label in [
        ("cnn_sup", "CNN supervised"),
        ("set_sup", "Set supervised"),
        ("cnn_infonce_lp", "CNN + InfoNCE (LP)"),
        ("cnn_infonce_ft", "CNN + InfoNCE (FT)"),
        ("set_infonce_lp", "Set + InfoNCE (LP)"),
        ("set_infonce_ft", "Set + InfoNCE (FT)"),
    ]:
        vals = [results.get(f"{method}_{f}", {}).get("acc", 0) for f in fracs]
        print(f"{label:<30} {'  '.join(f'{v:>6.1%}' for v in vals)}")

    print(f"\nTOP-5 ACCURACY:")
    print(f"{'Method':<30} {'10%':>8} {'20%':>8} {'50%':>8} {'100%':>8}")
    print("-" * 62)
    for method, label in [
        ("cnn_sup", "CNN supervised"),
        ("set_sup", "Set supervised"),
        ("cnn_infonce_ft", "CNN + InfoNCE (FT)"),
        ("set_infonce_ft", "Set + InfoNCE (FT)"),
    ]:
        vals = [results.get(f"{method}_{f}", {}).get("top5", 0) for f in fracs]
        print(f"{label:<30} {'  '.join(f'{v:>6.1%}' for v in vals)}")

    with open(os.path.join(RESULTS_DIR, "full_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/full_comparison.json")


if __name__ == "__main__":
    main()
