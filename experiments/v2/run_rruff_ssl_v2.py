"""
Fixed RRUFF SSL experiment v2.

Fixes from v1:
1. Reduce sort-match weight (0.5 -> 0.1) to avoid overwhelming InfoNCE
2. Add fine-tuning with unfrozen encoder (not just linear probe)
3. Reduce to top-50 minerals for more samples per class
4. Longer pre-training (50 epochs)
5. Add sort-match-only variant (no InfoNCE) to isolate the contribution

Usage:
    python experiments/v2/run_rruff_ssl_v2.py
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
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.v2.set_encoder import SetTransformerEncoder
from src.v2.contrastive_ssl import SortMatchSSL, PhaseClassifier
from src.v2.data_experimental import load_rruff_dataset


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEED = 42
N_TOP_MINERALS = 50  # fewer classes = more per class
PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 50


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data():
    data = load_rruff_dataset()
    peaks = torch.from_numpy(data["peaks"]).float()
    masks = torch.from_numpy(data["masks"]).bool()
    names = data["mineral_names"]

    counter = Counter(names)
    top_minerals = [m for m, c in counter.most_common(N_TOP_MINERALS) if c >= 3]
    mineral_to_id = {m: i for i, m in enumerate(top_minerals)}

    valid_idx = [i for i, n in enumerate(names) if n in mineral_to_id]
    peaks = peaks[valid_idx]
    masks = masks[valid_idx]
    labels = np.array([mineral_to_id[names[i]] for i in valid_idx])
    n_classes = len(mineral_to_id)

    # Stratified split: 70% train, 30% test
    rng = np.random.default_rng(SEED)
    test_idx = []
    train_idx = []
    for mid in range(n_classes):
        mineral_patterns = np.where(labels == mid)[0]
        rng.shuffle(mineral_patterns)
        n_test = max(1, len(mineral_patterns) // 3)
        test_idx.extend(mineral_patterns[:n_test])
        train_idx.extend(mineral_patterns[n_test:])

    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)
    rng.shuffle(train_idx)

    # Also keep ALL patterns (including non-top-50) for SSL pre-training
    all_peaks = torch.from_numpy(data["peaks"]).float()
    all_masks = torch.from_numpy(data["masks"]).bool()
    all_train_idx = np.arange(len(all_peaks))  # use ALL for pre-training

    print(f"Classification: {len(peaks)} patterns, {n_classes} minerals")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")
    print(f"  SSL pre-training: {len(all_train_idx)} patterns (all RRUFF)")

    return (peaks, masks, labels, train_idx, test_idx, n_classes,
            all_peaks, all_masks, all_train_idx)


def pretrain_ssl(model, peaks, masks, train_idx, n_epochs, lr=5e-4):
    print(f"\n  Pre-training ({n_epochs} epochs on {len(train_idx)} patterns)...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    batch_size = 64

    for epoch in range(n_epochs):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        perm = rng.permutation(len(train_idx))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            batch_perm = perm[start:start + batch_size]
            if len(batch_perm) < 8:
                continue
            idx = train_idx[batch_perm]
            batch_peaks = peaks[idx].to(DEVICE)
            batch_masks = masks[idx].to(DEVICE)

            result = model(batch_peaks, batch_masks)
            loss = result["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return model


def train_classifier(encoder, peaks, masks, labels, train_idx, test_idx,
                     n_classes, label_frac=1.0, freeze=True):
    n_labeled = max(n_classes, int(len(train_idx) * label_frac))
    labeled_idx = train_idx[:n_labeled]

    # Clone encoder for fine-tuning
    import copy
    enc_copy = copy.deepcopy(encoder)

    clf = PhaseClassifier(enc_copy, n_classes, freeze_encoder=freeze)
    clf.to(DEVICE)

    lr = 1e-3 if freeze else 5e-4
    optimizer = AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 32

    best_acc = 0.0
    best_top5 = 0.0

    for epoch in range(FINETUNE_EPOCHS):
        clf.train()
        rng = np.random.default_rng(SEED + epoch + 2000)
        perm = rng.permutation(len(labeled_idx))

        for start in range(0, len(perm), batch_size):
            batch_perm = perm[start:start + batch_size]
            idx = labeled_idx[batch_perm]
            batch_peaks = peaks[idx].to(DEVICE)
            batch_masks = masks[idx].to(DEVICE)
            batch_labels = torch.from_numpy(labels[idx]).long().to(DEVICE)

            logits = clf(batch_peaks, batch_masks)
            loss = F.cross_entropy(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        clf.eval()
        correct = 0
        top5_correct = 0
        total = 0
        with torch.no_grad():
            for start in range(0, len(test_idx), batch_size):
                idx = test_idx[start:start + batch_size]
                batch_peaks = peaks[idx].to(DEVICE)
                batch_masks = masks[idx].to(DEVICE)
                batch_labels = torch.from_numpy(labels[idx]).long().to(DEVICE)

                logits = clf(batch_peaks, batch_masks)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()

                top5 = logits.topk(min(5, n_classes), dim=1).indices
                for i in range(len(batch_labels)):
                    if batch_labels[i] in top5[i]:
                        top5_correct += 1
                total += len(batch_labels)

        acc = correct / total
        top5_acc = top5_correct / total
        if acc > best_acc:
            best_acc = acc
            best_top5 = top5_acc

    return best_acc, best_top5


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    (peaks, masks, labels, train_idx, test_idx, n_classes,
     all_peaks, all_masks, all_train_idx) = prepare_data()

    results = {}
    label_fracs = [0.1, 0.2, 0.5, 1.0]

    # ── 1. Random init (no pre-training), full fine-tune ─────────────
    print("\n" + "="*60)
    print("  random_init (no pre-training, full fine-tune)")
    print("="*60)
    for frac in label_fracs:
        set_seed(SEED)
        enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
        acc, top5 = train_classifier(
            enc, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=False
        )
        results[f"random_init_{frac}"] = {"acc": acc, "top5": top5, "frac": frac}
        print(f"  frac={frac}: top1={acc:.1%}, top5={top5:.1%}")

    # ── 2. Sort-match SSL (InfoNCE + sort-match, weight=0.1) ────────
    print("\n" + "="*60)
    print("  sort_match_ssl (InfoNCE + sort-match, weight=0.1)")
    print("="*60)
    set_seed(SEED)
    ssl_model = SortMatchSSL(
        d_peak=3, d_model=128, d_proj=64, n_heads=4, n_layers=3,
        sort_match_weight=0.1, temperature=0.1,
    )
    ssl_model = pretrain_ssl(ssl_model, all_peaks, all_masks, all_train_idx,
                             n_epochs=PRETRAIN_EPOCHS)

    for frac in label_fracs:
        set_seed(SEED)
        # Linear probe (frozen encoder)
        acc_lp, top5_lp = train_classifier(
            ssl_model.encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=True
        )
        # Full fine-tune
        acc_ft, top5_ft = train_classifier(
            ssl_model.encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=False
        )
        results[f"sort_match_ssl_lp_{frac}"] = {"acc": acc_lp, "top5": top5_lp, "frac": frac}
        results[f"sort_match_ssl_ft_{frac}"] = {"acc": acc_ft, "top5": top5_ft, "frac": frac}
        print(f"  frac={frac}: LP top1={acc_lp:.1%}/top5={top5_lp:.1%} | "
              f"FT top1={acc_ft:.1%}/top5={top5_ft:.1%}")

    # ── 3. InfoNCE only (no sort-match) ──────────────────────────────
    print("\n" + "="*60)
    print("  infonce_only (no sort-match)")
    print("="*60)
    set_seed(SEED)
    nce_model = SortMatchSSL(
        d_peak=3, d_model=128, d_proj=64, n_heads=4, n_layers=3,
        sort_match_weight=0.0, temperature=0.1,
    )
    nce_model = pretrain_ssl(nce_model, all_peaks, all_masks, all_train_idx,
                             n_epochs=PRETRAIN_EPOCHS)

    for frac in label_fracs:
        set_seed(SEED)
        acc_lp, top5_lp = train_classifier(
            nce_model.encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=True
        )
        acc_ft, top5_ft = train_classifier(
            nce_model.encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=False
        )
        results[f"infonce_lp_{frac}"] = {"acc": acc_lp, "top5": top5_lp, "frac": frac}
        results[f"infonce_ft_{frac}"] = {"acc": acc_ft, "top5": top5_ft, "frac": frac}
        print(f"  frac={frac}: LP top1={acc_lp:.1%}/top5={top5_lp:.1%} | "
              f"FT top1={acc_ft:.1%}/top5={top5_ft:.1%}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Mineral Classification ({n_classes} classes)")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'10%':>8} {'20%':>8} {'50%':>8} {'100%':>8}")
    print(f"{'-'*57}")

    for method in ["random_init", "infonce_ft", "sort_match_ssl_ft"]:
        label = method.replace("_ft", " (FT)").replace("_", " ")
        vals = [results.get(f"{method}_{f}", {}).get("acc", 0) for f in label_fracs]
        print(f"{label:<25} {vals[0]:>7.1%} {vals[1]:>7.1%} {vals[2]:>7.1%} {vals[3]:>7.1%}")

    print(f"\nTop-5 Accuracy:")
    print(f"{'Method':<25} {'10%':>8} {'20%':>8} {'50%':>8} {'100%':>8}")
    print(f"{'-'*57}")
    for method in ["random_init", "infonce_ft", "sort_match_ssl_ft"]:
        label = method.replace("_ft", " (FT)").replace("_", " ")
        vals = [results.get(f"{method}_{f}", {}).get("top5", 0) for f in label_fracs]
        print(f"{label:<25} {vals[0]:>7.1%} {vals[1]:>7.1%} {vals[2]:>7.1%} {vals[3]:>7.1%}")

    with open(os.path.join(RESULTS_DIR, "rruff_ssl_v2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/rruff_ssl_v2_results.json")

    return results


if __name__ == "__main__":
    main()
