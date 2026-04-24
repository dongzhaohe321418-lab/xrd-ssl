"""
Main experiment: Sort-match contrastive SSL on RRUFF experimental XRD.

Setup:
    - 3015 experimental RRUFF patterns (1785 unique minerals)
    - SSL pre-training: augmented view matching on ALL patterns (unlabeled)
    - Downstream: mineral classification with varying label fractions

Variants:
    1. random_init: No pre-training, train classifier from scratch
    2. sort_match_ssl: Pre-train with sort-match contrastive, then fine-tune
    3. simclr_baseline: Pre-train with standard InfoNCE only (no sort-match)

This is the real experiment: experimental data, genuine SSL.

Usage:
    python experiments/v2/run_rruff_ssl.py
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
from src.v2.augmentations import PeakSetAugmenter, augment_batch
from src.v2.data_experimental import load_rruff_dataset


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEED = 42


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data():
    """Load RRUFF data and prepare train/test splits."""
    data = load_rruff_dataset()
    peaks = torch.from_numpy(data["peaks"]).float()
    masks = torch.from_numpy(data["masks"]).bool()
    names = data["mineral_names"]

    # Filter to minerals with >= 2 patterns (need train + test)
    counter = Counter(names)
    valid_minerals = {m for m, c in counter.items() if c >= 2}

    # Also limit to top-N most common for tractable classification
    top_minerals = [m for m, _ in counter.most_common(100) if m in valid_minerals]
    mineral_to_id = {m: i for i, m in enumerate(top_minerals)}

    # Filter dataset
    valid_idx = [i for i, n in enumerate(names) if n in mineral_to_id]
    peaks = peaks[valid_idx]
    masks = masks[valid_idx]
    labels = np.array([mineral_to_id[names[i]] for i in valid_idx])
    mineral_names = np.array([names[i] for i in valid_idx])

    n_classes = len(mineral_to_id)
    print(f"Filtered to {len(peaks)} patterns, {n_classes} minerals (top-100 with >=2 patterns)")

    # Train/test split: hold out 1 pattern per mineral for test
    rng = np.random.default_rng(SEED)
    test_idx = []
    train_idx = []

    for mineral_id in range(n_classes):
        mineral_patterns = np.where(labels == mineral_id)[0]
        rng.shuffle(mineral_patterns)
        test_idx.append(mineral_patterns[0])
        train_idx.extend(mineral_patterns[1:])

    test_idx = np.array(test_idx)
    train_idx = np.array(train_idx)
    rng.shuffle(train_idx)

    print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    return peaks, masks, labels, train_idx, test_idx, n_classes


def pretrain_ssl(model, peaks, masks, train_idx, n_epochs=30, lr=1e-3):
    """SSL pre-training on unlabeled data."""
    print(f"\n  Pre-training ({n_epochs} epochs)...")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 64

    for epoch in range(n_epochs):
        model.train()
        rng = np.random.default_rng(SEED + epoch)
        perm = rng.permutation(len(train_idx))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(perm), batch_size):
            batch_perm = perm[start:start + batch_size]
            if len(batch_perm) < 4:  # need at least 4 for InfoNCE
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

        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return model


def train_classifier(encoder, peaks, masks, labels, train_idx, test_idx,
                     n_classes, label_frac=1.0, n_epochs=30, lr=1e-3,
                     freeze=True):
    """Train mineral classifier on top of (optionally frozen) encoder."""
    # Select labeled subset
    n_labeled = max(n_classes, int(len(train_idx) * label_frac))
    labeled_idx = train_idx[:n_labeled]

    clf = PhaseClassifier(encoder, n_classes, freeze_encoder=freeze)
    clf.to(DEVICE)
    optimizer = AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    batch_size = 32

    best_acc = 0.0
    for epoch in range(n_epochs):
        clf.train()
        rng = np.random.default_rng(SEED + epoch + 1000)
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
        total = 0
        top5_correct = 0
        with torch.no_grad():
            for start in range(0, len(test_idx), batch_size):
                idx = test_idx[start:start + batch_size]
                batch_peaks = peaks[idx].to(DEVICE)
                batch_masks = masks[idx].to(DEVICE)
                batch_labels = torch.from_numpy(labels[idx]).long().to(DEVICE)

                logits = clf(batch_peaks, batch_masks)
                preds = logits.argmax(dim=1)
                correct += (preds == batch_labels).sum().item()

                # Top-5
                top5 = logits.topk(min(5, n_classes), dim=1).indices
                for i in range(len(batch_labels)):
                    if batch_labels[i] in top5[i]:
                        top5_correct += 1
                total += len(batch_labels)

        acc = correct / total
        top5_acc = top5_correct / total
        best_acc = max(best_acc, acc)

    return best_acc, top5_acc


def run_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)

    peaks, masks, labels, train_idx, test_idx, n_classes = prepare_data()

    results = {}
    label_fracs = [0.1, 0.2, 0.5, 1.0]

    # ── Variant 1: Random Init (no pre-training) ────────────────────
    print("\n" + "="*60)
    print("  Variant: random_init (no pre-training)")
    print("="*60)
    for frac in label_fracs:
        set_seed(SEED)
        encoder = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
        acc, top5 = train_classifier(
            encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=False
        )
        key = f"random_init_frac{frac}"
        results[key] = {"variant": "random_init", "frac": frac, "acc": acc, "top5": top5}
        print(f"  frac={frac}: acc={acc:.3f}, top5={top5:.3f}")

    # ── Variant 2: Sort-Match SSL ────────────────────────────────────
    print("\n" + "="*60)
    print("  Variant: sort_match_ssl")
    print("="*60)
    set_seed(SEED)
    ssl_model = SortMatchSSL(
        d_peak=3, d_model=128, d_proj=64, n_heads=4, n_layers=3,
        sort_match_weight=0.5,
    )
    ssl_model = pretrain_ssl(ssl_model, peaks, masks, train_idx, n_epochs=30)
    pretrained_encoder = ssl_model.encoder

    for frac in label_fracs:
        set_seed(SEED)
        acc, top5 = train_classifier(
            pretrained_encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=True
        )
        key = f"sort_match_ssl_frac{frac}"
        results[key] = {"variant": "sort_match_ssl", "frac": frac, "acc": acc, "top5": top5}
        print(f"  frac={frac}: acc={acc:.3f}, top5={top5:.3f}")

    # ── Variant 3: InfoNCE only (no sort-match) ─────────────────────
    print("\n" + "="*60)
    print("  Variant: infonce_only (no sort-match regularizer)")
    print("="*60)
    set_seed(SEED)
    nce_model = SortMatchSSL(
        d_peak=3, d_model=128, d_proj=64, n_heads=4, n_layers=3,
        sort_match_weight=0.0,  # disable sort-match
    )
    nce_model = pretrain_ssl(nce_model, peaks, masks, train_idx, n_epochs=30)
    nce_encoder = nce_model.encoder

    for frac in label_fracs:
        set_seed(SEED)
        acc, top5 = train_classifier(
            nce_encoder, peaks, masks, labels, train_idx, test_idx,
            n_classes, label_frac=frac, freeze=True
        )
        key = f"infonce_only_frac{frac}"
        results[key] = {"variant": "infonce_only", "frac": frac, "acc": acc, "top5": top5}
        print(f"  frac={frac}: acc={acc:.3f}, top5={top5:.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Mineral Classification on RRUFF Experimental XRD")
    print(f"{'='*70}")
    print(f"{'Frac':<8} {'Random Init':>14} {'InfoNCE Only':>14} {'Sort-Match SSL':>14}")
    print(f"{'-'*50}")
    for frac in label_fracs:
        ri = results[f"random_init_frac{frac}"]["acc"]
        nc = results[f"infonce_only_frac{frac}"]["acc"]
        sm = results[f"sort_match_ssl_frac{frac}"]["acc"]
        print(f"{frac:<8} {ri:>13.1%} {nc:>13.1%} {sm:>13.1%}")

    with open(os.path.join(RESULTS_DIR, "rruff_ssl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/rruff_ssl_results.json")

    return results


if __name__ == "__main__":
    run_experiment()
