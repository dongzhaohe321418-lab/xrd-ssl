"""
Cross-domain transfer: pre-train on XRD, evaluate on Raman.

Tests whether Set Transformer representations transfer across
spectroscopic domains — the key generality argument.

Setup:
1. Pre-train Set+SSL on XRD peaks (RRUFF)
2. Fine-tune on Raman peaks (same minerals, different physics)
3. Compare: XRD-pretrained vs random init on Raman classification

If XRD pre-training helps Raman classification, the representations
capture mineral-level structure, not just XRD-specific features.

Usage:
    python experiments/v2/run_cross_domain.py
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
from src.v2.contrastive_ssl import SortMatchSSL
from src.v2.data_experimental import load_rruff_dataset
from src.v2.data_raman import load_rruff_raman_from_xrd_zip


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEED = 42
N_TOP = 30
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 80
BATCH_SIZE = 64


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def prepare_raman_data():
    raman = load_rruff_raman_from_xrd_zip()
    peaks = raman["peaks"]
    masks = raman["masks"]
    names = raman["mineral_names"]

    counter = Counter(names)
    top = [m for m, c in counter.most_common(N_TOP) if c >= 4]
    m2id = {m: i for i, m in enumerate(top)}
    nc = len(m2id)

    vi = [i for i, n in enumerate(names) if n in m2id]
    peaks_f = peaks[vi]; masks_f = masks[vi]
    labels = np.array([m2id[names[i]] for i in vi])

    rng = np.random.default_rng(SEED)
    test_idx, train_idx = [], []
    for mid in range(nc):
        mp = np.where(labels == mid)[0]
        rng.shuffle(mp)
        nt = max(1, len(mp)//3)
        test_idx.extend(mp[:nt]); train_idx.extend(mp[nt:])

    return (torch.from_numpy(peaks_f).float(), torch.from_numpy(masks_f).bool(),
            labels, np.array(train_idx), np.array(test_idx), nc)


def pretrain_on_xrd():
    """Pre-train Set Transformer on XRD peak sets."""
    xrd = load_rruff_dataset()
    peaks = torch.from_numpy(xrd["peaks"]).float()
    masks = torch.from_numpy(xrd["masks"]).bool()
    all_idx = np.arange(len(peaks))

    model = SortMatchSSL(d_peak=3, d_model=128, d_proj=64, n_heads=4,
                         n_layers=3, sort_match_weight=0.0, temperature=0.07)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PRETRAIN_EPOCHS)

    print("  Pre-training on XRD...")
    for ep in range(PRETRAIN_EPOCHS):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(all_idx))
        for s in range(0, len(perm), BATCH_SIZE):
            bp = perm[s:s+BATCH_SIZE]
            if len(bp) < 8: continue
            r = model(peaks[all_idx[bp]], masks[all_idx[bp]])
            opt.zero_grad(); r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep+1) % 20 == 0: print(f"    Epoch {ep+1}/{PRETRAIN_EPOCHS}")

    return model.encoder


def classify(encoder, data, labels, train_idx, test_idx, nc, label_frac, freeze=True):
    set_seed(SEED)
    n_labeled = max(nc, int(len(train_idx)*label_frac))
    labeled_idx = train_idx[:n_labeled]
    enc = copy.deepcopy(encoder)
    head = torch.nn.Sequential(
        torch.nn.Linear(enc.output_dim, enc.output_dim),
        torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(enc.output_dim, nc),
    )
    if freeze:
        for p in enc.parameters(): p.requires_grad = False
    params = [p for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad]
    opt = AdamW(params, lr=1e-3 if freeze else 5e-4, weight_decay=1e-4)
    best_acc, best_t5 = 0.0, 0.0
    for ep in range(FINETUNE_EPOCHS):
        enc.train() if not freeze else enc.eval(); head.train()
        perm = np.random.default_rng(SEED+ep).permutation(len(labeled_idx))
        for s in range(0, len(perm), 32):
            bp = perm[s:s+32]; idx = labeled_idx[bp]
            x = data[idx]; y = torch.from_numpy(labels[idx]).long()

            class W(torch.nn.Module):
                def __init__(s2, e): super().__init__(); s2.enc=e; s2.output_dim=e.output_dim
                def forward(s2, x): return s2.enc(x[:,:,:3], (x[:,:,1]>0))

            h = enc(x[:,:,:3], (x[:,:,1]>0)) if hasattr(enc,'d_model') else enc(x)
            logits = head(h)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval()
        c, t5c, tot = 0, 0, 0
        with torch.no_grad():
            for s in range(0, len(test_idx), 32):
                idx = test_idx[s:s+32]
                x = data[idx]; y = torch.from_numpy(labels[idx]).long()
                h = enc(x[:,:,:3], (x[:,:,1]>0)) if hasattr(enc,'d_model') else enc(x)
                logits = head(h)
                c += (logits.argmax(1)==y).sum().item()
                t5 = logits.topk(min(5,nc),dim=1).indices
                for i in range(len(y)):
                    if y[i] in t5[i]: t5c += 1
                tot += len(y)
        acc, top5 = c/tot, t5c/tot
        if acc > best_acc: best_acc, best_t5 = acc, top5
    return best_acc, best_t5


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)
    t0 = time.time()

    peaks, masks, labels, train_idx, test_idx, nc = prepare_raman_data()
    print(f"Raman classification: {len(peaks)} patterns, {nc} classes")

    fracs = [0.1, 0.2, 0.5, 1.0]
    results = {}

    # 1. Random init on Raman
    print("\n  Random init (no pre-training)...")
    for f in fracs:
        enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
        a, t5 = classify(enc, peaks, labels, train_idx, test_idx, nc, f, freeze=False)
        results[f"raman_random_{f}"] = {"acc": a, "top5": t5}
        print(f"    frac={f}: top1={a:.1%}, top5={t5:.1%}")

    # 2. XRD-pretrained, fine-tune on Raman
    xrd_enc = pretrain_on_xrd()
    print("\n  XRD-pretrained -> Raman fine-tune...")
    for f in fracs:
        a, t5 = classify(xrd_enc, peaks, labels, train_idx, test_idx, nc, f, freeze=False)
        results[f"raman_xrd_pretrained_{f}"] = {"acc": a, "top5": t5}
        print(f"    frac={f}: top1={a:.1%}, top5={t5:.1%}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  CROSS-DOMAIN TRANSFER: XRD -> Raman ({elapsed/60:.0f} min)")
    print(f"{'='*60}")
    print(f"{'Method':<30} {'10%':>8} {'20%':>8} {'50%':>8} {'100%':>8}")
    print("-"*62)
    for m, l in [("raman_random","Random init"), ("raman_xrd_pretrained","XRD pretrained")]:
        vals = [results.get(f"{m}_{f}", {}).get("acc", 0) for f in fracs]
        print(f"{l:<30} {'  '.join(f'{v:>6.1%}' for v in vals)}")

    with open(os.path.join(RESULTS_DIR, "cross_domain_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/cross_domain_results.json")


if __name__ == "__main__":
    main()
