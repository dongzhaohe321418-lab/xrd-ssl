"""
The key experiment: pre-train on 91K opXRD patterns, evaluate on RRUFF.

This tests whether large-scale SSL on unlabeled experimental data
produces better representations than small-scale (3K RRUFF) pre-training.

Variants:
1. No pre-training (random init)
2. RRUFF pre-train (3K patterns)
3. opXRD pre-train (91K patterns)
4. opXRD+RRUFF pre-train (94K patterns)

Usage:
    python experiments/v2/run_opxrd_pretrain.py
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
from src.v2.data_opxrd import load_opxrd_dataset


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEED = 42
N_TOP = 30
PRETRAIN_EPOCHS = 30  # fewer epochs but MUCH more data
FINETUNE_EPOCHS = 80
BATCH_SIZE = 128  # larger batch for 91K data


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def prepare_rruff_classification():
    data = load_rruff_dataset()
    peaks = data["peaks"]; masks = data["masks"]; names = data["mineral_names"]
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
        mp = np.where(labels == mid)[0]; rng.shuffle(mp)
        nt = max(1, len(mp)//3)
        test_idx.extend(mp[:nt]); train_idx.extend(mp[nt:])
    return (torch.from_numpy(peaks_f).float(), torch.from_numpy(masks_f).bool(),
            labels, np.array(train_idx), np.array(test_idx), nc)


def pretrain(peaks, masks, train_idx, n_epochs, batch_size=BATCH_SIZE, temp=0.07):
    model = SortMatchSSL(d_peak=3, d_model=128, d_proj=64, n_heads=4,
                         n_layers=3, sort_match_weight=0.0, temperature=temp)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(n_epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(train_idx))
        total_loss, n_b = 0, 0
        for s in range(0, len(perm), batch_size):
            bp = perm[s:s+batch_size]
            if len(bp) < 16: continue
            idx = train_idx[bp]
            r = model(peaks[idx], masks[idx])
            opt.zero_grad(); r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += r["loss"].item(); n_b += 1
        sched.step()
        if (ep+1) % 5 == 0:
            print(f"      Epoch {ep+1}/{n_epochs}: loss={total_loss/max(n_b,1):.4f} ({n_b} batches)")
    return model.encoder


def classify(encoder, data, labels, train_idx, test_idx, nc, label_frac, freeze=False):
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
            h = enc(x[:,:,:3], (x[:,:,1]>0)); logits = head(h)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval()
        c, t5c, tot = 0, 0, 0
        with torch.no_grad():
            for s in range(0, len(test_idx), 32):
                idx = test_idx[s:s+32]; x = data[idx]; y = torch.from_numpy(labels[idx]).long()
                logits = head(enc(x[:,:,:3], (x[:,:,1]>0)))
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

    peaks, masks, labels, train_idx, test_idx, nc = prepare_rruff_classification()
    print(f"RRUFF classification: {len(peaks)} patterns, {nc} classes")

    # Load pre-training data
    rruff = load_rruff_dataset()
    rruff_peaks = torch.from_numpy(rruff["peaks"]).float()
    rruff_masks = torch.from_numpy(rruff["masks"]).bool()

    opxrd = load_opxrd_dataset()
    opxrd_peaks = torch.from_numpy(opxrd["peaks"]).float()
    opxrd_masks = torch.from_numpy(opxrd["masks"]).bool()
    print(f"Pre-training pools: RRUFF={len(rruff_peaks)}, opXRD={len(opxrd_peaks)}")

    fracs = [0.1, 0.2, 0.5, 1.0]
    results = {}

    # 1. Random init
    print("\n  1. Random init...")
    enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
    for f in fracs:
        a, t5 = classify(enc, peaks, labels, train_idx, test_idx, nc, f)
        results[f"random_{f}"] = {"acc": a, "top5": t5}
        print(f"    frac={f}: top1={a:.1%}, top5={t5:.1%}")

    # 2. RRUFF pre-train (3K)
    print("\n  2. RRUFF pre-train (3K)...")
    enc_rruff = pretrain(rruff_peaks, rruff_masks, np.arange(len(rruff_peaks)), PRETRAIN_EPOCHS)
    for f in fracs:
        a, t5 = classify(enc_rruff, peaks, labels, train_idx, test_idx, nc, f)
        results[f"rruff_3k_{f}"] = {"acc": a, "top5": t5}
        print(f"    frac={f}: top1={a:.1%}, top5={t5:.1%}")

    # 3. opXRD pre-train (91K)
    print("\n  3. opXRD pre-train (91K)...")
    enc_opxrd = pretrain(opxrd_peaks, opxrd_masks, np.arange(len(opxrd_peaks)), PRETRAIN_EPOCHS)
    for f in fracs:
        a, t5 = classify(enc_opxrd, peaks, labels, train_idx, test_idx, nc, f)
        results[f"opxrd_91k_{f}"] = {"acc": a, "top5": t5}
        print(f"    frac={f}: top1={a:.1%}, top5={t5:.1%}")

    # 4. opXRD+RRUFF pre-train (94K)
    print("\n  4. opXRD+RRUFF pre-train (94K)...")
    all_p = torch.cat([opxrd_peaks, rruff_peaks])
    all_m = torch.cat([opxrd_masks, rruff_masks])
    enc_all = pretrain(all_p, all_m, np.arange(len(all_p)), PRETRAIN_EPOCHS)
    for f in fracs:
        a, t5 = classify(enc_all, peaks, labels, train_idx, test_idx, nc, f)
        results[f"all_94k_{f}"] = {"acc": a, "top5": t5}
        print(f"    frac={f}: top1={a:.1%}, top5={t5:.1%}")

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  opXRD SCALING RESULTS ({elapsed/60:.0f} min)")
    print(f"{'='*70}")
    print(f"{'Method':<30} {'10%':>8} {'20%':>8} {'50%':>8} {'100%':>8}")
    print("-"*62)
    for m,l in [("random","Random init"),("rruff_3k","RRUFF 3K"),("opxrd_91k","opXRD 91K"),("all_94k","All 94K")]:
        vals = [results.get(f"{m}_{f}",{}).get("acc",0) for f in fracs]
        print(f"{l:<30} {'  '.join(f'{v:>6.1%}' for v in vals)}")

    print(f"\nTOP-5:")
    for m,l in [("random","Random"),("rruff_3k","RRUFF 3K"),("opxrd_91k","opXRD 91K"),("all_94k","All 94K")]:
        vals = [results.get(f"{m}_{f}",{}).get("top5",0) for f in fracs]
        print(f"{l:<30} {'  '.join(f'{v:>6.1%}' for v in vals)}")

    with open(os.path.join(RESULTS_DIR, "opxrd_scaling_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/opxrd_scaling_results.json")


if __name__ == "__main__":
    main()
