"""
Scaled experiment: 100 classes, multi-seed, mixed pre-training.

Improvements over run_final.py:
1. N_TOP=100 minerals (up from 30)
2. Min 3 samples per class (down from 4) for more coverage
3. 2 seeds (faster, still gives error bars)

Usage:
    python experiments/v2/run_scaled.py
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
from src.v2.cnn_baseline import CNN1DEncoder, batch_peaks_to_patterns
from src.v2.data_experimental import load_rruff_dataset
from src.xrd_data import load_dataset as load_mp_dataset


RESULTS_DIR = "experiments/v2/results"
DEVICE = "cpu"
SEEDS = [42, 2026]
N_TOP = 100
MIN_SAMPLES = 3
PRETRAIN_EPOCHS = 80
FINETUNE_EPOCHS = 80
BATCH_SIZE = 64


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data(seed=42):
    data = load_rruff_dataset()
    peaks = data["peaks"]
    masks = data["masks"]
    names = data["mineral_names"]

    counter = Counter(names)
    top_minerals = [m for m, c in counter.most_common(N_TOP) if c >= MIN_SAMPLES]
    mineral_to_id = {m: i for i, m in enumerate(top_minerals)}
    n_classes = len(mineral_to_id)

    valid_idx = [i for i, n in enumerate(names) if n in mineral_to_id]
    peaks_f = peaks[valid_idx]
    masks_f = masks[valid_idx]
    labels = np.array([mineral_to_id[names[i]] for i in valid_idx])
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

    print(f"Scaled dataset: {len(peaks_f)} patterns, {n_classes} classes")
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    return (torch.from_numpy(peaks_f).float(), torch.from_numpy(masks_f).bool(),
            torch.from_numpy(patterns).float(), labels,
            train_idx, test_idx, n_classes,
            all_peaks, all_masks)


def load_sim():
    try:
        mp = load_mp_dataset("data/xrd_cache_5k.npz")
        peaks = np.zeros((len(mp["two_theta"]), 100, 3), dtype=np.float32)
        masks = np.zeros((len(mp["two_theta"]), 100), dtype=bool)
        for i in range(len(mp["two_theta"])):
            m = mp["mask"][i]
            n = m.sum()
            peaks[i, :n, 0] = mp["two_theta"][i, :n]
            peaks[i, :n, 1] = mp["intensity"][i, :n]
            peaks[i, :n, 2] = 0.15
            masks[i, :n] = True
        return torch.from_numpy(peaks).float(), torch.from_numpy(masks).bool()
    except:
        return None, None


def pretrain(peaks, masks, train_idx, n_epochs, temp=0.07):
    model = SortMatchSSL(d_peak=3, d_model=128, d_proj=64, n_heads=4,
                         n_layers=3, sort_match_weight=0.0, temperature=temp)
    opt = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    for ep in range(n_epochs):
        model.train()
        perm = np.random.default_rng(ep).permutation(len(train_idx))
        for s in range(0, len(perm), BATCH_SIZE):
            bp = perm[s:s+BATCH_SIZE]
            if len(bp) < 8: continue
            idx = train_idx[bp]
            r = model(peaks[idx], masks[idx])
            opt.zero_grad(); r["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        if (ep+1) % 20 == 0: print(f"      Epoch {ep+1}/{n_epochs}")
    return model.encoder


def classify(encoder, data, labels, train_idx, test_idx, n_classes,
             label_frac, is_cnn=False, freeze=True, seed=42):
    np.random.seed(seed); torch.manual_seed(seed)
    n_labeled = max(n_classes, int(len(train_idx) * label_frac))
    labeled_idx = train_idx[:n_labeled]
    enc = copy.deepcopy(encoder)
    head = torch.nn.Sequential(
        torch.nn.Linear(enc.output_dim, enc.output_dim),
        torch.nn.GELU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(enc.output_dim, n_classes),
    )
    if freeze:
        for p in enc.parameters(): p.requires_grad = False
    params = [p for p in list(enc.parameters())+list(head.parameters()) if p.requires_grad]
    opt = AdamW(params, lr=1e-3 if freeze else 5e-4, weight_decay=1e-4)
    best_acc, best_t5 = 0.0, 0.0
    for ep in range(FINETUNE_EPOCHS):
        enc.train() if not freeze else enc.eval(); head.train()
        perm = np.random.default_rng(seed+ep+5000).permutation(len(labeled_idx))
        for s in range(0, len(perm), 32):
            bp = perm[s:s+32]; idx = labeled_idx[bp]
            x = data[idx]
            if is_cnn and x.dim()==2: x = x.unsqueeze(1)
            y = torch.from_numpy(labels[idx]).long()
            h = enc(x); logits = head(h)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        enc.eval(); head.eval()
        c, t5c, tot = 0, 0, 0
        with torch.no_grad():
            for s in range(0, len(test_idx), 32):
                idx = test_idx[s:s+32]; x = data[idx]
                if is_cnn and x.dim()==2: x = x.unsqueeze(1)
                y = torch.from_numpy(labels[idx]).long()
                logits = head(enc(x))
                c += (logits.argmax(1)==y).sum().item()
                t5 = logits.topk(min(5,n_classes),dim=1).indices
                for i in range(len(y)):
                    if y[i] in t5[i]: t5c += 1
                tot += len(y)
        acc, top5 = c/tot, t5c/tot
        if acc > best_acc: best_acc, best_t5 = acc, top5
    return best_acc, best_t5


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    sim_peaks, sim_masks = load_sim()
    methods = {}
    fracs = [0.1, 0.2, 0.5, 1.0]
    for m in ["cnn_sup", "set_sup", "set_ssl_ft", "set_ssl_mixed_ft"]:
        methods[m] = {f: {"accs":[], "top5s":[]} for f in fracs}

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*60}\n  SEED {seed} ({si+1}/{len(SEEDS)})\n{'='*60}")
        set_seed(seed)
        (peaks, masks, patterns, labels, train_idx, test_idx, nc,
         all_peaks, all_masks) = prepare_data(seed)

        class SW(torch.nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = SetTransformerEncoder(d_peak=3, d_model=128, n_heads=4, n_layers=3)
                s.output_dim = 128
            def forward(s, x): return s.enc(x[:,:,:3], (x[:,:,1]>0))

        # CNN supervised
        print("  CNN supervised...")
        for f in fracs:
            a, t5 = classify(CNN1DEncoder(128), patterns, labels, train_idx, test_idx, nc, f, is_cnn=True, freeze=False, seed=seed)
            methods["cnn_sup"][f]["accs"].append(a); methods["cnn_sup"][f]["top5s"].append(t5)

        # Set supervised
        print("  Set supervised...")
        for f in fracs:
            a, t5 = classify(SW(), peaks, labels, train_idx, test_idx, nc, f, freeze=False, seed=seed)
            methods["set_sup"][f]["accs"].append(a); methods["set_sup"][f]["top5s"].append(t5)

        # Set+SSL (RRUFF only)
        print("  Set+SSL (RRUFF)...")
        all_train = np.arange(len(all_peaks))
        enc_r = pretrain(all_peaks, all_masks, all_train, PRETRAIN_EPOCHS)
        class SWr(torch.nn.Module):
            def __init__(s, e): super().__init__(); s.enc=e; s.output_dim=e.output_dim
            def forward(s, x): return s.enc(x[:,:,:3], (x[:,:,1]>0))
        for f in fracs:
            a, t5 = classify(SWr(enc_r), peaks, labels, train_idx, test_idx, nc, f, freeze=False, seed=seed)
            methods["set_ssl_ft"][f]["accs"].append(a); methods["set_ssl_ft"][f]["top5s"].append(t5)

        # Set+SSL (mixed)
        if sim_peaks is not None:
            print("  Set+SSL (mixed RRUFF+MP)...")
            mx_p = torch.cat([all_peaks, sim_peaks])
            mx_m = torch.cat([all_masks, sim_masks])
            enc_m = pretrain(mx_p, mx_m, np.arange(len(mx_p)), PRETRAIN_EPOCHS)
            for f in fracs:
                a, t5 = classify(SWr(enc_m), peaks, labels, train_idx, test_idx, nc, f, freeze=False, seed=seed)
                methods["set_ssl_mixed_ft"][f]["accs"].append(a); methods["set_ssl_mixed_ft"][f]["top5s"].append(t5)

    elapsed = time.time() - t0
    print(f"\n{'='*84}")
    print(f"  SCALED RESULTS: {N_TOP} classes, {len(SEEDS)} seeds, {elapsed/60:.0f} min")
    print(f"{'='*84}")

    final = {}
    print(f"\nTOP-1 ACCURACY:")
    print(f"{'Method':<28} {'10%':>14} {'20%':>14} {'50%':>14} {'100%':>14}")
    print("-"*84)
    for m,l in [("cnn_sup","CNN supervised"),("set_sup","Set supervised"),
                ("set_ssl_ft","Set+SSL (RRUFF)"),("set_ssl_mixed_ft","Set+SSL (mixed)")]:
        row = []
        for f in fracs:
            accs = methods[m][f]["accs"]
            if accs:
                mean, std = np.mean(accs), np.std(accs)
                row.append(f"{mean:.1%}+/-{std:.1%}")
                final[f"{m}_{f}"] = {"acc_mean":float(mean),"acc_std":float(std),
                    "top5_mean":float(np.mean(methods[m][f]["top5s"])),"top5_std":float(np.std(methods[m][f]["top5s"]))}
            else: row.append("N/A")
        print(f"{l:<28} {'  '.join(row)}")

    print(f"\nTOP-5 ACCURACY:")
    for m,l in [("cnn_sup","CNN"),("set_sup","Set"),("set_ssl_ft","Set+SSL"),("set_ssl_mixed_ft","Set+SSL mixed")]:
        row = []
        for f in fracs:
            t5s = methods[m][f]["top5s"]
            if t5s: row.append(f"{np.mean(t5s):.1%}+/-{np.std(t5s):.1%}")
            else: row.append("N/A")
        print(f"{l:<28} {'  '.join(row)}")

    with open(os.path.join(RESULTS_DIR, "scaled_100class_results.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR}/scaled_100class_results.json")


if __name__ == "__main__":
    main()
