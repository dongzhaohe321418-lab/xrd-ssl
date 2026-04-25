"""
opXRD data loader: 92K experimental XRD patterns for large-scale SSL.

The opXRD dataset contains diverse experimental XRD patterns from 8
institutions. Most patterns are unlabeled — perfect for SSL pre-training.

Since opXRD uses diverse instrument formats, we parse the raw files
by looking for simple two-column (2theta, intensity) text patterns
or using the opxrd library if available.
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
from tqdm import tqdm

from src.v2.data_experimental import extract_peaks, MAX_PEAKS, PEAK_FEATURES


def load_opxrd_dataset(
    zip_path: str = "data/opxrd.zip",
    cache_path: str = "data/opxrd_peaks.npz",
    max_patterns: int = None,
) -> Dict[str, np.ndarray]:
    """Load opXRD patterns and extract peaks.

    Attempts to parse each file as a two-column text file.
    Files that fail to parse are skipped.

    Returns dict with:
        peaks: (N, MAX_PEAKS, 3) peak features
        masks: (N, MAX_PEAKS) boolean masks
        filenames: (N,) source filenames
    """
    if os.path.exists(cache_path):
        print(f"Loading cached opXRD peaks from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    if not os.path.exists(zip_path):
        print(f"opXRD zip not found at {zip_path}")
        print("Download from: https://zenodo.org/records/15298026/files/opxrd.zip")
        return None

    print(f"Extracting peaks from opXRD ({zip_path})...")
    all_peaks = []
    all_n_peaks = []
    all_files = []
    failed = 0

    import json as json_mod

    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [f for f in zf.namelist() if f.lower().endswith('.json')]
        print(f"  Found {len(candidates)} JSON files")

        for fname in tqdm(candidates[:max_patterns] if max_patterns else candidates,
                          desc="Processing opXRD"):
            try:
                with zf.open(fname) as f:
                    data = json_mod.load(f)

                two_theta = data.get("two_theta_values", [])
                intensity = data.get("intensities", [])

                if len(two_theta) < 50:
                    failed += 1
                    continue

                tt = np.array(two_theta, dtype=np.float32)
                ii = np.array(intensity, dtype=np.float32)

                # Filter to reasonable XRD range
                valid = (tt > 3) & (tt < 120) & (ii >= 0)
                tt = tt[valid]
                ii = ii[valid]

                if len(tt) < 50:
                    failed += 1
                    continue

                peaks, n_peaks = extract_peaks(tt, ii)
                if n_peaks >= 3:
                    all_peaks.append(peaks)
                    all_n_peaks.append(n_peaks)
                    all_files.append(fname)
                else:
                    failed += 1

            except Exception:
                failed += 1

    if not all_peaks:
        print(f"No valid patterns found! ({failed} failed)")
        return None

    peaks_arr = np.stack(all_peaks)
    masks_arr = np.zeros((len(all_peaks), MAX_PEAKS), dtype=bool)
    for i, n in enumerate(all_n_peaks):
        masks_arr[i, :n] = True

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        peaks=peaks_arr,
        masks=masks_arr,
        filenames=np.array(all_files),
    )

    print(f"  {len(all_peaks)} patterns with >= 3 peaks")
    print(f"  {failed} files failed to parse")
    print(f"  Peaks per pattern: mean={np.mean(all_n_peaks):.1f}")
    print(f"  Saved to {cache_path}")

    return {"peaks": peaks_arr, "masks": masks_arr, "filenames": np.array(all_files)}
