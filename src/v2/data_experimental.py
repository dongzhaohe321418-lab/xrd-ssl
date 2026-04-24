"""
Data pipeline for experimental XRD patterns.

Handles:
1. RRUFF powder XRD data (labeled, clean XY files)
2. opXRD patterns (mostly unlabeled, diverse formats)

Peak extraction from raw XRD patterns using scipy.signal.find_peaks.

Each pattern is converted to a peak set: variable-length list of
(2theta, relative_intensity, estimated_FWHM) tuples, padded to max_peaks.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import gaussian_filter1d


MAX_PEAKS = 100  # max peaks per pattern
PEAK_FEATURES = 3  # 2theta, intensity, FWHM


def load_rruff_xy(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a RRUFF XY file (2theta, intensity).

    RRUFF XY files have a header comment block starting with ##,
    followed by two-column numeric data.
    """
    two_theta = []
    intensity = []

    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("!"):
                continue
            parts = line.split(",") if "," in line else line.split()
            if len(parts) >= 2:
                try:
                    tt = float(parts[0])
                    ii = float(parts[1])
                    two_theta.append(tt)
                    intensity.append(ii)
                except ValueError:
                    continue

    return np.array(two_theta, dtype=np.float32), np.array(intensity, dtype=np.float32)


def extract_peaks(
    two_theta: np.ndarray,
    intensity: np.ndarray,
    max_peaks: int = MAX_PEAKS,
    min_height_frac: float = 0.02,
    smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """Extract peaks from a raw XRD pattern.

    Args:
        two_theta: raw 2theta values
        intensity: raw intensity values
        max_peaks: maximum peaks to extract
        min_height_frac: minimum peak height as fraction of max intensity
        smooth_sigma: Gaussian smoothing sigma (in data points)

    Returns:
        peaks: (max_peaks, 3) array of [2theta, rel_intensity, est_FWHM]
        n_peaks: number of real peaks (rest are padding)
    """
    if len(two_theta) < 10 or intensity.max() <= 0:
        return np.zeros((max_peaks, PEAK_FEATURES), dtype=np.float32), 0

    # Smooth for peak detection
    smoothed = gaussian_filter1d(intensity, sigma=smooth_sigma)

    # Normalize to [0, 100]
    max_I = smoothed.max()
    if max_I > 0:
        smoothed_norm = smoothed / max_I * 100
    else:
        return np.zeros((max_peaks, PEAK_FEATURES), dtype=np.float32), 0

    # Find peaks
    min_height = min_height_frac * 100  # as fraction of normalized max
    peak_indices, properties = find_peaks(
        smoothed_norm,
        height=min_height,
        distance=3,  # minimum distance between peaks in data points
        prominence=min_height * 0.5,
    )

    if len(peak_indices) == 0:
        return np.zeros((max_peaks, PEAK_FEATURES), dtype=np.float32), 0

    # Estimate FWHM
    try:
        widths_result = peak_widths(smoothed_norm, peak_indices, rel_height=0.5)
        fwhm_points = widths_result[0]  # width in data points
        # Convert to degrees
        d2theta = np.median(np.diff(two_theta)) if len(two_theta) > 1 else 0.02
        fwhm_degrees = fwhm_points * d2theta
    except Exception:
        fwhm_degrees = np.full(len(peak_indices), 0.2)  # default FWHM

    # Sort by intensity (descending), take top max_peaks
    peak_intensities = smoothed_norm[peak_indices]
    order = np.argsort(-peak_intensities)
    peak_indices = peak_indices[order[:max_peaks]]
    peak_intensities = peak_intensities[order[:max_peaks]]
    fwhm_degrees = fwhm_degrees[order[:max_peaks]]

    n_peaks = len(peak_indices)

    # Build peak array
    peaks = np.zeros((max_peaks, PEAK_FEATURES), dtype=np.float32)
    peaks[:n_peaks, 0] = two_theta[peak_indices]  # 2theta
    peaks[:n_peaks, 1] = peak_intensities          # relative intensity
    peaks[:n_peaks, 2] = fwhm_degrees[:n_peaks]    # FWHM in degrees

    return peaks, n_peaks


def load_rruff_dataset(
    zip_path: str = "data/rruff_raw.zip",
    cache_path: str = "data/rruff_peaks.npz",
    max_peaks: int = MAX_PEAKS,
) -> Dict[str, np.ndarray]:
    """Load RRUFF dataset, extract peaks, cache as .npz.

    Returns dict with:
        peaks: (N, max_peaks, 3) peak features
        masks: (N, max_peaks) boolean masks
        mineral_names: (N,) mineral names (labels)
        filenames: (N,) source filenames
    """
    if os.path.exists(cache_path):
        print(f"Loading cached RRUFF peaks from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    print(f"Extracting peaks from RRUFF data ({zip_path})...")
    all_peaks = []
    all_n_peaks = []
    all_names = []
    all_files = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        xy_files = [f for f in zf.namelist()
                     if f.lower().endswith(".txt") and not f.startswith("__")]

        print(f"  Found {len(xy_files)} XY files")

        for fname in xy_files:
            # Extract mineral name from filename
            # RRUFF format: MineralName__RXXXXX__Powder__RAW__*.txt
            basename = os.path.basename(fname)
            parts = basename.split("__")
            if len(parts) >= 1:
                mineral = parts[0]
            else:
                mineral = "unknown"

            try:
                # Read from zip
                with zf.open(fname) as f:
                    content = f.read().decode("utf-8", errors="ignore")

                # Parse XY data
                two_theta = []
                intensity = []
                for line in content.strip().split("\n"):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("!"):
                        continue
                    parts_line = line.split(",") if "," in line else line.split()
                    if len(parts_line) >= 2:
                        try:
                            two_theta.append(float(parts_line[0]))
                            intensity.append(float(parts_line[1]))
                        except ValueError:
                            continue

                if len(two_theta) < 20:
                    continue

                tt = np.array(two_theta, dtype=np.float32)
                ii = np.array(intensity, dtype=np.float32)

                peaks, n_peaks = extract_peaks(tt, ii, max_peaks=max_peaks)

                if n_peaks >= 3:  # need at least 3 peaks
                    all_peaks.append(peaks)
                    all_n_peaks.append(n_peaks)
                    all_names.append(mineral)
                    all_files.append(fname)

            except Exception:
                continue

    if not all_peaks:
        raise RuntimeError(f"No valid patterns found in {zip_path}")

    # Stack arrays
    peaks_arr = np.stack(all_peaks)  # (N, max_peaks, 3)
    masks_arr = np.zeros((len(all_peaks), max_peaks), dtype=bool)
    for i, n in enumerate(all_n_peaks):
        masks_arr[i, :n] = True

    names_arr = np.array(all_names)
    files_arr = np.array(all_files)

    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(
        cache_path,
        peaks=peaks_arr,
        masks=masks_arr,
        mineral_names=names_arr,
        filenames=files_arr,
    )

    print(f"  {len(all_peaks)} patterns with >= 3 peaks")
    print(f"  Peaks per pattern: mean={np.mean(all_n_peaks):.1f}, "
          f"min={min(all_n_peaks)}, max={max(all_n_peaks)}")
    print(f"  Unique minerals: {len(set(all_names))}")
    print(f"  Saved to {cache_path}")

    return {
        "peaks": peaks_arr,
        "masks": masks_arr,
        "mineral_names": names_arr,
        "filenames": files_arr,
    }


class XRDPeakDataset:
    """PyTorch-compatible dataset for XRD peak sets."""

    def __init__(self, peaks: np.ndarray, masks: np.ndarray,
                 labels: Optional[np.ndarray] = None):
        self.peaks = torch.from_numpy(peaks).float()
        self.masks = torch.from_numpy(masks).bool()
        self.labels = labels  # string array or None

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        item = {"peaks": self.peaks[idx], "mask": self.masks[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item
