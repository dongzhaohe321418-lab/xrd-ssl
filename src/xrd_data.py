"""
XRD data pipeline: Materials Project download + pymatgen XRD simulation.

Downloads crystal structures from Materials Project, simulates powder XRD
patterns with pymatgen's XRDCalculator (Cu Ka, lambda=1.54 A), and extracts
top-50 peaks by intensity in 2theta in [5, 90] degrees.

Physics:
    XRD peak positions are determined by Bragg's law: n*lambda = 2*d*sin(theta),
    where d-spacings come from the crystal lattice. Peak intensities depend on
    atomic scattering factors, Lorentz-polarization correction, and structure
    factors. pymatgen handles all of this.

Usage:
    python src/xrd_data.py --n_structures 5000 --output data/xrd_cache.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


# ── Constants ────────────────────────────────────────────────────────

TOP_K_PEAKS = 50
TWO_THETA_MIN = 5.0
TWO_THETA_MAX = 90.0
WAVELENGTH = "CuKa"  # Cu K-alpha, 1.5406 A
MAX_ATOMS = 30        # keep structures small for GNN tractability
MAX_ENERGY = 0.1      # eV/atom above hull (near-stable structures)


def download_structures(
    n_structures: int = 5000,
    cache_dir: str = "data/mp_structures",
    chemical_systems: List[str] = None,
) -> List[Dict]:
    """Download structures from Materials Project.

    Filters:
        - Formation energy < 0.1 eV/atom above hull
        - <= 30 atoms in unit cell
        - Ordered (no partial occupancy)

    The search returns structures directly (no separate CIF download needed).
    This is 100x faster than per-ID downloads.

    Args:
        n_structures: max structures to download.
        cache_dir: directory for caching.
        chemical_systems: list of required elements, e.g. ["O"] for oxides.
            If None, downloads across all chemistries.
    """
    from mp_api.client import MPRester

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set MP_API_KEY environment variable. "
            "Get one at https://next-gen.materialsproject.org/api"
        )

    cache_path = Path(cache_dir) / "structures.json"
    cif_dir = Path(cache_dir) / "cifs"

    # Check cache first
    if cache_path.exists():
        print(f"Loading cached structures from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Count how many CIFs we actually have
        existing_cifs = sum(1 for e in cached if (cif_dir / f"{e['material_id']}.cif").exists())
        print(f"  {len(cached)} entries, {existing_cifs} CIFs on disk")
        if existing_cifs >= n_structures:
            return [e for e in cached if (cif_dir / f"{e['material_id']}.cif").exists()][:n_structures]

    print(f"Querying Materials Project for up to {n_structures} structures...")
    search_kwargs = dict(
        energy_above_hull=(0, MAX_ENERGY),
        num_sites=(1, MAX_ATOMS),
        fields=[
            "material_id",
            "structure",
            "formula_pretty",
            "symmetry",
            "nsites",
            "energy_above_hull",
        ],
    )
    if chemical_systems:
        search_kwargs["elements"] = chemical_systems

    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(**search_kwargs)

    # Filter and save — structures come directly from search, no per-ID call
    os.makedirs(cif_dir, exist_ok=True)
    results = []
    print(f"Processing {len(docs)} candidates...")
    for doc in tqdm(docs):
        if not doc.structure.is_ordered:
            continue

        mp_id = str(doc.material_id)
        cif_path = cif_dir / f"{mp_id}.cif"

        # Save CIF directly from the structure object
        if not cif_path.exists():
            try:
                doc.structure.to(filename=str(cif_path))
            except Exception as e:
                continue

        results.append({
            "material_id": mp_id,
            "formula": doc.formula_pretty,
            "spacegroup": doc.symmetry.symbol if doc.symmetry else "unknown",
            "crystal_system": doc.symmetry.crystal_system.value if doc.symmetry else "unknown",
            "nsites": doc.nsites,
        })
        if len(results) >= n_structures:
            break

    # Cache metadata
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Downloaded {len(results)} structures ({len(results)} CIFs saved)")
    return results


def simulate_xrd_pattern(
    cif_path: str,
    top_k: int = TOP_K_PEAKS,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Simulate XRD pattern from a CIF file.

    Returns:
        two_theta: (top_k,) array of peak positions in degrees.
        intensity: (top_k,) array of intensities normalized to max=100.
        n_real: number of real peaks (before padding).
    """
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    structure = Structure.from_file(cif_path)
    calc = XRDCalculator(wavelength=WAVELENGTH, symprec=0.1)
    pattern = calc.get_pattern(
        structure, scaled=True, two_theta_range=(TWO_THETA_MIN, TWO_THETA_MAX)
    )

    # Extract peaks: pattern.x = 2theta, pattern.y = intensity
    positions = np.array(pattern.x, dtype=np.float32)
    intensities = np.array(pattern.y, dtype=np.float32)

    # Sort by intensity descending, take top_k
    order = np.argsort(-intensities)
    positions = positions[order]
    intensities = intensities[order]

    n_real = min(len(positions), top_k)

    # Pad to top_k with sentinel (0, 0)
    two_theta = np.zeros(top_k, dtype=np.float32)
    intensity = np.zeros(top_k, dtype=np.float32)
    two_theta[:n_real] = positions[:n_real]
    intensity[:n_real] = intensities[:n_real]

    return two_theta, intensity, n_real


def build_dataset(
    n_structures: int = 5000,
    output_path: str = "data/xrd_cache.npz",
    cache_dir: str = "data/mp_structures",
) -> Dict[str, np.ndarray]:
    """Build the full XRD dataset: download + simulate + save.

    Output arrays in the .npz file:
        material_ids: (N,) string array
        two_theta: (N, 50) float32, peak positions in degrees
        intensity: (N, 50) float32, normalized intensities (max=100)
        mask: (N, 50) bool, True = real peak, False = padding
    """
    entries = download_structures(n_structures, cache_dir)

    cif_dir = Path(cache_dir) / "cifs"
    all_ids = []
    all_two_theta = []
    all_intensity = []
    all_mask = []
    failed = []

    print(f"Simulating XRD for {len(entries)} structures...")
    for entry in tqdm(entries):
        mp_id = entry["material_id"]
        cif_path = cif_dir / f"{mp_id}.cif"

        if not cif_path.exists():
            failed.append((mp_id, "CIF not found"))
            continue

        try:
            two_theta, intensity, n_real = simulate_xrd_pattern(str(cif_path))
            mask = np.zeros(TOP_K_PEAKS, dtype=bool)
            mask[:n_real] = True

            all_ids.append(mp_id)
            all_two_theta.append(two_theta)
            all_intensity.append(intensity)
            all_mask.append(mask)
        except Exception as e:
            failed.append((mp_id, str(e)))

    # Convert to arrays
    material_ids = np.array(all_ids)
    two_theta = np.stack(all_two_theta)
    intensity = np.stack(all_intensity)
    mask = np.stack(all_mask)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        material_ids=material_ids,
        two_theta=two_theta,
        intensity=intensity,
        mask=mask,
    )

    # Statistics
    n_peaks_per_struct = mask.sum(axis=1)
    print(f"\nDataset saved to {output_path}")
    print(f"  Structures: {len(material_ids)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Peaks per structure: mean={n_peaks_per_struct.mean():.1f}, "
          f"min={n_peaks_per_struct.min()}, max={n_peaks_per_struct.max()}")
    print(f"  2theta range: [{two_theta[mask].min():.2f}, {two_theta[mask].max():.2f}]")
    print(f"  Intensity range: [{intensity[mask].min():.2f}, {intensity[mask].max():.2f}]")

    if failed:
        fail_path = os.path.join(os.path.dirname(output_path), "xrd_failures.json")
        with open(fail_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"  Failure log: {fail_path}")

    return {
        "material_ids": material_ids,
        "two_theta": two_theta,
        "intensity": intensity,
        "mask": mask,
    }


def load_dataset(path: str = "data/xrd_cache.npz") -> Dict[str, np.ndarray]:
    """Load a previously built dataset."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build XRD dataset from Materials Project")
    parser.add_argument("--n_structures", type=int, default=5000)
    parser.add_argument("--output", type=str, default="data/xrd_cache.npz")
    parser.add_argument("--cache_dir", type=str, default="data/mp_structures")
    args = parser.parse_args()

    build_dataset(args.n_structures, args.output, args.cache_dir)
