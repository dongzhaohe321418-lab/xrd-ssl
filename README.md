# XRD-SSL: Sort-Match Self-Supervision for Powder XRD Peak Prediction

Extends the [NMR-SSL](https://arxiv.org/abs/2601.18524) sort-match framework
to powder X-ray diffraction (XRD) peak prediction from crystal structures.

## Key Results

Sort-match SSL with **10% labeled data** matches fully supervised performance:

| Variant | Labels | MAE (degrees 2theta) |
|---------|--------|---------------------|
| Supervised | 100% | 4.13 +/- 0.23 |
| **Sort-match SSL** | **10%** | **4.09 +/- 0.31** |
| Supervised | 10% | 7.38 +/- 0.40 |

At **2% labels**, SSL still achieves **4.23 degrees** while supervised collapses
to **11.92 degrees** (+65% improvement).

## Method

1. **Input**: Crystal structure (atomic coordinates + lattice)
2. **Model**: 4-layer GIN, 257k parameters (no torch_geometric)
3. **Output**: Set of 50 predicted peak positions {2theta_i}
4. **Loss**: Sort-match — sort predictions and targets independently,
   then L1. Proved optimal for 1D convex costs (O(n log n) vs O(n^3)).
5. **Data**: 5,000 Materials Project structures, pymatgen XRD simulation

## Quick Start

```bash
pip install -r requirements.txt
export MP_API_KEY='your_key'

# 1. Verify theorem (31/31 tests pass)
python3 -m pytest tests/ -v

# 2. Download data + simulate XRD
python3 src/xrd_data.py --n_structures 1000

# 3. Run main experiment
python3 experiments/sanity_check.py

# 4. Run low-label ablation (key result)
python3 experiments/run_low_label.py

# 5. Generate all figures
python3 figures/compile_all.py
```

## Project Structure

```
xrd-ssl/
  src/
    losses.py       -- 1D sort-match + masked variant + Hungarian reference
    losses_2d.py    -- 2D sliced-Wasserstein loss
    features.py     -- crystal graph construction (atoms, bonds, RBF)
    model.py        -- 4-layer GIN + optional intensity head
    xrd_data.py     -- Materials Project download + pymatgen XRD simulation
  tests/
    test_theorem.py -- 22 tests: sort-match == Hungarian to <1e-10
    test_2d.py      -- 9 tests: 2D counterexample, SW convergence
  experiments/
    sanity_check.py     -- 4-variant main comparison
    run_2d.py           -- 2D position+intensity experiment
    run_multiseed.py    -- 3-seed statistical validation
    run_low_label.py    -- label fraction sweep (key result)
    run_noise_robustness.py -- noise corruption experiment
    run_hungarian_benchmark.py -- runtime comparison
    failure_analysis.py -- worst-case structure analysis
  docs/
    preprint_v2.md   -- paper draft (post peer review)
    theorem_2d.md    -- 2D counterexample + SW justification
    review_round1.md -- simulated 5-reviewer peer review
    review_round2.md -- revision review (Minor Revision)
    lit-review.md    -- literature survey
  figures/           -- publication-quality PDF+PNG
```

## Key Findings

- **Theorem transfers**: Sort-match (NMR) works for XRD without modification
- **Low-label regime**: SSL advantage grows from +14% (50% labels) to +65% (2% labels)
- **Noise robust**: Tolerates 1-degree Gaussian noise (MAE +0.04), sensitive to spurious peaks (+29%)
- **73x faster**: Sort-match vs Hungarian at n=50 peaks
- **Intensity**: Not learned at current scale (R^2 near 0) — honest limitation
- **2D counterexample**: No exact 2D sort-match exists (formal proof)

## Citation

```bibtex
@article{dong2026xrdssl,
  title={Sort-Match Self-Supervision for Powder XRD Peak Prediction},
  author={Dong, Eric},
  year={2026},
  note={Preprint in preparation}
}
```

## Acknowledgments

Built with [Claude Code](https://claude.ai/claude-code) under human oversight.
Extends the NMR-SSL framework (arXiv: 2601.18524).
