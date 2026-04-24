# XRD-SSL: Sort-Match Self-Supervision for Powder XRD Peak Prediction

Extends the [NMR-SSL](https://arxiv.org/abs/2601.18524) sort-match framework
to powder X-ray diffraction (XRD) peak prediction from crystal structures.

## Key Claim

A GNN trained with **sort-match self-supervised loss** on unassigned XRD peak
lists achieves position MAE within **[TBD]** of fully supervised training,
using only **10%** labeled data. The 1D sort-match theorem (proved optimal for
convex costs) transfers directly from NMR chemical shifts to XRD 2theta
positions.

## Method

1. **Input**: Crystal structure (atomic coordinates + lattice)
2. **Model**: 4-layer GIN with manual message passing (no torch_geometric)
3. **Output**: Set of 50 predicted peak positions {2theta_i}
4. **Loss**: Sort-match loss -- sort predictions and targets independently,
   then compute L1. Proved equivalent to Hungarian matching for 1D convex
   costs (O(n log n) vs O(n^3)).

## Project Structure

```
xrd-ssl/
  src/
    losses.py       -- sort-match + masked variant + Hungarian reference
    features.py     -- crystal graph construction (atoms, bonds, RBF)
    model.py        -- 4-layer GIN encoder + peak prediction head
    xrd_data.py     -- Materials Project download + pymatgen XRD simulation
  tests/
    test_theorem.py -- numerical verification: sort-match == Hungarian
  experiments/
    sanity_check.py -- 3-variant comparison (supervised/random/sort-match)
    make_figures.py -- training curves + scatter plots
  docs/
    theorem_2d.md   -- 2D extension via sliced-Wasserstein (Session 2)
  figures/          -- generated plots (PDF+PNG)
  data/             -- cached structures and XRD patterns (gitignored)
```

## Quick Start

```bash
pip install -r requirements.txt
export MP_API_KEY='your_key'

# 1. Verify theorem
pytest tests/test_theorem.py -v

# 2. Download data + simulate XRD
python src/xrd_data.py --n_structures 1000

# 3. Run sanity experiment
python experiments/sanity_check.py

# 4. Generate figures
python experiments/make_figures.py
```

## Results (Session 1)

| Variant | Test MAE (degrees 2theta) |
|---------|--------------------------|
| Supervised | [TBD] |
| Sort-match SSL (10% labels) | [TBD] |
| Random-match | [TBD] |

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
