# Simulated Peer Review — Round 1

## Editorial Summary

Manuscript: "Sort-Match Self-Supervision for Powder XRD Peak Prediction"

Five reviewers with distinct expertise evaluated this manuscript. The consensus is **Major Revision**. The core idea is sound and the 1D theorem transfer is clean, but several significant issues must be addressed before publication.

---

## Reviewer 1 — Editor-in-Chief (npj Computational Materials)

### Summary
The manuscript demonstrates transfer of a sort-match SSL framework from NMR to XRD peak position prediction. The mathematical foundation is elegant and the 98% result at 10% labels is impressive.

### Strengths
1. Clean mathematical framework with formal guarantees
2. Strong quantitative result: 10% labels → 98% of supervised performance
3. Honest reporting of null results (xrd-biased sampling, intensity R²)
4. Good positioning relative to prior work

### Weaknesses
1. **Scale is too small for publication**: 1,000 structures is an undergraduate project scale. Materials Project has 170k+ structures. The community standard (Matbench, CGCNN, ALIGNN papers) is 10k-130k structures. Reviewers at npj Comp Mat will dismiss this immediately.
2. **No experimental validation**: All XRD patterns are simulated. The paper claims relevance to "unassigned experimental peak lists" but never tests on one.
3. **No comparison to existing XRD-ML baselines**: DeepXRD, Mat2Spec, and SimXRD-4M are cited but not compared against.

### Decision: **Major Revision**

### Required Changes
- R1.1: Scale to at least 10,000 structures across multiple chemical systems
- R1.2: Include at least one experimental validation (RRUFF or ICSD)
- R1.3: Compare to at least one existing ML baseline (DeepXRD or a CNN on continuous patterns)

---

## Reviewer 2 — Methodology (ML/OT Theory Expert)

### Summary
Solid application of the 1D sort-match theorem to a new domain. The 2D extension via sliced-Wasserstein is standard but well-executed. Some concerns about experimental rigor.

### Strengths
1. Theorem verification is exemplary (22 tests, machine precision)
2. Honest treatment of 2D counterexample
3. Clean separation of 1D and 2D experiments

### Weaknesses
1. **Only 1 seed**: All experiments use seed=42. Without multiple seeds, the "98%" claim is a point estimate with unknown variance. A different seed could give a very different ratio.
2. **No Sinkhorn comparison**: The paper uses Hungarian as reference but doesn't compare to entropic optimal transport (Sinkhorn). For large n, Sinkhorn is the practical alternative, not Hungarian.
3. **Sliced-Wasserstein hyperparameters not ablated**: n_slices=128 is used without justification. How does performance change at 32 or 512?
4. **The "random match" baseline is too weak**: Nobody would actually train with random matching. A more informative ablation would compare sort-match to "nearest-neighbor matching" or "fixed-order matching" (always match the i-th prediction to the i-th target).

### Decision: **Minor Revision**

### Required Changes
- R2.1: Run all experiments with 3 seeds and report mean ± std
- R2.2: Add n_slices ablation for the 2D experiment
- R2.3: Consider adding a Sinkhorn OT baseline
- R2.4: Replace or supplement random-match with a more realistic naive baseline

---

## Reviewer 3 — Domain Expert (Crystallography/XRD)

### Summary
As a crystallographer, I have concerns about the physical model and the practical relevance of this work.

### Strengths
1. The mathematical framework is interesting
2. The paper is well-structured and clearly written

### Weaknesses
1. **Top-50 peaks is physically inappropriate**: The number of Bragg reflections in 2theta ∈ [5°, 90°] varies from ~5 (high-symmetry cubic) to 500+ (large low-symmetry cells). Using a fixed top-50 truncates high-complexity structures and wastes capacity on simple ones. A variable-length approach with masking (which the code already supports!) should be the default.
2. **Why Cu Kα?**: The choice is standard for laboratory XRD but should be explicitly justified. Synchrotron sources use different wavelengths. Is the model wavelength-specific?
3. **Rietveld gives zero error**: For any known crystal structure, pymatgen can compute the exact XRD pattern with zero error. The entire premise of "predicting XRD from structure" via ML is questionable — why approximate what can be computed exactly? The practical value would be predicting XRD for structures that are NOT yet synthesized (computational screening), but this use case is never clearly stated.
4. **No peak broadening**: Real XRD peaks are not delta functions. They have width due to instrumental broadening, crystallite size (Scherrer equation), and microstrain. The simulated data ignores all of this.
5. **Intensity R² is negative**: This means the model is worse than predicting the mean intensity. Reporting this honestly is good, but it undermines the "2D extension" claim.

### Decision: **Major Revision**

### Required Changes
- R3.1: Clearly state the use case: why ML when Rietveld is exact
- R3.2: Address variable peak counts properly (not just padding)
- R3.3: Discuss wavelength dependence
- R3.4: Acknowledge peak broadening limitation
- R3.5: Either demonstrate positive intensity R² or remove the "2D extension" from the title/abstract

---

## Reviewer 4 — Broader Perspective (Materials Informatics)

### Summary
This is the second instantiation of a general framework (after NMR). The generality argument is the main selling point, but the current evidence is thin.

### Strengths
1. The "framework transfer" narrative is compelling
2. Potential for broad impact across spectroscopy types

### Weaknesses
1. **Incremental over NMR-SSL**: The 1D sort-match code is literally the same (copy-paste from NMR-SSL with a different data loader). The theoretical contribution is zero — the theorem was already proved. The only new contribution is showing it works on a different dataset, which is empirical validation, not a new method.
2. **No practical use case articulated**: Who benefits from this? A materials scientist who has a crystal structure can get exact XRD in milliseconds via pymatgen. A materials scientist who has XRD but not the structure needs the INVERSE problem. The paper addresses neither practical need convincingly.
3. **Missing connection to autonomous labs**: The paper mentions high-throughput screening but doesn't connect to real workflows. How would this integrate with, say, an autonomous XRD characterization pipeline?

### Decision: **Major Revision**

### Required Changes
- R4.1: Articulate a clear, specific use case where ML XRD prediction from structure is needed despite Rietveld being available
- R4.2: Differentiate the contribution from NMR-SSL more clearly — if it's "same method, new domain," own it and frame it as a benchmark paper
- R4.3: Include a paragraph on integration with autonomous characterization workflows

---

## Reviewer 5 — Devil's Advocate

### Summary
I'm going to push on whether this result is real and meaningful.

### Strengths
1. The 98% ratio is eye-catching
2. Honest null results

### Weaknesses
1. **Data leakage concern**: The "unlabeled" data uses the same pymatgen simulation as the "labeled" data. In a real SSL scenario, labeled data would be expert-curated Rietveld refinements and unlabeled data would be raw experimental patterns with noise, peak overlap, and background. The current setup is "SSL on clean simulated data" — a much easier problem than real SSL.
2. **The 98% number is misleading**: Both sort-match SSL and supervised use the SAME loss function (sort-match) on the SAME data. The only difference is that supervised computes the loss on 800 samples while SSL splits into 80 labeled + 720 unlabeled with the same loss. Since the loss function is identical, this is just "more data helps" — not an SSL contribution. True SSL would use DIFFERENT loss functions for labeled vs unlabeled data (e.g., supervised = MSE with known assignments, SSL = sort-match without assignments).
3. **Why not just use more labeled data?**: If you have 800 simulated structures, you can simulate XRD for all of them in seconds (the paper says ~47 structures/sec). So why pretend only 80 have labels? In the simulated setting, there's no cost to labeling. The SSL paradigm only makes sense with EXPERIMENTAL data where labeling is genuinely expensive.
4. **The 4° MAE is not useful**: For any practical crystallographic application, a 4° error in 2theta is enormous. A cubic structure with a=5A has its (111) peak at 2theta=31.73° and (200) at 36.82° — a 5° difference. At 4° MAE, the model can't reliably distinguish adjacent peaks.
5. **Model is too small**: 257k params is deliberately kept small to meet an arbitrary <500k target. State-of-the-art crystal GNNs (ALIGNN, M3GNet, MACE) have millions of parameters and pre-trained weights. Comparing a 257k GIN to nothing is not a fair evaluation.

### Decision: **Major Revision**

### Required Changes
- R5.1: Address the data leakage / clean simulation concern: what happens when unlabeled data has noise added?
- R5.2: Clarify the conceptual distinction between "more data helps" and "SSL helps" in this setting
- R5.3: State why the simulated-only setting is scientifically valid as a proof-of-concept
- R5.4: Put the 4° MAE in physical context — what tasks is this accurate enough for?
- R5.5: Compare to at least one larger baseline model

---

## Editorial Decision: MAJOR REVISION

### Synthesis

The manuscript presents a mathematically clean framework with a strong empirical result on a small, simulated dataset. However, all five reviewers raised significant concerns:

**Must address:**
1. Scale (1,000 → 10,000+ structures) [R1, R4]
2. Multiple seeds for statistical rigor [R2]
3. Clear use case / motivation beyond "it works" [R3, R4, R5]
4. Address the "is this really SSL or just more data?" question [R5]
5. Put MAE in physical context [R3, R5]

**Should address:**
6. At least one experimental validation or noise-corruption experiment [R1, R5]
7. Comparison to existing baselines [R1, R5]
8. Remove or qualify the "2D extension" given negative R² [R3]

**Nice to have:**
9. Sinkhorn comparison [R2]
10. Variable peak count handling [R3]
11. Autonomous lab integration discussion [R4]

The paper has potential for publication at npj Computational Materials or JCIM after addressing items 1-5 thoroughly and items 6-8 partially. In its current form, it reads as a promising preliminary study rather than a complete contribution.
