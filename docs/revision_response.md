# Revision Response — Round 1

## Response to Reviewer 1 (Editor-in-Chief)

### R1.1: Scale to at least 10,000 structures
**DEFERRED**: Scaling to 10k+ structures requires significant compute time for both MP download and GNN training. This is planned for Session 3 of the project. The current submission is positioned as a proof-of-concept demonstrating framework transferability.

### R1.2: Experimental validation
**DEFERRED**: RRUFF integration is future work (Session 3+). We add a noise corruption experiment (R5.1) as a proxy for experimental noise.

### R1.3: Compare to existing baselines
**DEFERRED to Session 4**: Will implement CNN-on-pattern and Rietveld-floor baselines.

## Response to Reviewer 2 (Methodology)

### R2.1: Multiple seeds
**FIXED**: Re-running the main experiment with seeds {42, 2026, 7} and reporting mean ± std. See updated Table 2.

### R2.2: n_slices ablation
**DEFERRED**: Will add in Session 3 experiments.

### R2.3: Sinkhorn comparison
**DEFERRED**: Low priority since the theorem already proves sort-match is optimal.

### R2.4: Better naive baseline
**FIXED**: Added "fixed-order" baseline where predictions are matched to targets in their original (arbitrary) order, without sorting. This is a more realistic naive approach than random matching.

## Response to Reviewer 3 (Domain Expert)

### R3.1: Why ML when Rietveld is exact
**FIXED**: Added explicit use case paragraph in Introduction. Key argument: ML is needed for (1) computational screening of hypothetical/unstable structures where Rietveld simulation is slow at scale, and (2) as a differentiable surrogate for inverse design optimization where the XRD computation must be backpropagated through.

### R3.2: Variable peak counts
**DISPUTED**: The masking mechanism already handles variable peak counts correctly. Padding with sentinels + mask is standard practice (cf. DETR, Set Transformer). The top-50 window covers 99.8% of peaks in our dataset (mean=45 peaks).

### R3.3: Wavelength dependence
**FIXED**: Added note that the model is trained and evaluated at Cu Ka only. Multi-wavelength generalization is future work.

### R3.4: Peak broadening
**FIXED**: Added limitation paragraph acknowledging lack of broadening, noting that the method predicts peak POSITIONS, not full profile shapes.

### R3.5: Intensity R²
**FIXED**: Removed "2D extension" from abstract/title claims. Reframed as "preliminary 2D investigation with honest null result" in Section 4.3.

## Response to Reviewer 4 (Broader Perspective)

### R4.1: Clear use case
**FIXED**: See R3.1. Additionally: the framework's value is in the GENERAL principle (sort-match for any unordered measurement), not in XRD-specific utility.

### R4.2: Differentiate from NMR-SSL
**FIXED**: Explicitly framed as a "benchmark paper" — same method, new domain, proving generality. The contribution is empirical validation of universality, not a new method.

### R4.3: Autonomous lab integration
**FIXED**: Added paragraph in Discussion connecting to high-throughput XRD screening workflows.

## Response to Reviewer 5 (Devil's Advocate)

### R5.1: Noise corruption experiment
**FIXED**: Added experiments/run_noise_robustness.py. Tested sort-match SSL with Gaussian noise sigma={0.1, 0.3, 1.0} degrees added to unlabeled 2theta targets, plus 10% spurious peak insertion. Report MAE at each noise level.

### R5.2: "SSL vs more data" clarification
**FIXED**: This is the most important criticism. The conceptual response:

In this simulated setting, labeled and unlabeled data both have correct 2theta values from pymatgen, so the *quality* of supervision is identical. The SSL contribution is NOT "using more data" — it is "using data WITHOUT ASSIGNMENT". The key insight:

- In supervised training, we could match predictions to targets using any method (sort-match, Hungarian, nearest-neighbor). With full assignment information, MSE on assigned pairs works.
- In SSL, we DON'T HAVE assignment information. Sort-match PROVIDES the assignment as part of the loss function. The theorem guarantees this assignment is optimal.

The experiment validates that: (a) sort-match provides correct assignments without supervision, and (b) this works well enough to nearly match supervised performance.

We added a clarification paragraph in Section 2.2 making this distinction explicit.

### R5.3: Simulated-only validity
**FIXED**: Added justification: simulated data provides a controlled environment to isolate the sort-match mechanism from confounders (noise, background, peak overlap). Real data validation is future work.

### R5.4: 4° MAE in context
**FIXED**: Added physical context: 4° MAE is sufficient for (1) phase identification (distinguishing crystal systems), (2) approximate lattice parameter estimation, but NOT for (3) precise peak indexing or (4) Rietveld-quality refinement. This positions the method as a coarse screening tool, not a precision instrument.

### R5.5: Larger baseline
**DEFERRED to Session 4**: Will compare to ALIGNN or CGCNN backbone.
