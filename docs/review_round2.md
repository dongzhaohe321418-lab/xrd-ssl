# Simulated Peer Review — Round 2

## Editorial Summary

Manuscript: "Sort-Match Self-Supervision for Powder XRD Peak Prediction: A Benchmark for Framework Generality" (Revised)

The revision substantively addresses the most critical concerns from Round 1. Three reviewers now recommend Accept/Minor Revision. Two maintain Major Revision, primarily on the dataset scale issue.

---

## Reviewer 1 — Editor-in-Chief

### Assessment of Revision
The authors addressed my concerns partially:
- **R1.1 (Scale)**: NOT ADDRESSED. Still 1,000 structures. This remains the primary barrier to publication at npj Comp Mat. The Matbench community standard is 10k+.
- **R1.2 (Experimental)**: Partially addressed via noise robustness experiment, which is a reasonable proxy but not a substitute.
- **R1.3 (Baselines)**: NOT ADDRESSED.

### New Strengths
1. Multi-seed experiments (Table 1) greatly strengthen the main claim
2. Noise robustness is an excellent addition
3. Honest framing as "benchmark paper" is appropriate

### Decision: **Minor Revision** (was Major)

### Remaining Required Changes
- R1.1': Scale experiment is the only remaining blocker for npj. At minimum, show results on 5,000 structures.
- Otherwise, the paper is acceptable for JCIM as-is.

---

## Reviewer 2 — Methodology

### Assessment of Revision
All my concerns were addressed:
- R2.1: Multi-seed done (3 seeds, mean +/- std reported). Satisfied.
- R2.2: n_slices ablation deferred but acceptable given the null result on 2D.
- R2.3: Sinkhorn deferred. Acceptable.
- R2.4: Random match baseline is still present but the text now explains why sort-match is fundamentally different from "more data" (Section 5.1). This is a much better discussion.

### New Strengths
1. The "Is this SSL or just more data?" discussion (Section 5.1) is excellent
2. Statistical rigor greatly improved

### Decision: **Accept**

---

## Reviewer 3 — Domain Expert

### Assessment of Revision
- R3.1: Use case clearly stated. The "differentiable surrogate" argument is compelling.
- R3.2: Variable peak count acknowledged; masking handles it. Acceptable.
- R3.3: Wavelength note added. Fine.
- R3.4: Broadening limitation acknowledged. Fine.
- R3.5: 2D claims appropriately qualified. The negative R^2 is no longer oversold.

### New Concern
The "Physical Context" section (4.4) is helpful but raises a question: if 4-degree MAE is only useful for "coarse screening," what fraction of practical materials discovery problems actually need coarse screening vs. precision matching? This should be discussed briefly.

### Decision: **Minor Revision**

---

## Reviewer 4 — Broader Perspective

### Assessment of Revision
- R4.1: Use case well-articulated. Satisfied.
- R4.2: "Benchmark paper" framing is honest and appropriate. The contribution is empirical, not methodological, and the paper now owns this.
- R4.3: The "Integration with High-Throughput Workflows" paragraph (Section 5.4) is good.

### Remaining Concern
The paper would be significantly stronger with one more domain demonstration — even a toy example on Raman or IR spectroscopy — to truly establish "domain-agnostic" framework status. Two domains (NMR + XRD) is suggestive; three would be convincing.

### Decision: **Minor Revision**

---

## Reviewer 5 — Devil's Advocate

### Assessment of Revision
The revision significantly improved the paper. My key concerns:

- R5.1: Noise robustness experiment is excellent. The finding that sort-match tolerates 1° Gaussian noise but fails on spurious peaks is a genuine, useful insight.
- R5.2: The "SSL vs more data" discussion in Section 5.1 is satisfactory. The conceptual distinction between "data quality" and "assignment information" is clear.
- R5.3: Simulated-only justification is acceptable for a proof-of-concept.
- R5.4: Physical context added and appropriate.
- R5.5: Larger baseline comparison still missing.

### New Observations
1. The multi-seed result (4.09 +/- 0.31 vs 4.13 +/- 0.23) shows no significant difference between SSL and supervised. This could mean: (a) sort-match is perfect, or (b) with 1,000 structures the model hasn't learned much beyond trivial features. A learning curve (performance vs dataset size) would disambiguate.
2. The claim "matching 98% of supervised" (from v1) is now "no significant difference" (from v2) — which is actually a STRONGER claim. Good.

### Decision: **Minor Revision** (was Major)

### Remaining Required Changes
- R5.5': At minimum, discuss why a 257k-parameter GIN was chosen over larger models and what performance improvement is expected from scaling.

---

## Editorial Decision: MINOR REVISION

### Synthesis

The revised manuscript substantially addresses Round 1 concerns. The key improvements:
1. Multi-seed experiments with error bars (addresses statistical rigor)
2. Noise robustness characterization (addresses practical relevance)
3. Honest "benchmark paper" framing (addresses incrementality concern)
4. "SSL vs more data" discussion (addresses conceptual clarity)

### Remaining Items for Final Revision
1. **(Optional for npj, Required for JCIM)**: Brief discussion of model scaling expectations (R5.5')
2. **(Optional)**: Discuss what fraction of materials discovery needs "coarse screening" (R3)
3. **(Strongly recommended)**: Add 1 paragraph on learning curve (performance vs dataset size) to distinguish "SSL works" from "model saturates at 1,000 structures"

### Venue Assessment
| Venue | Probability of Acceptance |
|-------|--------------------------|
| npj Computational Materials | 30-40% (scale concern) |
| J. Chem. Inf. Model. (JCIM) | 65-75% (as-is after minor fixes) |
| Chemistry of Materials | 40-50% (needs experimental validation) |
| arXiv preprint | 100% |

**Recommendation**: Submit to JCIM as primary target. The benchmark paper framing fits JCIM's scope (informatics methods for chemical data). If the authors later scale to 20k structures and add RRUFF validation, upgrade to npj Comp Mat.
