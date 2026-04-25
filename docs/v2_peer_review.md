# Peer Review: "Set-Based SSL for Experimental Powder XRD"

## Is this meaningful and innovative? — Honest Assessment

---

### REVIEWER A — ML Methods (NeurIPS/ICML scope)

**Innovation assessment: MODERATE**

The core claim is: "treating XRD as peak sets (Set Transformer) instead of 1D spectra (CNN) gives better representations, and SSL helps."

**What's genuinely new:**
- No prior work encodes XRD as discrete peak sets for classification. XCCP (2026), SimCLR-XRD (2025) all operate on full spectra. This is a real contribution.
- The physical argument (peaks are Bragg reflections = unordered set) is well-motivated and gives the representation a defensible inductive bias.

**What's NOT new:**
- Set Transformer is from 2019 (Lee et al.). InfoNCE is from 2018. Combining them is engineering, not methodology.
- The augmentations (noise, dropout, broadening) are standard in SSL literature.
- The sort-match component — which was supposed to be the novel loss — didn't work.

**Verdict:** The contribution is **empirical, not methodological**. It's a strong empirical study showing that set-based inductive bias matters for XRD. publishable at a workshop (NeurIPS ML4Science) or applications venue (Digital Discovery), but not a top ML venue.

**Rating: 5/10 for innovation, 7/10 for execution**

---

### REVIEWER B — Crystallography/Materials Science

**Meaningfulness assessment: HIGH for the field**

**Why this matters:**
1. XRD mineral identification is a real, daily task in materials labs. Current ML approaches use full-pattern matching against reference databases (PDF-4+, ICSD). This requires manual preprocessing and expert judgment.
2. The ~70% top-1 accuracy with SSL is useful for automated pre-screening. The 88% top-5 means the correct mineral is almost always in the shortlist.
3. The demonstration on REAL experimental data (RRUFF, not simulated) is critical. This immediately makes it more credible than the SimXRD-4M benchmark.

**Concerns:**
1. **30 classes is too small.** Real-world mineral identification involves 5,000+ known mineral species. The method needs to be tested at larger scale.
2. **~350 total patterns** is small. But SSL pre-training uses all 3,015 RRUFF patterns, which is reasonable.
3. **No multi-phase.** Most real-world XRD patterns are mixtures. Single-phase classification on RRUFF is a simplified task.

**What would make this truly impactful:**
- Scale to 100+ classes
- Test on opXRD (92K patterns, mostly unlabeled)
- Attempt multi-phase decomposition
- Benchmark against existing search/match software (X'Pert HighScore, MATCH!)

**Rating: 7/10 for meaningfulness, 6/10 for completeness**

---

### REVIEWER C — Devil's Advocate

**Is this really innovative, or is it just "Set Transformer + InfoNCE on XRD data"?**

**The uncomfortable truth:**
Yes, at its core, this is applying existing ML components (Set Transformer, InfoNCE) to a new data type (XRD peaks). The innovation is in the **combination and the argument for why sets are the right representation**, not in any new algorithm.

**However:**
Many impactful papers in applied ML are exactly this: taking a method from domain A and showing it works in domain B with proper motivation. The key question is whether the insight is non-obvious.

**Is "XRD peaks are sets" non-obvious?**
Partially. To a crystallographer, peaks are always indexed (hkl) — they're NOT unordered. But from the ML perspective, when you DON'T know the indexing, treating them as an unordered set and letting attention learn the relationships is a smart design choice. The fact that it outperforms CNN by 30+ percentage points suggests this is more than a trivial observation.

**The sort-match failure is actually interesting:**
The fact that sort-match regularization doesn't help is a negative result worth reporting. It suggests that for contrastive learning, global pattern similarity (InfoNCE) is more useful than peak-level reconstruction. This has implications for other scientific SSL applications.

**Rating: 6/10 for innovation**

---

### SYNTHESIS

**Is this a meaningful and innovative piece of work?**

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Novelty of the idea | 6/10 | Set representation for XRD is new; components are not |
| Quality of execution | 8/10 | Proper baselines, experimental data, honest reporting |
| Significance for the field | 7/10 | Real problem, real data, practical implications |
| Technical depth | 5/10 | No new algorithms, standard SSL pipeline |
| Completeness | 6/10 | Needs more classes, multi-phase, larger baselines |

**Overall: 6.5/10 — Publishable at an applications venue (Digital Discovery, JCIM), not at a top methods venue (NeurIPS, ICML)**

### What would push this to 8+/10:

1. **Scale to opXRD** (92K patterns) — the pre-training pool jumps from 3K to 92K. This is the single most impactful improvement.

2. **Multi-phase decomposition** — show that set-based representations can decompose mixture XRD patterns into constituent phases. This would be genuinely novel (no prior work does this with sets).

3. **Theoretical analysis** — prove that Set Transformer with peak-set inputs is a universal approximator for XRD pattern matching, or derive generalization bounds based on the set-matching structure. This would add methodological depth.

4. **Cross-domain transfer** — pre-train on XRD, transfer to Raman or IR spectroscopy (both also have peak-set structure). This would establish the generality claimed in the abstract.

### Honest bottom line:

The work is solid empirical science. The key insight (sets > spectra for XRD) is validated convincingly. The SSL component works well. It's not a breakthrough, but it's a useful contribution that would be welcomed by the experimental XRD community. The paper should lean into being an **empirical contribution** rather than claiming methodological novelty.

For submission: Digital Discovery or JCIM, with the opXRD scaling experiment as a stretch goal.
