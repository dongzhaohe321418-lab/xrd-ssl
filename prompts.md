# XRD-SSL 项目 Claude Code 完整 Prompt 集

**项目目标**：把 nmr-ssl 的 sort-match SSL 框架扩展到粉末 XRD 峰预测，证明方法通用性，目标发表 Paper 3。

**总工作量估计**：7 个 Session，约 20–30 小时 Claude Code 运行时间，合计约 2–3 周日历时间（每周末专注一个 session）。

**执行原则**：
- 每个 Session 完成后人工检查结果再进下一个
- 每一步完成后 commit
- 发现定理或实验问题立即暂停，不要堆砌代码

---

## Session 0：环境准备（你自己做）

```bash
cd ~
mkdir xrd-ssl && cd xrd-ssl
git init
gh repo create xrd-ssl --public --source=. --remote=origin

# 注册 Materials Project API Key
# https://next-gen.materialsproject.org/api
export MP_API_KEY='你的key'
echo "export MP_API_KEY='你的key'" >> ~/.zshrc

# 启动 Claude Code
claude
```

---

## Session 1：基础架构 + 1D Sort-Match 验证（2–3 小时）

```
I'm extending my nmr-ssl project (~/nmr-ssl) to XRD powder pattern 
prediction. Start by reading these files from the reference project:

1. ~/nmr-ssl/README.md — project structure and claims
2. ~/nmr-ssl/docs/theorem.md — the 1D sort-match theorem and proof
3. ~/nmr-ssl/src/losses.py — sort_match_loss implementation
4. ~/nmr-ssl/src/data.py — data parser pattern
5. ~/nmr-ssl/src/model.py — GIN encoder (no torch_geometric)
6. ~/nmr-ssl/src/train.py — 3-variant training loop
7. ~/nmr-ssl/tests/test_theorem.py — numerical verification style
8. ~/nmr-ssl/docs/2d/preprint_2d.pdf — the 2D sliced-Wasserstein extension

Then carry out Session 1 of the new xrd-ssl project in ~/xrd-ssl.

============================================================
PROJECT: XRD-SSL — Sort-match SSL for powder XRD prediction
============================================================

SCIENTIFIC GOAL
---------------
Train a model that takes a crystal structure and predicts its 
powder XRD peak list {(2θ_i, I_i)}, using unassigned experimental 
or simulated peak lists as self-supervision via sort-match loss.

Stage 1 (this session, positions-only):
  Match predicted 2θ positions against observed 2θ positions using 
  1D sort-match loss. Ignore intensity. Verify the framework transfers.

Stage 2 (future sessions):
  Extend to 2D (2θ, intensity) via weighted sliced-Wasserstein.

SESSION 1 DELIVERABLES
======================

A. Repository scaffolding
   - README.md, mirror style of ~/nmr-ssl/README.md
   - requirements.txt: pymatgen, torch>=2.0, numpy, scipy, 
     matplotlib, tqdm
   - LICENSE (CC-BY 4.0)
   - .gitignore excluding data/, results/, __pycache__, .DS_Store

B. Data pipeline — src/xrd_data.py
   - Use pymatgen MPRester (MP_API_KEY env var) to download 5000 
     oxide structures with formation energy < 0.1 eV/atom, <=30 atoms.
   - Cache as data/mp_cifs/*.cif and a structures.json index.
   - Simulate XRD with pymatgen.analysis.diffraction.xrd.XRDCalculator 
     (Cu Kα, λ=1.54 Å).
   - Extract top-50 peaks by intensity in 2θ ∈ [5°, 90°].
   - Pad with sentinel (2θ=0, I=0); provide a mask.
   - Save to data/xrd_cache.npz with arrays:
       material_ids: (N,) string
       two_theta: (N, 50) float32
       intensity: (N, 50) float32 (normalized max=100 per pattern)
       mask: (N, 50) bool

C. Features — src/features.py
   - Atom features: one-hot atomic number (1..100), Pauling 
     electronegativity, covalent radius, group+period one-hot.
   - Graph: nodes = atoms in unit cell, edges via 
     Structure.get_all_neighbors with 5 Å cutoff, periodic images OK.
   - Edge features: distance RBF-expanded (8 bins, 0 to 5 Å).
   - Dataclass: CrystalGraph(node_feats, edge_index, edge_feats, n_atoms).
   - Implement message passing manually like nmr-ssl. No torch_geometric.

D. Model — src/model.py
   - 4-layer GIN operating on CrystalGraph.
   - Global pooling = mean concat sum.
   - Output head: MLP producing 50 predicted 2θ values in degrees.
   - Target <500k params.

E. Loss — src/losses.py
   - Port sort_match_loss from ~/nmr-ssl/src/losses.py.
   - Add masked_sort_match_loss handling peak masks (carefully: 
     pad tokens must not contribute after sorting — comment the logic).
   - Add hungarian_reference for verification.

F. Theorem verification — tests/test_theorem.py
   - Random trials n ∈ [10, 50], assert sort_match_loss matches 
     hungarian_reference to 1e-10 under MSE and Huber costs.
   - Masked variant test.
   - Runtime target: <5 s.

G. Sanity experiment — experiments/sanity_check.py
   - Train on 1000 structures, 20 epochs, 3 variants:
       (1) supervised — MSE vs sorted ground truth
       (2) random_match — MSE vs randomly-permuted ground truth
       (3) sort_match — sort_match_loss
   - labeled_frac=0.1, 200-structure held-out test.
   - Report test MAE in degrees 2θ.
   - Target: sort_match within 10% of supervised. If not, FLAG.

H. Figures — experiments/make_figures.py
   - fig1: training curves (3 variants)
   - fig2: predicted vs true 2θ scatter for one test structure
   - Save as PDF+PNG in figures/.

CONSTRAINTS
===========
- Apple M4 Pro with MPS backend
- No torch_geometric
- Every theorem claim numerically verified
- Code style: concise, typed, docstrings explain physics
- All imports at top; no lazy imports
- Use np.random.default_rng

ORDER OF OPERATIONS
===================
Do in order, commit after each step:
1. Scaffold repo (A)
2. Losses + tests (E, F) — verify theorem BEFORE touching data
3. Data pipeline (B) — test on 20 structures first
4. Features + model (C, D)
5. Sanity experiment (G)
6. Figures (H)
7. Update README with real numbers from step 5

Final output: a summary with theorem max error, sanity MAE for 
all 3 variants, total runtime, git log.

Start with step 1.
```

**Session 1 完成后人工检查：**
```bash
pytest tests/test_theorem.py -v  # 必须全过
python experiments/sanity_check.py  # 看 sort_match 是否接近 supervised
open figures/fig1.pdf  # 训练曲线合理
```

---

## Session 2：扩展到 2D（位置 + 强度）— 核心创新（3–4 小时）

```
Session 1 verified 1D sort-match works on XRD positions. Now extend 
to 2D peaks using the strategy in ~/nmr-ssl/docs/2d/.

SCIENTIFIC QUESTION
===================
XRD peaks are 2D: (2θ, intensity). Two approaches:

APPROACH 1 — Sliced-Wasserstein (the nmr-ssl v2 way):
  Project 2D peaks onto random directions θ ∈ S^1, apply 1D 
  sort-match on each projection, average over directions.
  Already validated on HSQC data. Apply directly.

APPROACH 2 — Physics-informed sliced directions:
  XRD positions carry more information than intensities (positions 
  constrain lattice exactly; intensities depend on Debye-Waller, 
  texture, particle size). Instead of uniform S^1 sampling, bias 
  directions toward the 2θ axis.
  
Implement BOTH. Compare rigorously.

DELIVERABLES
============

A. Loss — src/losses_2d.py
   - sliced_wasserstein_loss(pred_2d, obs_2d, n_slices=128, 
                             direction_prior='uniform' | 'xrd_biased',
                             mask=None)
   - When direction_prior='xrd_biased': sample directions from 
     a von Mises distribution centered on the 2θ axis with 
     concentration κ (hyperparameter, default κ=2.0).
   - Normalize 2θ to [0,1] over [5°, 90°] and intensity to [0,1] 
     before projection.
   - Verify differentiability with a gradient check.

B. Mathematical justification — docs/theorem_2d.md
   - Statement: for 2D convex cost, is there an analog of the 
     exact 1D sort-match? Answer: no (give a concrete 
     counterexample with 3 points). 
   - Therefore sliced-Wasserstein is a principled approximation.
   - Derive why XRD-biased directions should converge faster to 
     the true 2D Wasserstein distance when position carries more 
     information than intensity.
   - Include numerical evidence: for 50 random 2D point sets, 
     compare approximation error of uniform vs xrd-biased sampling 
     at n_slices ∈ {8, 32, 128, 512} against exact 2D Hungarian.
   - Save as docs/theorem_2d.md AND docs/theorem_2d.pdf (via pandoc).

C. Numerical verification — tests/test_2d.py
   - Compare sliced-Wasserstein (many slices) against scipy 
     Hungarian on the 2D cost matrix.
   - Expected: sliced-Wasserstein with 512 slices agrees with 
     Hungarian to within 1% relative error.
   - Compare uniform vs xrd-biased convergence.

D. Model upgrade — src/model.py
   - Add a second output head predicting log-intensity for each 
     of the 50 peaks.
   - Sort-match is now 2D: inputs are pred (N,50,2) and obs (N,50,2).

E. 2D experiment — experiments/run_2d.py
   - Train 4 variants on 5000 structures, 30 epochs, labeled_frac=0.1:
       (1) supervised_2d
       (2) sort_match_1d_positions_only (baseline from Session 1)
       (3) sliced_ws_uniform
       (4) sliced_ws_xrd_biased
   - 3 seeds each.
   - Report test MAE for positions (°) and intensity (log R²).
   - Statistical test: paired t-test across seeds, xrd_biased vs 
     uniform.

F. Figures — experiments/make_figures_2d.py
   - fig3: sliced-Wasserstein approximation error vs n_slices, 
     uniform vs xrd-biased
   - fig4: 2D experiment bar plot with error bars
   - fig5: example predicted vs true 2D scatter for 3 test structures

CONSTRAINTS
===========
- Keep Session 1 code untouched; add new files
- Every claim in theorem_2d.md numerically verified
- If xrd_biased does NOT outperform uniform, say so honestly and 
  investigate — do not massage results
- Runtime target: <2 hours on M4 Pro for full 2D experiment

ORDER OF OPERATIONS
===================
1. Implement sliced_wasserstein_loss (A)
2. Write and verify theorem_2d.md claims numerically (B, C)
3. Upgrade model (D)
4. Run 2D experiment (E)
5. Generate figures (F)
6. Summary with statistical comparison

Start with step 1.
```

**Session 2 完成后人工检查：**
- `docs/theorem_2d.md` 里的反例和论证是否合理
- xrd_biased 是否真的优于 uniform（如果没有，是重要的 null result）
- 全 4 个 variant 的 MAE 是否有序：sort_match_2d < sort_match_1d ≈ supervised < random_match

---

## Session 3：大规模实验 + 鲁棒性（4–5 小时）

```
Sessions 1-2 established the method works. Now scale up and stress-test.

DELIVERABLES
============

A. Large-scale data expansion — src/xrd_data.py
   - Extend download from 5000 to 20000 Materials Project entries
   - Cover: oxides, sulfides, halides, nitrides (not just oxides)
   - Structure diversity report: crystal system distribution, 
     element coverage, number of atoms histogram
   - Save as data/xrd_cache_large.npz
   - Runtime: be patient with MP API rate limits, use exponential 
     backoff on errors

B. Low-label ablation — experiments/run_low_label.py
   - labeled_fraction ∈ {0.02, 0.05, 0.1, 0.2, 0.5}
   - Methods: supervised, sort_match_1d, sliced_ws_xrd_biased
   - 3 seeds each, 20000 structures
   - Expected: SSL advantage grows as labeled_fraction shrinks 
     (textbook SSL signature, same as nmr-ssl Figure 2)

C. Robustness — experiments/run_robustness.py
   - For the best SSL variant (from Session 2), corrupt the 
     unlabeled data:
       (i)   clean baseline
       (ii)  Gaussian noise on 2θ: σ=0.1°, 0.3°, 1.0°
       (iii) peak dropout: drop 10%, 25%, 50% of peaks
       (iv)  spurious peaks: add 10%, 25% random peaks
       (v)   combined worst case
   - Measure test MAE for each corruption level
   - 2 seeds each

D. Generalization split — experiments/run_ood.py
   - Two splits:
       (i)  random split (standard)
       (ii) chemical-system holdout: train on systems A, test on 
            systems B (e.g., train on oxides, test on sulfides)
   - Measure the generalization gap

E. Comparison to the naive baseline — experiments/run_naive.py
   - Use scipy.optimize.linear_sum_assignment as the "naive" 
     training loss (Hungarian per batch, full O(n^3) cost).
   - Compare: accuracy should be identical to sort_match 
     (the theorem guarantees this), but runtime should be 
     ~100x slower. Quantify both.

F. Figures — experiments/make_figures_full.py
   - fig6: low-label ablation (MAE vs labeled_frac, 3 methods)
   - fig7: robustness heatmap (corruption level × MAE)
   - fig8: OOD generalization gap bar chart
   - fig9: Hungarian vs sort-match wall-time comparison

G. Results dump — experiments/results_session3/
   - Save every experiment's full log as JSON
   - Include hyperparameters, git commit hash, runtime per run, 
     seed, test MAE

CONSTRAINTS
===========
- Expected total runtime: 8-12 hours. Use run_overnight.py pattern 
  from nmr-ssl (per-run caching, resume on failure).
- If any experiment crashes, auto-save partial results and continue.
- Track memory: 20000 structures × 50 peaks × GNN batch should 
  fit in M4 Pro RAM.
- If low-label ablation does NOT show SSL advantage, this is a 
  critical finding — flag immediately, diagnose, do not hide.

ORDER OF OPERATIONS
===================
1. Expand dataset (A)
2. Run low-label ablation (B) — this is the most important result
3. Run robustness (C)
4. Run OOD (D)
5. Run naive Hungarian comparison (E)
6. Generate all figures (F)
7. Consolidate results (G)
8. Summary with all numbers

Start with step 1.
```

**Session 3 完成后人工检查：**
- Low-label ablation 曲线形状：SSL advantage 随标注减少增大
- Hungarian vs sort-match 运行时对比（应该是 50–100x 差异）
- OOD 泛化是否有明显 gap

---

## Session 4：文献调研 + 基线对比（2–3 小时）

```
Before writing the paper, rigorously position the work against 
existing XRD deep learning literature.

DELIVERABLES
============

A. Literature survey — docs/lit-review.md
   Use WebFetch to verify every citation. NO VIBE CITING.
   
   Sections (each 3-5 verified papers):
   1. XRD peak indexing classical methods (Dicvol, Topas, GSAS II)
   2. CNN-based XRD classification (Park 2017, Oviedo 2019, 
      Ziletti 2018)
   3. XRD → crystal system/space group (Tiong 2020, Chen 2024)
   4. XRD → structure (DiffractGPT 2025, End-to-End PXRD 
      Advanced Science 2025)
   5. Semi-supervised XRD (Lolla 2022 NIST)
   6. Rietveld-style ML (Ozaki 2020, Chang 2020)
   
   For each: one-paragraph summary, what they solve, what they 
   don't solve, why sort-match SSL is complementary (NOT 
   competitive — most prior work is for different tasks).
   
   Explicit positioning paragraph: "We are the first to apply 
   set-matching SSL to XRD peak prediction. Prior semi-supervised 
   XRD work (Lolla 2022) addresses classification, not peak-level 
   regression. Prior generative models (DiffractGPT) solve the 
   inverse problem (XRD → structure), whereas we address the 
   forward problem (structure → XRD) with unassigned-peak 
   supervision."

B. Baseline comparisons — experiments/run_baselines.py
   Implement 3 baselines for a fair head-to-head:
   
   (i)  CNN-on-pattern: treat the full simulated continuous XRD 
        pattern as a 1D signal of length 8500 (binned at 0.01°), 
        use 1D CNN to predict peaks. No peak extraction needed 
        upfront. This is the mainstream approach.
   
   (ii) Rietveld-based MAE: for each test structure, simulate its 
        XRD with pymatgen and compute the oracle MAE against 
        ground truth. This is the theoretical floor.
   
   (iii) Naive regression: GNN predicts sorted peaks, MSE against 
         sorted ground truth (what our 'supervised' variant does, 
         but using fully labeled data).
   
   Compare all on the 20000-structure test split.
   Report: MAE, R², runtime per prediction.

C. Failure analysis — experiments/failure_analysis.py
   Identify the 100 structures with the highest sort_match_ssl 
   test MAE. For each, report:
     - Crystal system
     - Number of atoms in unit cell
     - Space group
     - Number of true peaks (vs padded)
   Generate failure_modes.md summarizing patterns:
     "Low-symmetry structures (triclinic, monoclinic) dominate 
      failure cases because they produce more peaks in 2θ ∈ [5°, 90°] 
      than our top-50 window can hold."
   This is honest science — report what doesn't work.

D. Runtime benchmarks — experiments/run_benchmarks.py
   - Training time per epoch: our method vs baseline (i)
   - Inference time per structure: all methods
   - Memory: peak GPU/MPS RAM usage
   - Report as table in docs/benchmarks.md

CONSTRAINTS
===========
- Every citation in lit-review.md verified via WebFetch
- Baselines implemented in good faith — no deliberate handicap
- Failure analysis is mandatory, not optional

ORDER OF OPERATIONS
===================
1. Literature survey (A)
2. Implement baselines (B)
3. Failure analysis (C)
4. Runtime benchmarks (D)

Start with step 1.
```

**Session 4 完成后人工检查：**
- 读 lit-review.md，判断每个引用是否真的读过
- 看 baseline 的 MAE 是否合理（不应碾压 SSL，否则工作没价值；也不应差太多）
- 失败分析是否诚实揭示问题

---

## Session 5：预印本撰写（3–4 小时）

```
All experiments done. Now write the paper.

Target venue options (choose based on results):
  - npj Computational Materials (ideal if MAE is competitive)
  - Journal of Chemical Information and Modeling
  - Chemistry of Materials

DELIVERABLES
============

A. Full preprint — docs/preprint_v1.md
   Structure (Nature-CS style, 8-12 pages):
   
   TITLE: "Sort-Match Self-Supervision for Powder XRD Peak 
           Prediction: Extending the NMR Framework to Diffraction 
           Data"
   
   ABSTRACT (150 words):
     - Problem: XRD peak prediction from structure is exact via 
       Rietveld but slow; ML methods need labeled data that's 
       expensive to curate
     - Method: sort-match SSL, proved optimal for 1D convex costs 
       (NMR Paper 1), extended to 2D via sliced-Wasserstein with 
       XRD-biased directions
     - Result: X% relative MAE improvement in low-label regime
     - Takeaway: set-matching SSL is general — works for any 
       scalar or vector measurement set
   
   1. Introduction (1 page)
      - XRD is foundational, Rietveld is exact but slow for 
        high-throughput screening
      - Supervised ML needs peak-labeled data, but most XRD data 
        in literature is unassigned
      - NMR faces the same problem; sort-match SSL solved it
      - Can we transfer the framework?
   
   2. Theoretical framework (1.5 pages)
      - Restate 1D sort-match theorem (reference nmr-ssl Paper 1)
      - Why XRD intensity breaks 1D assumption
      - 2D extension via sliced-Wasserstein
      - Physics-informed direction sampling
      - Include the counterexample from theorem_2d.md showing no 
        exact 2D sort-match exists
   
   3. Methods (1 page)
      - Data: Materials Project, 20000 structures, pymatgen XRD 
        simulation
      - Model: 4-layer GIN (no torch_geometric)
      - Training: AdamW, MPS backend, 30 epochs
      - Metrics: position MAE (°), intensity log R²
   
   4. Results (3 pages)
      4.1 Theorem verification (Table 1, numerical max error)
      4.2 Main comparison at labeled_frac=0.1 (Table 2)
      4.3 Low-label ablation (Figure 2 — the strongest result)
      4.4 Robustness to corruption (Figure 3)
      4.5 2D intensity prediction (Figure 4)
      4.6 OOD generalization (Figure 5)
      4.7 Runtime vs Hungarian (Figure 6 — elegance argument)
   
   5. Discussion (1 page)
      - When sort-match SSL helps most (low-label)
      - Failure modes (low-symmetry structures)
      - What this is NOT: not a Rietveld replacement for 
        well-crystalline materials; not a solution to inverse 
        problem (structure determination)
      - Future work: apply to single-crystal XRD, Raman, IR, 
        PXRD with real experimental data (RRUFF)
   
   6. Conclusion (0.3 pages)
   
   All numerical claims backed by JSON logs in 
   experiments/results_session3/.
   
   Include the nmr-ssl v2 precedent as a citation: "This work is 
   the second instantiation of a general framework first 
   demonstrated for NMR chemical shifts [cite nmr-ssl preprint]."

B. Figure compilation — figures/compile.py
   Assemble all figures from Sessions 1-3 into paper-ready PDFs:
     - Fig1: sort-match schematic (draw fresh with matplotlib, 
       2 panels: 1D NMR case, 2D XRD case)
     - Fig2: main result = low-label ablation (re-plot from 
       session 3)
     - Fig3: robustness heatmap
     - Fig4: 2D intensity prediction scatter
     - Fig5: OOD generalization
     - Fig6: runtime vs Hungarian
   All figures must be single-column or two-column clean, 
   Nature-CS style: sans-serif, no 3D effects, error bars 
   explicit, panel labels a/b/c bottom-right.

C. LaTeX/PDF compilation — experiments/compile_preprint.py
   Use pandoc to convert docs/preprint_v1.md to preprint.tex, 
   then xelatex to preprint.pdf.
   Embed all figures.
   Target: 12 pages including references.

D. Placeholder check — scripts/check_placeholders.py
   Scan preprint_v1.md for any remaining [XXX], TODO, TBD, ???. 
   If found, list them and abort. All numbers must be real.

E. Citation verification — scripts/verify_citations.py
   Parse preprint_v1.md references. For each, check:
     - DOI resolvable
     - Title matches abstract
   Report any citation that fails verification.

CONSTRAINTS
===========
- Every number traced to a JSON log
- No "vibe citations" — if you can't verify it, remove it
- Honest framing: if OOD gap is large, say so; if xrd_biased 
  didn't beat uniform, say so
- Abstract written LAST, after all results are in

ORDER OF OPERATIONS
===================
1. Draft structure of preprint_v1.md with all sections but 
   placeholders for numbers
2. Fill in numbers from JSON logs (use the same fill_preprint.py 
   pattern as nmr-ssl)
3. Compile figures (B)
4. Run placeholder and citation checks (D, E)
5. Compile PDF (C)

Start with step 1.
```

**Session 5 完成后人工检查：**
- 读整份 preprint_v1.pdf，看论证链条是否自洽
- 每个数字都能追溯到 experiments/results/ 里的 JSON
- 没有 [XXX] 或 TODO
- 引用全部通过验证

---

## Session 6：模拟同行评审 + 修改（2–3 小时）

```
Simulate a 5-reviewer peer review cycle on the preprint, then 
revise.

DELIVERABLES
============

A. Simulated review — docs/review_round1.md
   Act as 5 distinct reviewers, each with a different focus:
   
   REVIEWER 1 — Editor-in-Chief (npj Computational Materials)
     - Is the fit right? (materials informatics + ML methodology)
     - Is the contribution novel given existing XRD ML work?
     - Major revision / minor revision / reject?
   
   REVIEWER 2 — Methodology (ML/OT theory expert)
     - Is the 2D sliced-Wasserstein argument rigorous?
     - Is the xrd_biased direction sampling properly justified?
     - Should we compare to Sinkhorn OT as an alternative?
     - Are error bars computed correctly?
   
   REVIEWER 3 — Domain (crystallography/XRD)
     - Is Rietveld the right floor comparison?
     - Does "top-50 peaks" make physical sense? Low-symmetry 
       structures have more peaks.
     - Why Cu Kα? Why 2θ ∈ [5°, 90°]?
     - Is the Materials Project simulated data a reasonable 
       proxy for experiment?
   
   REVIEWER 4 — Perspective (broader materials informatics)
     - Is this incremental to nmr-ssl?
     - What's the practical use case? Who benefits?
     - How does this connect to autonomous labs and high-throughput 
       screening?
   
   REVIEWER 5 — Devil's advocate
     - Could the SSL improvement be a confounder (e.g., data 
       leakage, larger effective model capacity, longer training)?
     - Is the "10% relative improvement" practically meaningful 
       given Rietveld can do 0% error?
     - Why not benchmark against a much stronger GNN (M3GNet, 
       MACE) instead of 4-layer GIN?
   
   Each reviewer writes: summary, strengths, weaknesses, 
   decision (Accept / Minor Rev / Major Rev / Reject), 
   5-10 specific comments.
   
   At the end: editorial decision synthesizing all 5.

B. Revision response — docs/revision_response.md
   For every actionable comment from R1-R5, respond with one of:
     - FIXED: done, with pointer to the change in preprint_v2.md
     - DEFERRED: flagged as future work, explain why it's out of 
       scope
     - DISPUTED: respectfully explain why the reviewer is wrong
   No comment can be ignored.

C. Preprint v2 — docs/preprint_v2.md
   Apply all FIXED revisions to preprint_v1.md.
   Maintain a visible diff in docs/diff_v1_v2.md showing every 
   change.

D. Re-verify claims — scripts/reverify_after_revision.py
   Re-run placeholder check and citation verification on v2.
   Re-compile PDF.

E. Honest venue assessment — docs/venue_assessment.md
   Given the final preprint, honestly estimate probability of 
   acceptance at:
     - npj Computational Materials
     - J. Chem. Inf. Model. (JCIM)
     - Chem. Mater.
     - arXiv preprint only
   Include: what reviewers would likely demand at each venue.

CONSTRAINTS
===========
- 5 reviewers must be genuinely distinct personalities 
  (don't make them all agree)
- At least ONE reviewer should push back strongly on something 
  (Devil's Advocate role)
- Revision must address reviewer concerns substantively, not 
  cosmetically
- If a reviewer uncovers a real problem (e.g., missing baseline), 
  run the extra experiment

ORDER OF OPERATIONS
===================
1. Write all 5 reviews (A)
2. Plan revisions (B)
3. Apply revisions (C)
4. Re-verify (D)
5. Venue assessment (E)

Start with step 1.
```

**Session 6 完成后人工检查：**
- 5 位审稿人是否真的角度不同
- Devil's Advocate 有没有发现真问题
- 修改是否实质性

---

## Session 7：最终打包 + arXiv 提交（1–2 小时）

```
Final packaging for arXiv submission and GitHub release.

DELIVERABLES
============

A. Final README — README.md
   Rewrite with final results. Structure:
     - One-line description
     - Key claim (with actual numbers)
     - Install
     - Reproduce main result (exact commands)
     - Citation (arXiv when available)
     - Acknowledgments (Claude Code, nmr-ssl precedent)

B. FINAL_REPORT.md — mirror the nmr-ssl FINAL_REPORT.md style
   Sections:
     - 60-second summary
     - Headline numbers (tables from all experiments)
     - Peer review verdict
     - Artifacts produced (code, tests, experiments, figures, 
       docs with line counts)
     - Honest venue assessment
     - Author contribution statement
     - Acknowledgment of Claude Code assistance under human 
       oversight

C. arXiv submission prep — arxiv/
   - arxiv/preprint_arxiv.tex (single .tex file with all figures 
     inlined or in arxiv/figures/)
   - arxiv/metadata.md with: title, abstract, authors, 
     primary category (cond-mat.mtrl-sci), secondary 
     (cs.LG, physics.comp-ph), MSC/ACM class
   - arxiv/submission_checklist.md: files to upload, size check, 
     license statement

D. GitHub release — scripts/create_release.sh
   Commands (don't execute, just write the script):
     git tag v1.0
     git push origin v1.0
     gh release create v1.0 --title "Preprint v1" 
                           --notes-file FINAL_REPORT.md 
                           arxiv/preprint_arxiv.pdf

E. Follow-up work ideas — docs/future_work.md
   Real list, not handwaving:
     1. Apply to RRUFF experimental data (address Reviewer 3 
        concern that simulated data is a proxy)
     2. Extend to single-crystal diffraction (2D detector images)
     3. Combine with MACE/M3GNet backbone for fairer comparison 
        to state-of-the-art GNNs
     4. Integrate into autonomous XRD indexing workflow 
        (crystallography companion agent)
     5. Apply sort-match to other unassigned measurement sets 
        (Raman, IR, mass spec, EELS) — each is a potential Paper 4

F. Project audit — docs/audit.md
   - Total lines of code
   - Total experiments run
   - Total compute time used
   - Git commit count
   - All comparable to nmr-ssl as a calibration

CONSTRAINTS
===========
- No new experiments in this session — packaging only
- All commit messages clean
- arXiv PDF must compile reproducibly
- Acknowledgment section must credit Claude Code explicitly, 
  same spirit as nmr-ssl

ORDER OF OPERATIONS
===================
1. Final README (A)
2. FINAL_REPORT.md (B)
3. arXiv prep (C)
4. Release script (D)
5. Future work (E)
6. Audit (F)

Print final summary with all key numbers and arXiv-ready status.

Start with step 1.
```

**Session 7 完成后人工操作：**
```bash
# 自己上传 arXiv（Claude Code 不能代你操作浏览器）
cd arxiv/
# 检查 preprint_arxiv.pdf 和所有 figures
# 去 arxiv.org 登录，按 metadata.md 的指引提交

# GitHub 发布
bash scripts/create_release.sh
```

---

## 总时间表

| Session | 内容 | 预计时间 | 交付物 |
|---|---|---|---|
| 0 | 环境准备 | 5 分钟 | 仓库+API Key |
| 1 | 1D 基础架构 + 定理验证 | 2–3h | 定理测试通过 + sanity MAE |
| 2 | 2D sliced-Wasserstein 扩展 | 3–4h | theorem_2d.md + 4-variant 实验 |
| 3 | 大规模实验 + 鲁棒性 | 4–5h | 低标注消融 + OOD + 鲁棒性 |
| 4 | 文献调研 + 基线对比 | 2–3h | lit-review + 3 个基线对比 |
| 5 | 预印本撰写 | 3–4h | preprint_v1.pdf |
| 6 | 模拟同行评审 + 修改 | 2–3h | preprint_v2.pdf + 修改记录 |
| 7 | 打包 + arXiv 准备 | 1–2h | arXiv 准备就绪 + GitHub release |

**总计：18–24 小时 Claude Code 运行时间，2–3 个周末可完成。**

---

## 重要提示

**不要一次把所有 prompt 都塞给 Claude Code。** 每个 Session 独立执行，每次新开一个对话（或者用 `/clear` 清空历史），让它 focus 在当前任务。

**每个 Session 之间做人工检查。** 如果某个 Session 的结果有问题（比如定理验证失败、实验结果不合理），暂停，不要进入下一 Session。Claude Code 会不自觉地 "修好" 表面错误而忽视根本问题。

**保留完整日志。** 每个 Session 完成后把 Claude Code 的输出保存到 `logs/session_N.md`。这是你论文里"full peer-review iteration log"的素材，也是和 nmr-ssl 一脉相承的做法。

---

现在就可以开始 Session 0 + Session 1。做完告诉我结果，我根据实际情况帮你调整后续 Session 的 prompt。
