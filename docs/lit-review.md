# Literature Review: ML for XRD and SSL for Crystallography

## 1. XRD Peak Prediction (Forward: Structure -> XRD)

**DeepXRD** (Dong, Zhao, Song, et al., 2022, ACS Applied Materials & Interfaces)
- Predicts XRD spectrum from material COMPOSITION only (no structure)
- ResNet-20 architecture; 63-68% peak match accuracy
- Limitation: composition-only input creates ambiguity (polymorphs)
- Our positioning: we use full crystal STRUCTURE, not just composition

**Mat2Spec** (Chen et al., 2022, Nature Communications)
- Predicts phonon DOS and electronic DOS from crystal structure
- Uses graph attention networks + contrastive learning
- Uses Wasserstein distance as EVALUATION metric (not training loss)
- Our positioning: we use Wasserstein-based TRAINING loss

## 2. XRD Classification / Inverse Problem

**SimXRD-4M** (Cao et al., ICLR 2025, arXiv:2406.15469)
- 4M simulated PXRD patterns from 120k MP structures
- Benchmarks 21 models for crystal symmetry classification
- Key finding: long-tail distribution of crystal systems
- Our positioning: we address forward prediction, not classification

**DiffractGPT** (2024, J. Phys. Chem. Lett.)
- GPT for structure determination from XRD (inverse problem)
- Our positioning: we solve the FORWARD problem (complementary)

**XtalNet** (Lai et al., 2024-2025, Advanced Science)
- End-to-end crystal structure from PXRD
- 90.2% top-10 match rate on hMOF-100

**PXRDGen/PXRDnet** (2025, Nature Communications)
- XRD encoder + diffusion generator + Rietveld refinement
- 82% (1-sample) matching rate

## 3. Self-Supervised Learning for Crystal Properties

**Crystal Twins** (Magar et al., 2022, npj Computational Materials)
- Barlow Twins / SimSiam for crystal GNNs (CGCNN, GIN)
- 17-37% improvement over supervised baselines
- First SSL framework for crystal property prediction

**CDSSL** (New et al., 2024, ICML AI4Science)
- Denoising pretext task: perturb atoms, predict distances
- Outperforms non-SSL across materials

**DSSL** (Fu, Wei, Hu, 2024, J. Phys. Chem. Lett.)
- Physics-guided dual SSL: node masking + coordinate perturbation
- Up to 26.89% improvement

**Self-supervised XRD representations** (2025, Crystals/MDPI)
- SimCLR/Barlow Twins for learning XRD representations
- First SSL directly on XRD patterns (not structures)

## 4. Permutation-Invariant and OT Losses

**DETR** (Carion et al., ECCV 2020)
- Hungarian matching for set prediction in object detection
- Our sort-match is the 1D specialization with O(n log n)

**Learning with Wasserstein Loss** (Frogner et al., NeurIPS 2015)
- Foundational: Wasserstein as training loss for ML
- Entropic regularization for efficient computation

## 5. Crystal GNN Baselines

**CGCNN** (Xie & Grossman, 2018, PRL) — foundational crystal GNN
**MEGNet** (Chen et al., 2019, Chem. Mater.) — universal graph network
**ALIGNN** (Choudhary & DeCost, 2021, npj Comp. Mat.) — with bond angles
**Matformer** (Yan et al., NeurIPS 2022) — periodic graph transformer

## 6. Key Gap This Work Fills

No prior work applies:
1. Sort-match / optimal-transport TRAINING LOSS to XRD prediction
2. SSL to the FORWARD prediction problem (structure -> XRD)
3. Proved-optimal set matching for spectroscopic peak prediction beyond NMR

The combination of "sort-match loss + forward XRD + SSL" is novel.
