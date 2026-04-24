# Future Work — Concrete Next Steps

## 1. Scale to 20,000+ Structures (Session 3 continuation)

5,000 structures are downloaded and cached. Scripts are ready to run
the full low-label ablation and noise experiments at this scale.

Expected outcome: SSL advantage should be maintained or increase with
more data, as the model can learn more complex structure-XRD relationships.

## 2. RRUFF Experimental Validation

The RRUFF database contains ~14,000 mineral XRD patterns with
corresponding crystal structures. This is the natural test for
whether sort-match SSL works on real experimental data with noise,
background, and peak broadening.

Steps:
- Download RRUFF database (rruff.info)
- Match structures to MP entries where possible
- Train on simulated XRD (labeled) + RRUFF experimental (unlabeled)
- Evaluate on held-out RRUFF patterns

## 3. Stronger GNN Backbones

Replace 4-layer GIN (257k params) with:
- **ALIGNN** (line graph neural network, includes bond angles)
- **CGCNN** (crystal graph CNN, community standard)
- **MACE** (higher-order equivariant, SOTA for materials)

This addresses Reviewer 5's concern about model capacity.

## 4. Multi-Wavelength Generalization

Current model is Cu Ka only. Could we:
- Add wavelength as a model input
- Train on multiple wavelengths simultaneously
- Test zero-shot transfer to unseen wavelengths

## 5. Other Spectroscopy Types (Papers 4, 5, ...)

Sort-match applies to ANY unordered scalar measurement set:
- **Raman spectroscopy**: peak positions in cm^{-1}
- **IR spectroscopy**: absorption peak positions
- **Mass spectrometry**: m/z peak positions
- **EELS**: energy loss peak positions

Each is a potential standalone paper demonstrating further generality.
