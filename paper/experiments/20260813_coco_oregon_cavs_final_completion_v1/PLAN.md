# Implementation plan

1. Validate protocols, source hashes, detector hashes, roles and split identities.
2. Import immutable dedicated OQ and completed COCO MAUR artifacts.
3. Build Oregon MAUR contextual rewards using the frozen Current@640 geometry and
   predictions, then train 5-fold OOF plus full-fit, three seeds.
4. Train same-backbone CostInValue-OQ on both datasets and COCO ORIC/MORIC
   AP50-95 seeds required for symmetric comparison.
5. Evaluate single seeds, ensemble, TwoStage-OQ, scalar baselines and ROI extension
   under identical Fixed-byte/Fixed-call capacities.
6. Run nested Random-50 and standard paired bootstrap; import strict 2x2 bootstrap
   only after its completion marker and hashes are valid.
7. Generate machine-readable manifests plus Chinese paper-ready report.

The pipeline is resumable. Each stage writes an atomic completion marker only
after validating the complete expected artifact set.

