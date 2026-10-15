# V0.8 Revision Notes

## Scope

- Preserved V0.7 unchanged.
- Created a separate Chinese scientific-final package using evidence available through 2026-07-25.
- Primary frozen evidence comes only from `20260725_three_dataset_costaware_selector_riskgate` and its immutable 20260723 source matrix.

## Major Corrections

1. Replaced every scientific placeholder in the abstract, results, analysis, and conclusion.
2. Corrected the main claim from a universal Ranker/ROI method to a decoupled architecture plus calibration-only cost-aware controller.
3. Reported 13/30 gate passes as 7 non-identical passes plus 6 identical-action passes; reported all 17 fallbacks.
4. Added all 30 frozen final work points, non-trivial AP/KB deltas, and per-dataset final-oracle regret.
5. Preserved the conservative false-negative result: rejected COCO Fixed-byte candidates later improved final AP, but were not restored.
6. Removed the unexecuted global-one-configuration ablation from completed comparisons.
7. Replaced the unsupported Ranker heatmap placeholder with frozen cross-domain selection analysis.
8. Marked every E2E value as estimated and retained the missing real-device evaluation as a limitation.
9. Added the published DOI `10.1109/SEC62691.2024.00021` for reference [13].
10. Did not invent academic citations, URLs, or licenses for ENA24 and Oregon.

## Evidence Boundaries

- COCO uses the 20260723 frozen-final cache protocol; it must not be merged with older COCO val diagnostic tables.
- ENA24 selector/gate are near-duplicate-group-disjoint, not known event/camera-disjoint.
- Oregon final is post-hoc exploratory reuse and its bootstrap is image-level.
- The 200-replicate gate is pointwise empirical risk screening, not a formal simultaneous guarantee.
- The final oracle is diagnostic only and never changes a frozen choice.

## Author-Controlled Submission Fields

The manuscript is scientifically complete, but the following cannot be inferred safely:

- author names, affiliations, corresponding-author details;
- funding and CRediT roles;
- ENA24/Oregon provenance, licenses, and public URLs;
- anonymized repository URL, release date, and archival DOI;
- ethics wording required by the institutions and source datasets.
