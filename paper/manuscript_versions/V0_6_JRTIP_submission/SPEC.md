# Spec: JRTIP Manuscript V0.6 Submission Package

## Objective

Convert the reviewed V0.5 Chinese manuscript into a publication-oriented JRTIP package without overwriting the original document. The package must separate the concise main manuscript from reproducibility-oriented supplementary material and must not overstate the T41 real-scene evidence.

## Deliverables

- A polished Chinese master manuscript for author review.
- An English JRTIP submission manuscript.
- A JRTIP double-column Word build and rendered PDF.
- Supplementary material containing detailed ablations and internal experiment mappings.
- A figure-drawing specification for Visio or draw.io.
- A point-by-point response to all 31 Word comments.
- A verified bibliography and citation-audit table.

## Scientific Terminology

- Formal method name: Cost-Aware Frame-ROI Uploader (CAFRU).
- Chinese name: 成本感知帧级-ROI级上传器.
- Main-text method labels: CAFRU-Accuracy, CAFRU-Balanced, and CAFRU-Bandwidth.
- Internal labels such as Learned v1/v2, f0.60/r0.08, and E18-E23 are forbidden in the main manuscript.
- Frame and ROI probability thresholds are written as mathematical symbols `tau_f` and `tau_r`.
- SpeciesNet-generated boxes are called `SpeciesNet-assisted pseudo-reference annotations`, not ground truth.

## Main-Manuscript Evidence

Always retain:

1. COCO main comparison and operating points.
2. One consolidated core ablation table.
3. Weak-network and deadline-aware queue results.
4. Difficult-subset or uploader-selectivity evidence.
5. T41 end-to-end prototype costs and on-device policy overhead.
6. Explicit limitations and validity boundaries.

Move to supplementary material:

1. Full threshold sweeps and secondary ablation tables.
2. Per-experiment E-code descriptions.
3. Full HTTP field schemas and detailed logs.
4. Extended JPEG, network, queue, and metric-fairness tables.

## Layout Constraints

- Target journal: Journal of Real-Time Image Processing.
- Use the journal double-column format.
- Main manuscript must be no more than 12 pages, including references and author biographies.
- Abstract target: 150-250 English words.
- Keywords: 4-6.
- Figures must not contain embedded figure numbers or captions.
- Main figures: at most 5.
- Main tables: target 5-6.

## Boundaries

- Always preserve the original reviewed DOCX.
- Always distinguish algorithmic COCO evidence from T41 prototype evidence.
- Always report experimental limitations honestly.
- Never call pseudo-reference annotations manual GT.
- Never claim production-ready T41 stability from the current runtime results.
- Never compare latency or accuracy across incompatible settings without stating the difference.

## Success Criteria

- Every Word comment has a recorded response and disposition.
- No forbidden internal method labels remain in the main manuscript.
- Every non-trivial related-work claim has a verified citation.
- All bibliography entries have verified title, authors, venue/year, and DOI or stable URL where available.
- The rendered double-column PDF is at most 12 pages and has no clipped text, overlapping tables, or unreadable figures.
- Main text and supplement use consistent metric definitions and numeric values.

