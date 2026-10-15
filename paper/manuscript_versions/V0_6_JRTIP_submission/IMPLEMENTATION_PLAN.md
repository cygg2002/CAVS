# Implementation Plan: JRTIP Manuscript V0.6

## Phase 1: Evidence and Layout Audit

- [x] Download and inspect the official JRTIP Word template.
- [x] Inventory all V0.5 sections, figures, tables, references, and reviewed comments.
- [x] Map each main result to its source experiment and decide main-text versus supplement placement.
- [x] Verify baseline, metric, and experimental-setting consistency.

Verification: produce `CONTENT_AUDIT.md` with retained, merged, and supplementary items.

## Phase 2: Terminology and Literature

- [x] Replace internal experiment language with CAFRU terminology and mathematical symbols.
- [x] Search edge-cloud inference, selective offloading, edge video analytics, confidence calibration, queue scheduling, and real-time detection literature.
- [x] Verify DOI metadata and claim-to-citation alignment.
- [x] Produce the final numbered bibliography and `CITATION_AUDIT.md`.

Verification: no unverified or duplicate reference remains.

## Phase 3: Manuscript Rewrite

- [x] Rewrite the Chinese master manuscript using a concise journal structure.
- [x] Create the English JRTIP manuscript from the stabilized Chinese master.
- [x] Create supplementary material with detailed ablations and implementation schemas.
- [x] Write the figure prompts and exact node/arrow descriptions for Visio or draw.io.
- [x] Write the 31-item review response document.

Verification: terminology scan passes and every main claim maps to a table, figure, experiment, or citation.

## Phase 4: Double-Column Build

- [x] Add an isolated JRTIP double-column build path without changing V0.5 outputs.
- [x] Generate Word and PDF artifacts.
- [x] Render PDF pages to images and inspect typography, tables, figures, equations, and page count.
- [x] Iteratively move secondary material to the supplement until the main manuscript is within 12 pages.

Verification: main PDF is at most 12 pages with zero visible layout defects.

## Phase 5: Submission Audit

- [x] Check title page, abstract length, keywords, headings, figure/table numbering, references, declarations, and data/code availability.
- [x] Check that T41 and pseudo-reference claims remain within measured evidence.
- [x] Produce `SUBMISSION_READINESS_CHECKLIST.md` with complete, author-input-required, and future-experiment items.

Verification: all P0 manuscript issues are closed or explicitly marked as requiring author information.
