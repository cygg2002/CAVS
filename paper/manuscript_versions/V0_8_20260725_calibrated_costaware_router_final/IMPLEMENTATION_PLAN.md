# Implementation Plan: JRTIP V0.8 Final Package

## Phase 1: Evidence Contract

- [x] Preserve V0.7 and freeze the evidence cutoff.
- [x] Record allowed and forbidden claims.
- [x] Add failing tests for gate counts, final row counts, regrets, and figures.

## Phase 2: Figures

- [x] Generate the online/offline architecture figure.
- [x] Generate three-dataset AP-request-cost plots.
- [x] Generate gate pass/fallback composition.
- [x] Generate Edge/Original/Q90/ROI action composition.
- [x] Verify dimensions, nonblank pixels, labels, and source row counts.

## Phase 3: Manuscript

- [x] Replace all scientific placeholders with frozen results.
- [x] Remove unfinished global-one-config claims from completed experiments.
- [x] Add aggregate, non-trivial pass, and per-work-point result tables.
- [x] Correct DOI metadata and evidence limitations.
- [x] Run numerical and placeholder audit.

## Phase 4: Word and QA

- [x] Build editable double-column Word manuscript.
- [x] Render Word to PDF and inspect all eight pages.
- [x] Verify embedded figures, tables, references, and page count.
- [x] Write revision notes and submission blockers.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Old and new COCO AP protocols are mixed | High | Use only 20260723 frozen-final scope in the primary controller tables |
| Gate pass count is overstated | High | Split identical and non-identical passes in every summary |
| Word conversion exposes Markdown/LaTeX | Medium | Use the existing editable equation cleaner and rendered PDF QA |
| Missing author metadata is mistaken for scientific incompleteness | Medium | Isolate it in submission declarations and revision notes |
