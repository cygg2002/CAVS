# Submission Readiness Checklist: JRTIP V0.6

## 1. Current Status

The V0.6 package is a submission-oriented draft rather than an internal
experiment archive. The reviewed V0.5 Word document was not overwritten.

Generated manuscript artifacts:

| Artifact | Status |
|---|---|
| `JRTIP_MANUSCRIPT_V0_6_EN_SUBMISSION.md` | English submission manuscript source |
| `JRTIP_MANUSCRIPT_V0_6_EN_SUBMISSION.docx` | Generated editable Word manuscript |
| `rendered/JRTIP_MANUSCRIPT_V0_6_EN_SUBMISSION.pdf` | Earlier rendered PDF for layout checking; re-export from the latest DOCX before submission |
| `rendered/JRTIP_MANUSCRIPT_V0_6_EN_SUBMISSION_latest.pdf` | Latest 7-page Word export with English workflow figures |
| `JRTIP_MANUSCRIPT_V0_6_ZH_MASTER.md` | Chinese master manuscript |
| `JRTIP_MANUSCRIPT_V0_6_ZH_MASTER.docx` | Generated Chinese Word manuscript |
| `rendered/JRTIP_MANUSCRIPT_V0_6_ZH_MASTER_latest.pdf` | Latest 7-page Word export with Chinese workflow figures |
| `ONLINE_RESOURCE_1_SUPPLEMENT.md` | Supplementary material source |
| `ONLINE_RESOURCE_1_SUPPLEMENT.docx` | Generated supplementary Word file |
| `rendered/ONLINE_RESOURCE_1_SUPPLEMENT.pdf` | Earlier rendered supplementary PDF; re-export from the latest DOCX before submission |
| `rendered/ONLINE_RESOURCE_1_SUPPLEMENT_latest.pdf` | Latest 4-page Word export |
| `FIGURE_DRAWING_SPEC_ZH.md` | Visio/draw.io drawing specification for workflow figures |
| `REVIEW_RESPONSE_31_ITEMS_ZH.md` | Response plan for reviewed V0.5 comments |
| `CONTENT_AUDIT.md` | Evidence, terminology, and main/supplement split audit |
| `CITATION_AUDIT.md` | Citation metadata and claim-to-citation audit |

## 2. Checks Passed

| Item | Result |
|---|---|
| Current JRTIP page-budget check | Latest English Word export is 7 A4 double-column pages, below the 12-page limit |
| Abstract length | 241 English words, within the 150-250 word requirement |
| Keywords | 6 keywords |
| Internal method names | `Learned v2`, `f0.60/r0.08`, and raw experiment-folder labels removed from manuscript and supplement |
| Method name | Formalized as Cost-Aware Frame-ROI Uploader (CAFRU) |
| T41 label boundary | SpeciesNet boxes are described as pseudo-reference annotations, not ground truth |
| Main/supplement split | Main paper reduced to 5 figures and 5 tables; full sweeps moved to Online Resource 1 |
| Literature scope | Edge-cloud inference, selective offloading, video analytics, freshness, calibration, and deployment references audited |
| Word generation | English, Chinese, and supplementary DOCX files regenerated with embedded workflow figures |
| Earlier PDF layout sanity check | Earlier rendered PDF pages inspected for page-count, footer, list, table, and equation issues |
| Source-like residue scan | No `Learned v2`, `f0.`, `r0.`, `E18`, `E23`, `pseudo-GT`, or obvious LaTeX residue found in generated DOCX files |
| Workflow figures | Fig. 1, Fig. 2, and Fig. 5 have separate editable Chinese/English draw.io sources and are embedded by manuscript language |
| Latest PDF visual QA | No figure clipping, text overflow, placeholder boxes, or split Table 5 in the 7-page English and Chinese exports |
| Metric fairness | Primary edge, cloud, and CAFRU claims use the common deployment confidence threshold of 0.25 |
| T41 FPS boundary | Queue results are labeled as scheduling-isolation throughput with local YOLO26s bypassed |

Notes:

1. The `pypdf` page counter reports non-fatal object-offset warnings on
   Microsoft Word exported PDFs, but the PDFs render and page counts are
   readable.
2. The generated Word equations are editable linear expressions, not final
   journal-quality OMML equations. They are readable for review, but the final
   author-side template pass should convert important equations to proper
   Word equation objects if required.
3. Files ending in `_latest.pdf` were exported after the bilingual figure and
   Table 5 pagination updates. Files without `_latest` are older QA artifacts.

## 3. Must Be Completed Before Actual Submission

| Priority | Item | Required action |
|---|---|---|
| P0 | Author metadata | Fill author names, affiliations, corresponding author, active e-mail, and ORCID if available |
| P0 | Funding statement | Replace the placeholder with actual funding or the no-funding statement |
| P0 | Author contributions | Fill CRediT roles for all authors |
| P0 | Code availability | Provide anonymized review link, public repository, DOI, or a clear release-upon-acceptance statement |
| P0 | Data availability | Decide what derived result tables, logs, scripts, and T41 image examples can be shared |
| P0 | Ethics and consent | Confirm participant consent and anonymization policy for T41 indoor person images |
| P0 | Manual T41 annotation boundary | If real-scene AP is claimed, manually verify the 100 T41 frames; otherwise keep the current pseudo-reference wording |
| P0 | Template compliance | Open final DOCX in the official JRTIP template and verify spacing, captions, references, and page count |
| P0 | Final submission PDF | After author metadata and official-template formatting are complete, export and inspect the definitive submission PDF |
| P0 | Reproducibility environment | Add the exact server CPU/GPU, OS, Python, PyTorch, CUDA, Ultralytics revision, and T41 toolchain versions used for the reported runs |
| P0 | Cover letter | Write a cover letter emphasizing the real-time image-processing problem, not only mAP improvement |

## 4. Recommended But Not Blocking For First Submission

| Priority | Item | Why it helps |
|---|---|---|
| P1 | Python/C++ policy feature equivalence test | Enables a stronger claim that the T41 policy exactly reproduces the Python CAFRU features |
| P1 | More manually verified T41 frames | Upgrades the real-device section from pipeline feasibility to stronger real-scene detection evidence |
| P1 | Longer video sequence | Strengthens deadline-aware queue and stale-result claims |
| P1 | Outdoor or application-specific dataset | Tests domain transfer beyond COCO |
| P1 | Cleaner qualitative examples | Helps reviewers understand false-positive suppression, missed-object recovery, and skip decisions |

## 5. Evidence Boundaries To Preserve

Do not claim:

1. SpeciesNet annotations are ground truth unless they are manually verified.
2. T41 deployment is production-stable; current NNA instability remains a
   limitation.
3. The T41 board-policy run exactly reproduces the Python CAFRU operating
   point; current C++ features are approximate.
4. ROI review is the dominant contributor. Current evidence shows frame-level
   selection is the main contributor and ROI review is a limited supplement.
5. CAFRU exceeds cloud-only accuracy. Under the common deployment threshold
   and JPEG Q80, CAFRU is below cloud-only and should be framed as recovering
   part of the cloud gain at lower upload, compute, and latency cost.
6. The 6.6-6.8 FPS T41 queue result is full detector throughput. It excludes
   local YOLO26s loading and inference.

## 6. Final Submission Package Recommendation

Submit or prepare:

1. Main manuscript DOCX and PDF.
2. Online Resource 1 DOCX/PDF.
3. High-resolution or vector figures for Fig. 1, Fig. 2, and Fig. 5.
4. Source figures for Fig. 3 and Fig. 4.
5. An anonymized code/result archive, or a code-availability statement.
6. Cover letter.
7. Any required conflict-of-interest, ethics, and author contribution forms.
