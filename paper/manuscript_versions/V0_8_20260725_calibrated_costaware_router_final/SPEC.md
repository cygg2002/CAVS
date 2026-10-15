# Spec: JRTIP V0.8 Chinese Final Manuscript Package

## Objective

Preserve the V0.7 draft and create a content-complete Chinese manuscript package
whose scientific claims match the frozen evidence available on 2026-07-25. The
package contains a revised Markdown manuscript, four source-backed figures, an
editable double-column Word manuscript, revision notes, and build/QA artifacts.

## Tech Stack

- Markdown for the auditable source manuscript.
- Python 3.12 standard library and Pillow for deterministic figures.
- `python-docx` through the existing JRTIP V0.6 builder for Word generation.
- LibreOffice headless rendering plus Poppler for visual QA.

## Commands

```bash
PY=/Users/cygg_macmini/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
$PY -m unittest discover -s tests -v
$PY build_figures.py
$PY build_docx.py
FONTCONFIG_FILE="$PWD/fontconfig-macos.xml" soffice \
  -env:UserInstallation=file:///tmp/lo-v08-render \
  --headless --convert-to pdf --outdir rendered \
  JRTIP_MANUSCRIPT_V0_8_ZH_FINAL.docx
```

## Project Structure

```text
JRTIP_MANUSCRIPT_V0_8_ZH_FINAL.md  final Chinese source
JRTIP_MANUSCRIPT_V0_8_ZH_FINAL.docx editable Word manuscript
figures/                           generated manuscript figures
rendered/                          PDF and page previews used for QA
tests/                             figure/data contract tests
build_figures.py                   frozen-CSV figure builder
build_docx.py                      thin wrapper around the existing Word builder
REVISION_NOTES.md                  evidence and revision audit
```

## Code Style

Use small pure helpers, explicit paths relative to the repository, no new
dependencies, and no mutation of source experiment artifacts.

## Testing Strategy

- Unit-test row counts, gate categories, and regret summaries against the frozen
  CSV/JSON artifacts.
- Build all figures and assert non-empty PNG outputs.
- Inspect the DOCX XML for remaining scientific placeholders.
- Render the DOCX to PDF and inspect page count plus representative page images.
- Use the package fontconfig file during LibreOffice conversion so macOS CJK
  fonts remain visible in the PDF.

## Boundaries

- Always: preserve V0.7, cite frozen artifacts, label E2E as estimated, retain
  negative/fallback results, and distinguish identical-action passes.
- Ask first: changing author identity, affiliations, funding, dataset licenses,
  or public repository URLs.
- Never: select methods using final, fabricate the unfinished global-one-config
  ablation, call pointwise bootstrap a formal guarantee, or invent measured E2E.

## Success Criteria

- No scientific `待填`, `TBD`, or post-final placeholder remains.
- All 30 frozen work points and all three oracle-regret summaries are traceable.
- The manuscript reports 13/30 gate passes as 6 identical plus 7 non-identical.
- Four figures are generated from immutable experiment outputs.
- Word generation and PDF rendering succeed, with figures and tables visible.
- The only remaining completion items are author-controlled submission metadata.

## Open Questions

- Final author names, affiliations, corresponding-author details, CRediT roles,
  funding, dataset licenses/URLs, and public code DOI remain author decisions.
