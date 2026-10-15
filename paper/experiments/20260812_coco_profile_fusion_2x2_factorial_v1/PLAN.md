# Plan

1. Freeze and validate source hashes, canonical roles, action geometry and the
   completed Current-Reversible anchor.
2. Build current@640 and historical_cost@640 MAUR action files from the exact raw
   ROI predictions and geometry already used by Reversible.
3. Generate fold-contained fit and calibration contextual rewards only for the
   two MAUR actions; reuse the frozen Reversible reward map.
4. Run the calibration Teacher 2×2 gate without reading final.
5. If the gate opens, train the three missing cells for five OOF folds plus full-fit
   using the three frozen seeds; reuse only the exact Current-Reversible anchor.
6. Evaluate fit OOF, calibration and public val2017 once under Fixed-byte and
   Fixed-call 10/25/40.
7. Run factorial paired bootstrap and write a Chinese report with main effects,
   interaction, provenance and limitations.
