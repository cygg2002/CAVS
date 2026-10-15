# Implementation plan

1. Freeze the protocol, source paths, role counts, method matrix, and protocol
   separation rules.
2. Add pure contract tests for action masking, capacity, row uniqueness, source
   tiering, final-use restrictions, and expected comparison counts.
3. Build a source adapter that loads canonical cached predictions and frozen
   checkpoints without modifying any source experiment.
4. In parallel per dataset, produce final Raw/Calibrated ActionValue scores and
   evaluate the Track A matrix.
5. Run 50-seed Random and 1,000-replicate paired bootstrap with bounded CPU
   process pools.
6. Fingerprint and import completed Track B/C/D results into explicitly separated
   tables.
7. Validate row counts, capacities, identities, source hashes, and result tiers;
   write Chinese and machine-readable summaries.
8. Launch the resumable pipeline with `nohup`; do not continuously poll it.

