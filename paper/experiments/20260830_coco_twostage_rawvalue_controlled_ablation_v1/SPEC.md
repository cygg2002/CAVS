# COCO raw-value TwoStage controlled ablation

## Question

Under the frozen CAVS-OQ scores and detector caches, what is lost when every
image is first projected to one cloud action before Fixed-byte capacity
selection, instead of retaining both Original and Q90 candidates until the
joint image-action scheduler runs?

## Controlled protocol

- Reuse the exact three frozen CAVS-OQ full-fit score tensors from the
  20260813 final-completion experiment; do not retrain any learner.
- Reuse the same COCO final split, detector predictions, action byte costs,
  Reversible fusion, budgets, and Fixed-byte solver.
- For each image, choose

  `a_i* = argmax_{a in {Original,Q90}} v(i,a)` and `s_i = v(i,a_i*)`.

- Retain only `{Edge, a_i*}` for that image and pass the **raw** `s_i` to the
  same capacity solver. No empirical percentile or other score transform is
  allowed.
- Evaluate all three individual seeds and their score ensemble at 10%, 25%,
  and 40% budgets.

This isolates the scheduling granularity: joint image-action capacity
scheduling versus per-image action projection followed by capacity selection.
It does not test a separate scalar-value learner.

## Acceptance checks

- Unit tests prove raw scores are preserved and tie-breaking is deterministic.
- Each result satisfies the same Fixed-byte capacity check as the source
  experiment.
- Source result and score artifacts are read-only; outputs are written only
  under this experiment directory.
- Results include all 12 point evaluations (3 seeds + ensemble, 3 budgets),
  source CAVS-OQ AP, and their paired point-estimate difference.
