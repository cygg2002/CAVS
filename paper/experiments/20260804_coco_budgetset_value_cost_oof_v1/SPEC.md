# Spec: COCO BudgetSet Value/Cost OOF Diagnostic v1

## Objective

Determine whether the registered BudgetSet failure is primarily caused by
entangling contextual AP value with payload-aware boundary supervision. Compare
the frozen current BudgetSet against a cost-free SetValue model and a DualHead
model on five-fold out-of-fold predictions over the same 5,000 development
images.

This is a development-only diagnostic. It cannot access audit or registered
final predictions, annotations, metrics, checkpoints, or reports, and it cannot
be reported as a new paper-level result.

## Frozen Contract

- Source: only the `development` role and its frozen Edge/Cloud Original caches
  from `20260804_coco_budgetset_registered_paper_v1`.
- Detector, payload, AP evaluator, budgets, ORIC/MORIC implementation, model
  dimensions, optimizer, epoch limit, and three seeds remain unchanged.
- Five deterministic folds cover all 5,000 images exactly once as OOF targets.
- Existing development contextual AP rewards are reused. Their 32-image
  contexts may cross OOF folds; this transductive-within-development limitation
  is disclosed and is shared by every learned method.
- No detector inference is run.

## Methods

1. `Current-BudgetSet`: frozen regression, within-image, and CostEligible
   boundary loss with the existing cost-bearing inputs.
2. `SetValue`: same set encoder and optimizer, but cost-bearing scene/action
   features are zeroed, the budget input is fixed to zero, and only contextual
   AP50-95 regression plus within-image ordering are trained.
3. `DualHead`: the same cost-free value path plus a separate utility head. The
   utility head alone receives log payload and budget and is trained with the
   CostEligible boundary loss. Deployment always uses the value head; exact
   payload enters fixed-byte scheduling once, through CostEligible.
4. Frozen OOF controls: `ORIC-AP50`, `MORIC-AP50`, `ORIC-AP5095`, and
   `MORIC-AP5095`.

## Evaluation

- Fixed-call: exact 10%, 25%, and 40% top-K selection by the deployed value.
- Fixed-byte: the same value is passed to CostEligible with exact Original-byte
  capacity.
- Report Official AP50-95/AP50/AP75, KB/image, score-vs-contextual-gain
  Spearman, score-vs-log-payload Spearman, and top-K overlap with the
  contextual-gain teacher.
- The strongest external OOF control is selected independently at each
  fixed-call budget.

## Commands

```bash
python -m unittest discover \
  -s paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/tests -v

python paper/experiments/20260804_coco_budgetset_value_cost_oof_v1/scripts/run_oof.py
```

## Project Structure

```text
configs/protocol.json  source binding, folds, methods, and frozen gate
scripts/oof_core.py    pure fold, feature-isolation, dual-head, and metrics code
scripts/run_oof.py     development-only training/evaluation pipeline
tests/                 unit and source-boundary tests
runs/                  resumable OOF artifacts and logs
notes/results_zh.md    generated Chinese interpretation
```

## Code Style

Use typed, deterministic Python functions and additive experiment-local files.
Do not modify either completed source experiment.

```python
folds = deterministic_folds(image_ids, fold_count=5, seed=20260804)
assert sorted(image_id for fold in folds for image_id in fold) == sorted(image_ids)
```

## Testing Strategy

- Unit-test fold isolation, feature stripping, value/utility separation,
  ranking diagnostics, and gate behavior before implementation.
- Source-contract tests forbid audit/final path access and require the role to
  remain exactly `development`.
- Run a small synthetic training smoke before the full remote process.

## Boundaries

- Always: keep all writes inside this experiment; record hashes and OOF fold
  membership; report negative outcomes.
- Ask first: change source development population, rewards, detector protocol,
  model dimensions, budgets, seeds, or success gate.
- Never: read audit/final predictions or metrics; tune on audit/final; describe
  the diagnostic as paper-final evidence.

## Success Criteria

- All 5,000 development images receive exactly one OOF prediction per method.
- `SetValue` and `DualHead` value scores have no direct payload or budget input.
- Fixed-byte applies exact payload cost once, only in CostEligible.
- A candidate passes only if it beats the strongest OOF ORIC/MORIC control by
  at least `+0.001` AP at two budgets and loses no more than `0.001` at the
  remaining budget.
- The report distinguishes objective mismatch from representation failure.

## Open Questions

None. The user approved this diagnostic contract before implementation.
