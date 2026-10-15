# Spec: COCO CAVS-OQR Edge-source diagnosis

## Objective

Quantify whether the gap between the published cached CAVS-OQR result and
the RK3588 FP16 result is primarily associated with the switch from
batch-shaped `rect=True` cache inference to one-image inference, or with
the RKNN FP16 Edge output itself.

## Inputs

- Published Current+MAUR point results and selections.
- Frozen-policy PyTorch one-image deployment result.
- Frozen-policy RK3588 FP16 deployment result.

All inputs must cover the same 5,000 COCO val2017 image IDs and use the
same three fixed-byte budgets: 10%, 25%, and 40%.

## Method

For each budget, compare the published selection with the PyTorch
one-image and RKNN FP16 selections. Report exact action agreement, uploaded
set Jaccard, shared-upload action agreement, action-transition counts, and
the recorded AP50-95 difference. The script only reads existing artifacts;
it does not infer, upload, train, or overwrite upstream results.

## Success Criteria

- Every compared plan contains exactly 5,000 identical image IDs.
- PyTorch one-image and RKNN FP16 runs have the same frozen policy SHA-256.
- Every plan satisfies its recorded fixed-byte capacity.
- The report contains all three budgets and writes both JSON and Markdown.

## Boundaries

- Never modify source experiment directories or board artifacts.
- Never relabel a cached point result as a physical deployment result.
- Never attribute the residual RKNN difference to quantization without the
  source-isolation comparison.
