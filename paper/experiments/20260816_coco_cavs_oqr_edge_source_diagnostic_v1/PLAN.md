# Plan: COCO CAVS-OQR Edge-source diagnosis

1. Normalize the published sparse selections and the two complete plans to
   one action per COCO image. Verify: each plan has the same 5,000 IDs.
2. Compare plans and point results at 10%, 25%, and 40%. Verify: action
   agreement, upload-set Jaccard, and transition counts reconcile.
3. Check the frozen policy SHA and capacity constraints, then emit an
   immutable result report. Verify: the diagnostic test and full script pass.
