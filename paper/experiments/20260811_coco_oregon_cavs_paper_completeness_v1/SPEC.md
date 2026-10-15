# COCO/Oregon CAVS paper-completeness diagnostics v1

## Scope

This isolated, post-hoc cache-level experiment fills four documented gaps in the
completed canonical Cloud-V3 benchmark. It must not read or write
`20260811_coco_currentoqr12_profile_scale_extension_v1`.

1. Direct Raw CAVS paired bootstrap against the frozen strongest ORIC/MORIC,
   Confidence, Random, and DualHead selections. COCO uses image resampling;
   Oregon uses event-block primary and image-block sensitivity resampling.
2. Raw CAVS action-mask ablation: Edge+Original, Edge+Q90, and OQ Joint.
3. Oregon DCSB threshold fitting from the canonical fit role only, using the
   existing external-baseline implementation in a new output directory.
4. Class/size (COCO), class/size/camera/event (Oregon) grouped effects and a
   deterministic two-action exact-solver audit against the frozen heuristic.

## Frozen protocol

- canonical roles: COCO 9,678/571/5,000; Oregon 9,923/4,961/4,962;
- fixed-byte primary and fixed-call sensitivity at 10%, 25%, 40%;
- one thousand deterministic paired bootstrap replicates per comparison;
- no detector inference, no learner training, no calibration threshold fitting;
- DCSB is the sole exception and is fit only on Oregon fit=9,923;
- final is evaluation-only and never used for training or policy selection.

All values are post-hoc evidence. This experiment does not create a hidden
registered final and cannot be used to tune a later method.
