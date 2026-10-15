# Spec: Oregon YOLO26x Cloud-V5 additional 30-epoch fine-tune

## Objective

Train a new isolated Oregon YOLO26x O2M checkpoint for 30 epochs from the
completed Cloud-V4 `best.pt`, using exactly the frozen Cloud-V4 recipe. The
experiment tests whether one more equal-length fine-tuning stage improves
validation AP50-95 beyond Cloud-V4's epoch-29 best of 0.55068.

This is a fresh stage initialized from the best weights. It deliberately resets
the optimizer and warmup instead of resuming Cloud-V4's final optimizer state.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/train_cloudv5.py --preflight-only`
- Train: `python scripts/train_cloudv5.py`

## Project Structure

- `configs/protocol.json`: frozen source provenance and training recipe
- `scripts/train_cloudv5.py`: validated training entry point
- `tests/test_protocol.py`: protocol and import checks
- `train.log`: detached process output, created at launch

## Code Style

Use the minimal Cloud-V4 launcher pattern: validate paths, checkpoint SHA and
frozen settings with the standard library, then import `ultralytics` only for
the actual training call.

## Testing Strategy

Unit tests require the Cloud-V5 train dictionary to equal Cloud-V4's dictionary
and freeze the source checkpoint, source hash, isolated destination and
non-resume semantics. Server preflight verifies the actual source files and
SHA-256 before launch.

## Boundaries

- Always: use the same Oregon detector split and O2M model YAML as Cloud-V4.
- Always: use batch 8, image size 640, seed 0 and a fresh 30-epoch stage.
- Always: write to a new Cloud-V5 run directory.
- Never: modify or resume inside Cloud-V4.
- Never: silently change the detector data, class map or training recipe.

## Success Criteria

- Unit tests and remote preflight pass.
- A detached Cloud-V5 process starts on GPU 0 with the frozen batch 8 recipe.
- The new run directory and first training output are created without changing
  Cloud-V4.
- At completion, `results.csv` contains 30 rows and `best.pt`/`last.pt` exist.
- Cloud-V5's best AP50-95 is compared with Cloud-V4 best AP50-95=0.55068.

## Open Questions

None. The user explicitly requested another 30 epochs from Cloud-V4 best.
