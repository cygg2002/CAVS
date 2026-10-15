# Spec: Oregon YOLO26x Cloud-V4 additional 30-epoch fine-tune

## Objective

Train a new isolated Oregon YOLO26x O2M checkpoint for 30 epochs from the
completed Cloud-V3 `best.pt`, using the same recipe as Cloud-V3. The experiment
tests whether another equal-length fine-tuning stage improves validation
AP50-95 beyond Cloud-V3's epoch-28 best (0.54282).

This is a fresh fine-tuning stage from the best weights. It intentionally does
not resume Cloud-V3's epoch-30 optimizer state.

## Commands

- Test: `python -m unittest discover -s tests -v`
- Preflight: `python scripts/train_cloudv4.py --preflight-only`
- Train: `python scripts/train_cloudv4.py`

## Project Structure

- `configs/protocol.json`: frozen source provenance and training recipe
- `scripts/train_cloudv4.py`: validated training entry point
- `tests/test_protocol.py`: protocol and import checks
- `train.log`: background process output, created at launch

## Code Style

The runner follows the existing Cloud-V3 entry point: standard-library
validation first, followed by a local `ultralytics` import only when training.

## Testing Strategy

Unit tests freeze the source checkpoint identity, output isolation, epochs,
batch size, image size, optimizer, seed, and non-resume semantics. Preflight
then verifies all source files and the checkpoint SHA-256 on the server.

## Boundaries

- Always: use the same Oregon detector split, O2M model YAML, and Cloud-V3 recipe.
- Always: write to a new experiment and run directory.
- Never: overwrite or resume inside the completed Cloud-V3 directory.
- Never: change detector data, class mapping, image size, or batch size silently.

## Success Criteria

- Protocol tests and server preflight pass.
- A detached training process starts with batch 8 and produces its own run.
- Training reaches 30/30 epochs and produces `best.pt`, `last.pt`, and 30 metric rows.
- The best AP50-95 is compared against Cloud-V3 best AP50-95=0.54282.

## Open Questions

None. The user requested another 30 epochs; the frozen interpretation is a new
30-epoch stage initialized from Cloud-V3 best weights.
