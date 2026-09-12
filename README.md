# Arbor

Detect direct daughter arteries branching off the abdominal aorta from a CT volume + binary aorta mask.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

## Project structure

See `../toralis-branchseed-program-outline.md` for the full module map, data flow, and team work-split. Quick summary:

| File | Stage | Owner |
|---|---|---|
| `io_utils.py` | A — loading | Person 1 |
| `roi.py` | B — search shell | Person 2 |
| `region_growing.py` | C — primary detector | Person 2 |
| `vesselness.py` | C — secondary filter | Person 2 |
| `components.py` | D — connected components | Person 3 |
| `ostium.py` | E — ostium + dedup | Person 3 |
| `tracing.py` | F — eligibility + centerline | Person 4 |
| `measurements.py` | G — seed/radius/direction | Person 4 |
| `output.py` | H — JSON writer | Person 1 |
| `visualize.py` | I — sanity-check figures | Person 4 |
| `validate.py` | local scoring | Person 4 |
| `run.py` | CLI / integration | rotate |

## Status

Skeleton only — every function is a documented stub (`raise NotImplementedError`). See docstrings for the contract each function must satisfy. Run `pytest tests/` to see the (currently skipped) test scaffolding for each module.

## Background docs

- `../toralis-fundamentals.md` — challenge spec
- `../toralis-branchseed-solution-plan.md` — pipeline plan
- `../toralis-branchseed-math.md` — math behind each stage
- `../toralis-branchseed-program-outline.md` — this structure + work split
