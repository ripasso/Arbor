# Branchseed Challenge — Program Outline & Work Split

This doc turns the solution plan (`toralis-branchseed-solution-plan.md`) into a concrete **file/module structure** and a **team work-split**, so multiple people can build in parallel without blocking each other. Still no implementation — this is architecture (what files exist, what each function's job is) and team logistics.

---

## 1. Proposed file structure

```
branchseed/
  run.py                  # CLI entry point (--image, --aorta-mask, --output)
  config.py                # tunable constants (dilation radius, Frangi params, thresholds)

  io_utils.py              # Stage A: loading
  roi.py                   # Stage B: search shell
  vesselness.py            # Stage C: Frangi/Hessian tube enhancement
  region_growing.py        # Stage C (primary detector per §11 math doc): connectivity-based growth
  components.py            # Stage D: connected components + aorta-adjacency test
  ostium.py                # Stage E: contact centroid, dedup, planarity/crop-face rejection
  tracing.py               # Stage F: centerline, eligibility check, proximal segment
  measurements.py          # Stage G: seed/radius/direction computation
  output.py                # Stage H: JSON schema writer
  visualize.py             # Stage I: required sanity-check figures

  validate.py              # local P/R/F1 + distance scoring against dev-set references
  tests/
    test_io_utils.py
    test_roi.py
    test_components.py
    test_ostium.py
    test_tracing.py
    test_output.py

  data/                     # (gitignored) local copy of the dataset for dev/testing
  outputs/                  # generated prediction JSONs + figures
```

---

## 2. Module responsibilities & interfaces

Each module's job, inputs, and outputs — no logic, just contracts. This is what lets people work independently: everyone agrees on these shapes up front.

### `io_utils.py` — Stage A
- `load_case(image_path, mask_path) -> (image: sitk.Image, mask: sitk.Image)`
  Handles gzip-disguised files (magic-byte sniff), verifies matching geometry.
- `voxel_to_mm(image, index) -> (x, y, z)`
  Thin wrapper over `TransformIndexToPhysicalPoint`.

### `roi.py` — Stage B
- `build_search_shell(mask, margin_mm) -> shell_mask`
  Dilate mask by `margin_mm` (converted to voxels via spacing), subtract original mask.

### `vesselness.py` — Stage C (secondary filter)
- `compute_vesselness(image, roi_mask, scales_mm) -> vesselness_map`
  Multi-scale Frangi score, restricted to ROI voxels only (performance).

### `region_growing.py` — Stage C (primary detector)
- `grow_from_mask(image, mask, hu_low, hu_high) -> candidate_mask`
  Region-grow outward from the aorta mask through intensity-consistent voxels; `hu_low`/`hu_high` come from per-case calibration (see `config.py` / calibration step below).
- `calibrate_hu_range(image, mask) -> (hu_low, hu_high)`
  Computes intensity stats **inside the given aorta mask** to set a per-patient bright-blood threshold (mitigation from math doc §11).

### `components.py` — Stage D
- `label_components(candidate_mask) -> list[Component]`
  `Component` = voxel set + bounding box + touches_aorta flag.
- `filter_touching_aorta(components, mask) -> list[Component]`

### `ostium.py` — Stage E
- `find_contact_centroid(component, mask) -> voxel_index`
- `is_cropped_face(contact_point, mask) -> bool` (planarity/z-extreme test)
- `deduplicate_ostia(candidates) -> list[Candidate]` (merge/split rule handling)

### `tracing.py` — Stage F
- `extract_centerline(component) -> ordered_list[voxel_index]` (skeleton + path ordering)
- `check_eligibility(centerline, spacing_mm) -> bool` (≥5mm traceable)
- `trim_to_proximal_segment(centerline, spacing_mm) -> ordered_list[voxel_index]` (≤10mm or first skeleton bifurcation)

### `measurements.py` — Stage G
- `get_seed_point(centerline, ostium, spacing_mm) -> (x, y, z)` (5mm along path)
- `get_radius(mask_or_candidate, seed_point, spacing_mm) -> float` (EDT value)
- `get_direction(centerline_segment) -> unit_vector` (PCA / normalized diff)

### `output.py` — Stage H
- `build_json(case_id, daughters: list[Daughter]) -> dict` (matches required schema)
- `write_json(dict, output_path)`

### `visualize.py` — Stage I
- `plot_case(image, mask, ostia, directions, save_path)`

### `validate.py`
- `match_predictions_to_references(pred_json, ref_json, tolerance_mm) -> (TP, FP, FN)`
- `score_case(pred_json, ref_json) -> dict` (P/R/F1, mean ostium distance, etc.)
- `score_all(pred_dir, ref_dir) -> summary_table`

---

## 3. Data flow (one glance)

```
image, mask
   │
   ▼
[io_utils] ── loaded volumes, verified geometry
   │
   ▼
[roi] ── search shell
   │
   ├──────────────┐
   ▼              ▼
[region_growing] [vesselness]   (primary + secondary signal)
   │              │
   └──────┬───────┘
          ▼
     candidate_mask
          │
          ▼
   [components] ── components touching aorta
          │
          ▼
     [ostium] ── deduped ostium candidates (crop-face filtered)
          │
          ▼
    [tracing] ── eligible centerlines, proximal segments
          │
          ▼
  [measurements] ── seed, radius, direction per branch
          │
          ▼
     [output] ── prediction.json
          │
          ▼
   [visualize] ── sanity-check figures (≥3 cases)

(separately) [validate] compares prediction.json vs dev-set references → local score
```

---

## 4. Suggested work split (assumes a 4-person team)

Split along the **dependency chain**, with shared interface contracts (section 2) agreed on *first* so people can stub/mock and work in parallel instead of waiting on each other.

### Person 1 — Data & I/O foundation (blocking for everyone, do first/fast)
- `io_utils.py` (loading, gzip handling, geometry checks, voxel↔mm helper)
- `config.py` skeleton
- `output.py` (JSON schema writer — schema is fixed by the spec, doesn't depend on the rest of the pipeline)
- Sets up `tests/` scaffolding and a couple of known-good sample cases everyone can test against

> Why first: nothing else can be tested end-to-end until loading works. This person should finish fastest and then float to help others.

### Person 2 — Detection core
- `roi.py` (search shell)
- `region_growing.py` (primary detector + per-case HU calibration)
- `vesselness.py` (secondary Frangi filter)

> This is the most algorithmically involved piece (per math doc §2 and §11) — pair with Person 3 if possible once components/ostium logic needs real candidate masks to test against.

### Person 3 — Instance logic
- `components.py` (connected components + adjacency test)
- `ostium.py` (contact centroid, crop-face rejection, dedup/merge-split rules)

> Can start immediately using a **fake/synthetic candidate mask** (e.g., a hand-made small binary blob) as a stub, without waiting on Person 2's real detector — swap in the real one once ready. This is the key parallelization trick.

### Person 4 — Measurements, validation, visualization
- `tracing.py` (centerline, eligibility, proximal trim)
- `measurements.py` (seed, radius, direction)
- `validate.py` (local P/R/F1 scoring against dev-set references)
- `visualize.py` (required figures)

> Can also start with a synthetic/stubbed centerline before Person 3's real ostium output exists. `validate.py` is independent of the whole pipeline (just compares two JSON files) — good to build early since it's needed to test everything else later.

### Integration owner (rotate, or Person 1 once free)
- Wire everything together in `run.py`
- Run the full pipeline end-to-end on real dev-set cases
- Own the README + 5-minute demo + failure-case notes for submission

---

## 5. Suggested build order (mirrors §5 of the solution plan)

1. **Hour 0–1**: Agree on interfaces (section 2 above) as a team. Person 1 gets `io_utils` + `output` working — unblocks everyone.
2. **Hour 1–4 (parallel)**: Person 2 builds detection; Person 3 builds instance logic against a stub mask; Person 4 builds measurements against a stub centerline + starts `validate.py`.
3. **Hour 4–6**: Integrate — swap stubs for real modules, run on 1–2 real dev cases, fix interface mismatches.
4. **Hour 6+**: Run `validate.py` against dev-set references, tune thresholds (calibration, dilation margin, Frangi knobs), iterate.
5. **Final hour(s)**: visualizations, README, demo prep.

---

## 6. Why this split works

- The **stub-first trick** (Person 3 and 4 building against fake intermediate data before Person 2's real detector is ready) is what actually lets 4 people work in parallel on what is otherwise a strictly sequential pipeline (each stage needs the previous stage's output).
- `output.py` and `validate.py` are **schema-only** — they don't need any of the vision algorithms to be done first, so they're good early/independent tasks.
- The riskiest, most time-consuming piece (per the math doc's critical evaluation) is **detection + false-positive suppression** (Person 2's module) — giving it a dedicated owner (and a floating helper once Person 1 is free) reflects where the real difficulty lives.
