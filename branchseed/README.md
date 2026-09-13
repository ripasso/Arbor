# Branchseed — discovering aortic branch origins

Given a CT volume and a binary mask of the parent abdominal aorta only, this
finds every artery that leaves that aorta and reports each one as a separate
daughter instance: the centre of its opening, a seed 5 mm along it, its initial
direction and its local lumen radius, in physical millimetres.

Classical image processing throughout. No training data, no GPU, no network
access, no per-case parameters.

## Setup

```
pip install -r requirements.txt
```

## Run

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

`--case-id` is optional; without it the case takes the name of the folder the
image sits in. Volumes may be `.nii` or `.nii.gz`, and the loader sniffs the
file rather than trusting the extension, because part of the development set
ships gzip-compressed under a plain `.nii` name.

To reproduce the whole development set and the figures:

```
python run_all.py            # writes out/predictions/*.json and out/assets/*
```

## Output

```json
{
  "case_id": "subject001",
  "parent": { "instance_id": "aorta" },
  "daughters": [
    {
      "instance_id": "branch_001",
      "parent_instance_id": "aorta",
      "ostium_xyz_mm": [12.4, -31.8, 184.6],
      "seed_xyz_mm": [15.1, -29.7, 181.2],
      "radius_mm": 2.7,
      "direction_xyz": [0.56, 0.43, -0.71]
    }
  ]
}
```

A case whose parent lumen is not opacified also carries a `quality` block saying
so, since on such a study nothing below the wall can be separated from muscle by
brightness and the output is a best effort under a much higher bar.

Coordinates are in the physical frame `SimpleITK.TransformIndexToPhysicalPoint`
returns, not voxel indices. `direction_xyz` is a unit vector pointing from the
opening into the daughter. Each real daughter appears once. A case with no
eligible daughter returns an empty `daughters` list rather than a guess.

## Method

**1 — Learn this scan's contrast.** Aortic opacification across the development
set runs from about 85 HU to 580 HU, so no fixed threshold survives. The window
is read off the supplied lumen, then pushed down a ladder only as far as the
wall will take: each step is scored by how many separate, plausibly sized
patches of bright tissue touch the wall, and the threshold that resolves the
most openings without any single patch swallowing more than about an eighth of
the wall is the one used. Too high and faint branches never reach the wall; too
low and adjacent liver and bowel fuse the footprints into one sheet. The peak
between those failures is a stable operating point.

**1b — Refuse scans that are not angiograms.** The whole method rests on a
daughter lumen being brighter than the tissue it runs through, which is only
true when the parent is opacified. The lower quartile of the aortic lumen is
tested against an absolute floor of 150 HU: iodinated arterial blood sits far
above it, unopacified blood sits near 40. Across the development set this reads
194 HU or higher on twenty-three cases and 50 and 18 on subjects 18 and 24,
which are not contrast studies at all. Above the floor nothing changes. Below
it, the acceptance bar rises to something only a genuinely bright, tubular
structure can clear, and the case is flagged in the output. An earlier build
returned 34 and 16 daughters on those two scans; every one was noise.

Comparing the lumen against its local background instead does not work here:
the vertebral body and other opacified vessels sit in that background and drag
it up on perfectly good scans.

**2 — One footprint, one ostium.** Where bright tissue meets the wall it leaves
a patch, and each patch is one daughter instance. That is the rule the brief
sets out: two origins are separate only when they are separate at the wall, and
a common trunk counts once however soon it divides afterwards.

**3 — Grow outward, never sideways.** Those footprint labels are then propagated
away from the wall in order of distance from the aortic surface. A voxel can
only inherit a label from something nearer the aorta than itself, so two
branches can never fuse into a single blob, and a leak into neighbouring tissue
stays attached to the origin it came from instead of swallowing the abdomen.
That containment is what makes a permissive threshold safe to use.

**3b — Settle the count away from the wall, not at it.** One footprint, one
ostium only holds if the footprints are separate at the threshold the scan
forced on us, and a permissive threshold runs two neighbouring contact patches
together. The count then collapses silently: one patch, one ostium, and the
second artery is gone. So each label is probed outward, the distance is taken at
which it separates into the most arms that are each large enough to be an artery
and each persist at least 3 mm further out, and those arms are propagated back
inward to the wall so every one owns its own piece of the footprint. It is the
mirror of the growth in step 3. This was the single largest source of missed
branches and it is invisible without a reference, because the output looks
perfectly reasonable either way.

**4 — Walk the first ten millimetres.** Each branch is followed outward shell by
shell to the first bifurcation or 10 mm, whichever comes first. The seed is the
point 5 mm along that path. The direction is a line fit over three times the
vessel's own radius. The radius is measured on a half-maximum contour in the
plane across the vessel at the seed, which keeps it sub-voxel on 1.5 mm scans
instead of stepping in half-voxel jumps. A shell is about one voxel thick, so a
vessel running obliquely can miss one entirely; two or three empty shells are
tolerated before the trace is called finished, because treating the first gap as
the end of the vessel was deleting whole branches before they reached a filter.

**Taken from prior work.** The pipeline was written from first principles and
then checked against the literature on the same task, which corrected two
things:

* *The centre of an opening is not the centroid of its footprint.* Tahoces et
  al. (Med Biol Eng Comput 2020) take the contact point as the voxel furthest
  from the edge of the contact area. An opening cut obliquely through the wall
  is often crescent shaped, and on 467 contact patches measured here the
  centroid fell outside its own patch 26.8% of the time, usually landing inside
  the aortic lumen; the rim-distance maximum is inside every time. The reported
  ostium moves by a median of 1.5 mm, against an inter-observer agreement of
  2.5 mm reported in that same paper. The distance has to be measured along the
  wall: the contact patch is one voxel thick, so an ordinary distance transform
  of it is half a voxel everywhere and its maximum falls wherever ties break.
* *The direction window should scale with the vessel.* Riffaud et al. (Med Biol
  Eng Comput 2022) fit a branch's direction over three times its own radius. A
  flat window makes a 1 mm lumbar artery and a 4 mm renal share a fit, so the
  radius is measured on a provisional fit and the direction fitted again at the
  right scale.

One thing from the literature did not survive contact with this data. Danilov
et al. (Computation 2016) clean vessel masks by walking distance layers inward
from the farthest one and dropping voxels with no neighbour further out. Ported
directly, it anchors on each label's farthest point, which here is frequently a
leak rather than the vessel, so it kept the leak and deleted the branch. It
cost subject 2 both its coeliac and mesenteric arteries and was reverted.

**Geometry.** The centerline is a geodesic between the two ends of the lumen,
traced through a cost field that prefers the centre so it does not hug the wall
on curves. Cross-sectional frames are anchored to the patient's anterior
direction at every station rather than transported along the curve, so clock
position means the same thing at every level and in every case — which is what
makes the maps comparable across a cohort.

**What is deliberately excluded.** Origins sitting on the flat faces created by
cropping the segment; wall-hugging sheets whose opening is out of all proportion
to the lumen they feed; anything too bright to be contrast, which is calcium or
bone; anything that cannot be followed 5 mm. The terminal division of the aorta
into the iliacs is reported separately from the daughters, since the brief puts
it outside the core task.

## Files

| Path | What it is |
|---|---|
| `run.py` | the required CLI, one case in, one JSON out |
| `pipeline.py` | per-case orchestration, filters, and the prediction record |
| `branches.py` | contrast window, outward labelled growth, tracing, radius |
| `aorta.py` | mask cleanup, centerline, patient-anchored frames, wall unwrapping |
| `bsio.py` | NIfTI reading and the voxel-to-physical-millimetre conversion |
| `export.py` | the figures and the atlas data the visualisation is built from |
| `run_all.py` | batch driver over a folder of `subjectNNN/` cases |
| `score.py` | scores a filter setting against a reference set, with a sweep |
| `diag.py` | per reference daughter: matched, or which filter refused it |
| `whymiss.py` | for a miss with no nearby candidate, which stage lost it |
| `publish.py` | assembles `site/` from `out/`, and names anything missing |
| `site/` | the interactive visualisation, generated entirely from the outputs |

## Visual checks

`out/assets/` holds, for every case, the flattened wall map with its detected
origins, and a sprite sheet of patient-aligned axial slices with the supplied
mask overlaid. The page in `site/` puts them together: pick an origin on the
flattened map and the slice viewer jumps to the level it was found on, with the
direction vector drawn as it projects into that slice.

## Accuracy

A draft reference exists for five of the twenty-five cases, 19 daughter
instances in all. Against it: F1 0.491, recall 0.684, precision 0.382, median
ostium error 1.47 mm against an inter-observer agreement of 2.5 mm. Its authors
describe it as expert-review-pending and state that it is not exhaustive, so
**recall against it is meaningful and precision is pessimistic by an unknown
amount**. `EVALUATION.md` has the full account, including the three defects it
exposed, which thresholds moved and why, and two improvements that were measured
and then deliberately not taken.

```
python score.py       # sweep every threshold against the reference, one at a time
python diag.py        # per reference daughter: matched, or which filter refused it
```

## Runtime

Single-threaded, CPU only, peak memory under 2 GB. Median 13 s per case, slowest
26 s, well inside the 60 s budget. Cost scales with the length of aorta supplied
rather than with file size. Per-case timings for the whole development set are in
the table at the bottom of the visualisation.
