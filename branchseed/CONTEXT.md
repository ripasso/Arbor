# Context

Everything a new session, or a person picking this up cold, needs to know.
`README.md` says how to run it. This file says why it is the way it is.

---

## 1. What this is

The Toralis Labs **Branchseed** hackathon challenge, entered in the category
*Most Unique Use of the Healthcare Dataset*.

**The task.** Given a CT volume and a binary mask of the parent abdominal aorta
*only*, detect every artery that leaves that aorta and return each as a separate
daughter instance. Per daughter: the centre of its opening, a seed 5 mm along
it, an initial direction unit vector, and the local lumen radius, in physical
millimetres. No anatomical names anywhere: `branch_001`, `branch_002`, and so
on. The daughters are visible in the CT but are never in the supplied mask.

**Required CLI.**

```
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

**Organiser environment.** Hidden test set, 4 CPU cores, 8 GB RAM, no GPU, no
network. Initial target 60 s per case.

**Grading.**

| Weight | Criterion | What is assessed |
|---|---|---|
| 45% | Branch discovery | precision, recall, F1 on the number of daughter instances |
| 25% | Ostium localisation | physical distance between predicted and reference ostium centres |
| 15% | Daughter-instance quality | seed lies on the matched daughter, direction follows its proximal path, radius consistent with the lumen |
| 10% | Compute efficiency | runtime and peak memory on the organiser's CPU-only system |
| 5% | Reproducibility | valid output, documented setup, runs on unseen cases |

A visual server is explicitly *nice to have, not required*.

---

## 2. Where things are

| What | Where |
|---|---|
| Dataset, 25 cases | `~/Downloads/TORALIS CHALLENGE/subjectNNN/` on Rastin's Mac |
| Submission zip | `~/Downloads/TORALIS CHALLENGE/branchseed_submission.zip` |
| Published page | Claude artifact "The Unrolled Aorta", in the user's artifact gallery |
| Predictions | `out/predictions/subjectNNN.json`, one per case |
| Figures and atlas data | `out/assets/`, consumed by `site/` |

**Data quirk that will bite you.** Subjects 16 to 25 are gzip-compressed but
named plain `.nii`. A reader that trusts the extension fails on ten of the
twenty-five. `bsio.load_nifti` sniffs the magic bytes instead.

**Second data quirk.** Subject 24's direction cosines are not orthonormal, so
SimpleITK refuses to open it at all. Our loader reads it. Worth mentioning in
the demo, and worth watching if the organiser's reference pipeline reads with
SimpleITK.

---

## 3. Current state

165 daughters across 25 cases, 552 candidates refused. Per-case counts run from
1 to 26; supplied aortic segments run from 57 mm to 490 mm, which is most of why
the counts differ so much.

Median 22.0 s per case, slowest 92.3 s, 9.7 minutes for the whole set, peak
resident memory 1.42 GB. Single-threaded, so three of the four allowed cores sit
idle. **One case exceeds the 60 s budget:** subject 18, at 92 s on a 322 mm
segment. Cost tracks the length of aorta supplied, not file size.

Two cases are flagged as not being angiograms: subject 18 and subject 24.

**There is no ground truth.** The development set ships no reference
annotations, so precision and recall cannot be computed here. Every validation
in this project is therefore intrinsic or comparative, never a score. Do not let
anyone present a number as accuracy.

---

## 4. How the method works, and why

Classical image processing end to end. No training data, no GPU, no network, no
per-case parameters.

**1. Learn this scan's contrast.** Aortic opacification across the set runs from
85 HU to 580 HU, so no fixed threshold survives. The window is read off the
supplied lumen, then pushed down a ladder only as far as the wall will take:
each rung is scored by how many separate, plausibly sized bright patches touch
the wall, and the rung that resolves the most openings without any single patch
swallowing more than about an eighth of the wall wins. Too high and faint
branches never reach the wall; too low and adjacent liver and bowel fuse the
footprints into one sheet. The peak between those two failures is stable.

**2. Refuse scans that are not angiograms.** The whole method rests on a
daughter lumen being brighter than the tissue around it, which is only true when
the parent is opacified. Test: lower quartile of the aortic lumen against an
absolute floor of 150 HU. Reads 194 HU or higher on twenty-three cases, 50 and
18 on subjects 18 and 24. The groups do not overlap. Below the floor the
acceptance bar rises steeply and the case is flagged. An earlier build returned
34 and 16 daughters on those two scans and every one was noise.

**3. One footprint, one ostium.** Where bright tissue meets the wall it leaves a
patch, and each patch is one daughter instance. This is exactly the rule the
brief sets out: two origins are separate only when they are separate at the
wall, and a common trunk counts once however soon it divides.

**4. Grow outward, never sideways.** Footprint labels propagate away from the
wall in order of distance from the aortic surface. A voxel can only inherit from
something nearer the aorta than itself, so two branches can never fuse into one
blob, and a leak into neighbouring tissue stays attached to the origin it came
from instead of swallowing the abdomen. **This containment is what makes a
permissive threshold safe, and it is the part of the method that is genuinely
ours.** None of the three reference papers does this.

**5. Walk the first ten millimetres.** Each branch is followed outward shell by
shell to the first bifurcation or 10 mm, whichever comes first. Seed is 5 mm
along that path, not 5 mm along a straight line, so it stays inside a vessel
that curves. Direction is a line fit over three times the vessel's own radius.
Radius is a half-maximum contour in the plane cut across the vessel at the seed,
which keeps it sub-voxel on 1.5 mm scans.

**Geometry.** Centerline is a geodesic between the two ends of the lumen through
a cost field that prefers the centre, so it does not hug the wall on curves.
Cross-sectional frames are anchored to the patient's anterior direction *at
every station* rather than transported along the curve. That last choice is what
makes clock position mean the same thing at every level and in every patient,
which is what makes the flattened maps comparable across a cohort.

---

## 5. Decisions that are not obvious

**Why "origin" and "daughter" are not interchangeable.** The brief's unit is the
daughter instance. A vessel arising from another daughter is explicitly *not* a
direct aortic daughter, so a common trunk that forks 6 mm later is one origin
*and* one daughter instance. They map one to one. The page counts daughters
because that is the brief's word.

**Why an absolute HU floor and not a local comparison.** The first version of
the contrast gate compared the lumen to its local background and flagged subject
001, which is a perfectly good scan. The vertebral body and other opacified
vessels sit in that background and drag it up. An absolute floor has no such
failure mode.

**Why the ostium is not the footprint centroid.** Measured over 467 contact
patches: the centroid falls outside its own patch **26.8%** of the time, usually
landing inside the aortic lumen. The rim-distance maximum is inside every single
time. Median shift 1.5 mm, against a published inter-observer agreement of
2.5 mm. Directly worth points on the 25% criterion.

**Why "rim distance" and not a plain distance transform.** The contact patch is
one voxel thick. An ordinary distance transform of it is half a voxel everywhere
and its maximum lands wherever ties happen to break, which made positions
erratic. The distance has to be measured *along the wall*, to the nearest wall
voxel that is not part of the patch.

**Why the direction window scales.** A 1 mm lumbar artery and a 4 mm renal were
sharing a flat 6 mm fitting window. Now three times the local radius, measured
on a provisional fit then refitted.

---

## 6. Things tried that did not work

Keep this list. It is the most expensive knowledge in the project.

**Frangi / Sato vesselness as a detector.** 32 s per case on its own, and
vesselness collapses exactly at the ostium where the geometry stops being
tubular, so growth could not reach the wall. Danilov hits the same wall and adds
a dedicated aorta-border cleaning step for it. If vesselness is ever revisited,
use it as a *rescorer on the traced 5 to 10 mm segment*, where the structure
genuinely is a tube, not as a detector.

**Danilov's distance-layer peeling.** Walks layers inward from the farthest one,
dropping voxels with no neighbour further out. Ported directly it anchors on each
label's farthest point, which on this data is frequently a leak rather than the
vessel, so it kept the leak and deleted the branch. Cost subject 2 both its
coeliac and mesenteric arteries. Reverted. If retried, it needs an anchor that
is not "farthest".

**Rotation-minimising frames along the centerline.** Drifted and then flipped
180 degrees partway down, which silently rotated clock positions and made
anatomy look wrong. Replaced by per-station anchoring to the patient's anterior
direction.

**A footprint-ratio filter tightened to 3.6 with an "opening wider than the
parent" rule.** Killed real celiac origins, whose footprints are legitimately
large relative to the aorta. Reverted to 4.2 with no parent-ratio rule.

---

## 7. Known limitations

- **No accuracy number exists.** Intrinsic checks only.
- **Subject 18 exceeds the compute budget** at 92 s. Fixable by decimating the
  centerline on long segments if it ever binds.
- **Subjects 22 and 25 return 25 and 26 daughters.** Plausible for 221 mm and
  490 mm of aorta including intercostals and lumbars, but unverified.
- **CT slice sprites are soft.** Stored at 176 px per slice and upscaled.
  Sharpening means regenerating every sprite, a full pipeline pass.
- The iliac bifurcation is reported separately from the daughters, since the
  brief puts it outside the core task.

---

## 8. Reference papers and what they gave us

Three papers were supplied partway through and reviewed after the pipeline was
already written. Convergence with them is independent, not copied.

- **Tahoces et al., Med Biol Eng Comput 2020**, automatic detection of aortic
  anatomical landmarks in CTA. Their "contact areas" are our footprint patches,
  arrived at separately. Gave us the ostium-centre correction. Reports 91.8%
  recall, 98.8% precision on 30 test cases, and **inter-expert disagreement of
  2.5 ± 2.1 mm**, which is the noise floor for criterion 2.
- **Riffaud et al., Med Biol Eng Comput 2022**, branch detection from abdominal
  aortic segmentation by graph matching. Gave us the scale-adaptive direction
  window. 239 segmentations, 102 patients.
- **Danilov et al., Computation 2016**, cardiovascular segmentation at several
  scales. Source of the peeling idea that did not work here, and independent
  confirmation that vesselness misbehaves near the aorta.

**Do not present their recall figures as our target.** Both search a known,
fixed list of seven or eight *named* arteries with hard anatomical priors, such
as the two renals lying within 35 mm of each other. This brief forbids that and
asks for whatever is actually present, lumbar and intercostal arteries included.
Their 91.8% is measured on the easy seven.

---

## 9. Still to do

- Record the five-minute demo. The page is ordered for it: flattened map, then
  the scan check, then the cohort, then the runtime table.
- Decide whether to chase subject 18's runtime.
- Optional, if time: vesselness as a rescorer on the traced segment.

---

## 10. House style for this project

- The page is white, Helvetica, no web fonts, one red for daughters and one blue
  for branch-free stretches. Both validated for colour-blind separation.
- Prose is plain and specific. No em dashes.
- Every claim on the page is either computed from the predictions or cited.
  Nothing is asserted as accuracy that has not been measured.
