# Branchseed Challenge — Solution Plan (No Code Yet)

This is a plan for *how* we'd approach solving the Branchseed challenge, based on the spec + direct inspection of the real dataset. No implementation yet — just the strategy, pipeline steps, and things to watch out for.

> 📘 **For non-bio/non-imaging folks**: plain-English callout boxes like this one are sprinkled throughout to translate the medical-imaging jargon (masks, voxels, vesselness filters, etc.) into everyday terms.

---

## 0. What we confirmed by inspecting the real data

- 25 subjects total, each with `origN.nii` (CT) + `maskN.nii` (binary aorta mask), matching the spec exactly.
- **Image size varies per subject** (e.g., 512×512×174 vs 303×290×185 vs 299×299×201) — nothing can assume a fixed volume shape.
- **Voxel spacing varies per subject** (e.g., 0.78mm vs 0.9mm vs 1.5mm isotropic) — this matters a lot, since the spec's "5mm" and "10mm" distances must be computed in physical mm, not voxel counts. Every distance-based step must use the actual spacing from each file, not a hardcoded value.
- **Intensity range** is standard CT Hounsfield Units (roughly -2048 to ~1700), consistent with contrast-enhanced CTA as described in the brief.
- Masks are clean binary volumes (values are only 0 and 1).
- **Data quirk found**: subjects 001–015 are plain NIfTI files, but subjects 016–025 are actually gzip-compressed data saved with a `.nii` extension instead of `.nii.gz`. A naive loader that trusts the file extension will crash on these — the loader needs to detect gzip magic bytes regardless of extension and handle both cases transparently. This is exactly the kind of "hidden gotcha" the hidden test set could also contain, so it should be handled defensively from the start.

---

## 1. Overall strategy

Given the constraints — CPU-only, no GPU, no internet, ~60s/case target, must generalize to unseen cases with zero manual intervention — the right approach is a **classical image-processing pipeline**, not a trained deep model. Reasons:
- No labeled training data for *this exact task* is provided (only ~25 cases total, and only some have example outputs) — not enough to train a reliable detector from scratch.
- **Classical vessel-enhancement** + **geometric reasoning** is well-understood for this exact problem type (it's essentially a vessel-branch-detection task, a classic vascular image analysis problem), and it's fast/deterministic/CPU-friendly.
- The rules (eligibility distance, proximal tracing distance, no invented anatomy) are explicit, deterministic thresholds — better suited to explicit geometric logic than to a learned model with limited data.

So: **classical pipeline first**, and only consider lightweight ML as a refinement if time allows (e.g., a small classifier to filter false-positive candidates), not as the core detector.

---

## 2. Pipeline stages

### Stage A — Robust loading
- Load `origN` and `maskN` via SimpleITK, auto-detecting gzip-disguised files regardless of extension.
- Verify image and mask share the same size/spacing/origin/direction (sanity check per spec: "same grid and physical-coordinate system").
- Keep everything in physical space in mind from the start — always convert voxel index → mm via `TransformIndexToPhysicalPoint`, never report raw indices.

### Stage B — Define the search region
- We don't need to scan the entire CT volume — only the region right around the aorta matters (branches originate *at* the aortic wall).
- Take the given aorta mask and **dilate it by a small physical margin** (e.g., a handful of mm, converted to voxels using each subject's actual spacing) to get a thin "search shell" around the aorta surface.
- This shell, plus a bit of extra margin outward, becomes our region of interest (ROI) — this keeps later steps fast (small volume instead of full CT) and focused only on plausible branch locations, which also directly helps hit the 60s/case runtime budget.

> 📘 **Plain-English version**: A **voxel** is just a 3D pixel — a tiny cube of the scan. "**Dilating**" a shape just means puffing it up a bit in every direction (like inflating a balloon slightly) — so we take the given aorta shape and grow it outward a little to create a thin "shell" zone right around its surface, since that's the only place branches could possibly start. Searching only in this thin shell (instead of the whole 3D scan) is much faster — like only checking the edges of a highway for exits instead of scanning the entire map.

### Stage C — Vessel enhancement in the ROI
- Apply a tubular-structure/vesselness enhancement filter (Hessian-eigenvalue based, e.g. a Frangi-style filter, available in standard libraries) to the CT intensities inside the ROI.
- This highlights bright, tube-shaped structures (i.e., contrast-filled vessels) and suppresses blobby/planar structures (organs, bone, noise) — separating "this looks like a vessel" from "this is just bright tissue."
- Combine this vesselness response with a simple intensity threshold (since contrast-filled vessels are reliably bright/high-HU) to get candidate vessel voxels.

> 📘 **Plain-English version**: A "**vesselness filter**" is a piece of math that looks at a 3D image and asks, at every point, "does this local region look like a thin tube, or does it look like a blob/flat sheet?" Blood vessels are tube-shaped, so this filter lights up on vessels and stays dark on rounder things like organs or flat things like bone surfaces — it's basically a "tube detector." "**HU**" (Hounsfield Units) is just the brightness scale CT scans use; contrast-filled blood shows up reliably bright on this scale, so we can also just filter by "is this bright enough to be contrast-filled blood."

### Stage D — Candidate branch detection
- Within the ROI, take all candidate vessel voxels that are **not** part of the aorta mask itself.
- Group them into connected components (a standard connected-component labeling step).
- For each connected component, check whether it actually touches the aorta's outer surface (i.e., is adjacent to the aorta mask boundary). Components that touch = candidate daughter branches. Components that float disconnected from the aorta = discard (not a direct daughter).

> 📘 **Plain-English version**: A "**connected component**" is just a blob of touching voxels that form one continuous shape — like using the paint-bucket/magic-wand tool in an image editor to select one connected splotch of color. So this step groups all the "looks like a vessel" voxels into separate blobs, then checks which blobs are actually physically touching the aorta (real candidate branches) versus floating nearby unconnected (probably a different structure, like a rib or another organ's vessel, so we ignore it).

### Stage E — Ostium identification & de-duplication
- For each surviving candidate component, find the contact point(s) where it meets the aorta wall — this is the candidate **ostium**.
- Handle two tricky rules from the spec carefully here:
  - **Two close but genuinely separate origins** must stay as two instances — so don't over-merge just because two ostia are physically near each other; check whether they're actually two distinct connected vessel bodies at the wall, not one.
  - **A common trunk that splits shortly after leaving the aorta** should be treated as **one** ostium — so if a single connected component touches the aorta at one contact zone and only splits into two after leaving the wall, that's one branch instance, not two.
- Discard contact points located on the **flat cropped top/bottom faces** of the aorta volume (these are artifacts of how the scan was cropped, not real anatomy) — detectable because those faces are flat/planar cross-sections at the very top or bottom slice of the mask, not curved vessel-wall geometry.

### Stage F — Eligibility check & proximal tracing
- For each candidate ostium, trace the vessel outward from the aortic wall along the candidate component:
  - Confirm the contrast-filled lumen is continuously traceable for **at least 5mm** beyond the wall (using physical mm, via the subject's actual spacing) — if it fades out or breaks earlier, discard as ineligible (likely noise or a partial/non-visible vessel).
  - Continue tracing up to **10mm beyond the ostium, or until the first downstream bifurcation** (whichever comes first) — this proximal segment is what all our later measurements are based on.
- A reasonable way to trace: work along the connected component using its centerline (e.g., via a distance-transform + skeleton/thinning approach, or by stepping along the local intensity centroid slice by slice) rather than the raw voxel mask, since we need a 1D path through the vessel to measure direction and locate the seed point.

> 📘 **Plain-English version**: A "**centerline**" is just the imaginary line running down the exact middle of a tube-shaped vessel (like the yellow dashed line down the center of a road) — we need this instead of the whole blobby 3D shape because it's much easier to measure direction/position along a simple line than along a fat 3D blob. A "**skeleton/thinning**" algorithm is a standard technique that shrinks a 3D blob down to this thin centerline automatically, similar to how you might trace the core of a worm-shaped object.

### Stage G — Compute required outputs per accepted branch
- **Ostium centre**: the identified contact point on the aortic wall, converted to physical mm.
- **Daughter seed**: the point on the traced centerline exactly 5mm outward from the ostium along the path.
- **Local radius**: estimate the vessel's local radius at the seed point — a distance-transform value at that point (distance to nearest non-vessel voxel) converted to mm using the spacing gives a reasonable radius estimate.
- **Direction**: the local tangent direction of the centerline at the ostium (or the normalized vector from ostium to seed, as a simpler approximation), expressed as a unit vector.
- Assign a unique `branch_00X` ID to each accepted daughter, in a consistent order (e.g., sorted by position), and set `parent_instance_id = "aorta"`.

> 📘 **Plain-English version**: A "**distance transform**" is a simple calculation that, for every point inside the vessel, tells you "how far away is the nearest edge/wall." At the very center of a tube, that distance roughly equals the tube's radius — so this gives us an easy way to estimate how thick the vessel is at any point, without needing to manually measure it. A "**tangent direction**" is just "which way is this line pointing at this exact spot," and a "**unit vector**" is just a direction expressed as an arrow of length exactly 1 (so it only describes direction, not distance).

### Stage H — Output assembly
- Write one JSON file per case matching the required schema exactly (case_id, parent block, daughters array with all required fields).
- If no eligible daughters are found for a case, output an empty `daughters` list — never fabricate a branch.

### Stage I — Required visual checks
- For at least 3 cases, produce a simple figure (e.g., an axial slice or 3D scatter) showing: the aorta mask, the detected ostium points, and small arrows indicating the daughter direction vectors. This is explicitly meant to be a sanity check, not a polished UI — simplicity is fine and expected here.

---

## 3. Validation plan

- Use the small development subset (the cases that come with example reference outputs) to check our pipeline's outputs against ground truth before trusting it on the hidden test set.
- Basic checks to run per dev case: does our count of detected branches roughly match the reference count (branch discovery), how far off are our ostium points from the reference ones (ostium localisation), and are our seed/direction/radius values in a sane range relative to the reference.
- Since the official scoring formula (precision/recall/F1 for discovery, distance for localisation, etc.) is known from the brief, we can approximate the same scoring locally on the dev set to self-evaluate before submission.

---

## 4. Performance considerations (to hit the ~60s/case, CPU-only, 8GB RAM target)

- Never run expensive filters (like vesselness) on the *entire* CT volume — always restrict to the small dilated ROI around the aorta.
- Prefer vectorized array operations (NumPy/SciPy) over per-voxel Python loops.
- Keep the pipeline dependency-light (standard scientific Python stack: SimpleITK/nibabel, NumPy, SciPy, scikit-image) so it installs cleanly in an offline, no-internet evaluation environment.

> 📘 **Plain-English version**: "**Vectorized operations**" just means doing a calculation on an entire array of numbers all at once (fast, built-in), instead of writing a manual loop that processes one number at a time in plain Python (very slow for large 3D images). This is purely a performance/coding-style detail, not a medical concept — the takeaway is just "use the fast, built-in math tools, not slow manual loops," so the program finishes within the time limit.

---

## 5. Suggested build order (minimum viable → refined)

1. **V0 (skeleton)**: Loader + CLI wrapper that reads image/mask, outputs a valid (even if empty) JSON in the correct schema — gets the required interface working first.
2. **V1 (naive detector)**: Dilate mask → find bright connected components touching the aorta surface → treat each component's contact centroid as ostium, its principal axis as direction, and its cross-sectional size as radius. No proper centerline tracing yet — this is the fastest path to a working, if rough, end-to-end submission.
3. **V2 (refined)**: Add proper vesselness enhancement (Stage C), a real centerline-based tracing (Stage F), and the merge/split edge-case handling (Stage E) to improve precision/recall and localisation accuracy.
4. **V3 (polish)**: Tune thresholds against the dev-set reference outputs, add the required visualizations, write the README + runtime/failure-case notes for the 5-minute demo.

---

## 6. Key pitfalls to explicitly guard against (from the spec's "important cases")

| Pitfall | Guard |
|---|---|
| False positive on cropped flat aorta ends | Detect and exclude flat/planar contact regions at the very top/bottom of the mask |
| Merging two real, separate nearby ostia into one | Check for genuinely distinct connected components at the wall, not just proximity |
| Splitting one common-trunk origin into two | Only count the first contact point at the aortic wall as one instance, even if it forks shortly after |
| Counting a branch-of-a-branch as a direct daughter | Only accept components that touch the aorta mask directly, not components that only touch another daughter |
| Hallucinating a branch that isn't really there or is out of the scan's coverage | Require the 5mm continuous-traceability eligibility check before accepting any candidate |
| Reporting voxel indices instead of physical mm | Always convert via `TransformIndexToPhysicalPoint` before writing output |
| Crashing on the gzip-disguised `.nii` files (confirmed present in subjects 016–025) | Loader must sniff file contents, not trust the `.nii` extension |

---

## 7. Note on the earlier "Hemodynamic Risk Atlas" idea

That earlier idea was based on a *different* Toralis dataset (`public-data-trials`, AV-graft CFD hemodynamics with TAWSS/OSI fields). This Branchseed challenge uses **completely different data** (CT + aorta masks) and a **completely different task** (branch detection, not flow visualization). The two are unrelated — this plan supersedes that direction for this specific challenge.
