# Toralis Labs Track — Fundamentals & Problem Brief

This doc is just background/context — no solution ideas yet. Use it to think from first principles.

> 📘 **For non-bio folks**: throughout this doc, plain-English callout boxes like this one translate the medical/anatomical jargon so you don't need a biology background to follow along.

---

## 1. What Toralis Labs Actually Does

- **Company pitch**: "AI surgical digital twins for vascular intervention" — they build software that helps clinicians study a patient's specific blood vessel anatomy, check how well a medical device (like a stent or graft) will fit, evaluate "seal quality," and estimate procedural risk *before* doing the actual surgery.
- In short: take a patient's medical scan → build a computer model of their blood vessels → simulate/predict what will happen with different devices or surgical choices → give the surgeon a risk estimate ahead of time.
- Website: https://toralislabs.com
- On GitHub they also go by **Radiel Health** and have repos on: AV fistula/graft flow modeling (`AVFlow`, `Bifurcation`, `coronary-bifurcation-batch`), a general CFD fluid-dynamics benchmark (`LidDrivenCavity`), and a `neuro` repo.

---

## 2. The Real Challenge Brief: "Branchseed Challenge — Discover Aortic Branch Origins"

> ⚠️ This is the actual, detailed task spec from Toralis (`Branchseed challenge.pdf`) — it's far more specific than the one-line devpost blurb ("visualize and organize a healthcare dataset"). Treat this PDF as the source of truth.

### The task in one sentence
> Given a CT volume and a binary mask containing only the parent abdominal aorta, detect every eligible artery that directly leaves the aorta and return each one as a separate daughter instance.

### Medical background
- The abdominal aorta gives rise to several arteries (e.g., celiac trunk, superior mesenteric artery (SMA), renal arteries, inferior mesenteric artery, common iliac arteries).
- Their number, position, and direction **vary between patients**, and a given scan may only show part of the aorta.
- This anatomical unpredictability complicates surgical repair — hence the need for automated detection of *where branches leave the aorta* for each individual patient.
- Imaging context: CT angiography (CTA) — a contrast agent is injected so blood appears bright on CT. An axial slice only shows a cross-section of each vessel; the system must effectively identify bright vascular structures and follow them across consecutive slices to figure out branch origins in 3D.
- **Important framing**: this is not "find the SMA" or other named arteries — it's a general detection task. The system must discover whatever branches are actually present in each case, without assuming a fixed anatomical list.

> 📘 **Plain-English version**: The **aorta** is the main highway of a blood vessel running down the middle of the body. Smaller roads (**arteries**) branch off it to feed organs like the kidneys and intestines. Every person's "road map" is a little different — different number of exits, different spots, different angles. A **CT scan** is like a stack of X-ray slices through the body; doctors inject a dye (**contrast agent**) so blood shows up bright/white in the images, making vessels easy to spot. Your job isn't to name each "exit" (artery) — just to find every place a smaller road splits off the highway.

### What you're given
Roughly 25 paired NIfTI volumes, structured as:
```
data/
  subject001/
    orig1.nii      # the CT volume
    mask1.nii      # binary mask: 1 = parent aorta lumen, 0 = everything else
  subject002/
    orig2.nii
    mask2.nii
  ...
```
- Image and mask share the same grid / physical-coordinate system.
- **Daughter arteries are visible in the CT but NOT included in the mask** — the mask only marks the parent aorta. The mask is just a "search anchor"; it is not sufficient on its own.
- Aortic coverage varies per case (some scans show a longer/shorter segment).
- A small dev subset includes example reference outputs; the final hidden test set does not.

### Definitions (important — precise language matters for scoring)
| Term | Meaning |
|---|---|
| Parent aorta | The lumen given by the supplied binary mask |
| Direct daughter | An artery whose lumen connects directly to the parent aorta |
| Ostium centre | The centre of the opening where a direct daughter leaves the parent aorta |
| Daughter seed | A point at the centre of the daughter lumen, 5 mm outward from the ostium along the daughter's path |
| Daughter radius | Estimated local lumen radius at the daughter seed, in mm |
| Daughter instance | One independently detected branch, given one unique ID |

> 📘 **Plain-English version**: "**Lumen**" just means the hollow inside of a tube (the actual open channel blood flows through — like the inside of a pipe, not the pipe wall). "**Ostium**" is just a fancy word for "the doorway/opening" — literally where a branch pipe connects to the main pipe. A "**seed**" point is just a reference point a little way down the branch (5mm in) used to measure it, since right at the doorway/junction things are messy and hard to measure precisely. "**Radius**" is how thick that branch pipe is (half its diameter). Think of the whole thing like: highway (aorta) → exit ramp doorway (ostium) → a point a bit down the ramp (seed) → how wide the ramp is (radius) → which way the ramp points (direction).

### Exactly what the system must do, per unseen case
1. Examine the CT around the supplied aorta mask.
2. Detect every eligible artery arising **directly** from the aorta.
3. Assign each detected artery a unique daughter-instance ID (`branch_001`, `branch_002`, … — **no anatomical names allowed**).
4. Link every daughter instance back to the parent aorta.
5. For each daughter, report: the ostium centre, the daughter seed point, direction, and local radius.

### Eligibility rule for a branch to count
- Its contrast-filled lumen must be traceable for **at least 5 mm beyond the aortic wall**.
- Its origin must meet a minimum size threshold (specified with the final dataset).
- Trace the proximal branch **up to 10 mm beyond the ostium, or until the first downstream bifurcation** (whichever comes first) — this proximal segment is what's used to estimate the seed, radius, and direction.

> 📘 **Plain-English version**: "**Proximal**" means "close to the starting point" (as opposed to "distal," which means further away/downstream). "**Bifurcation**" just means a fork — the point where one vessel splits into two. So this rule just says: only look at the first short stretch of the branch (up to 10mm) right after it leaves the aorta, and stop early if it forks again before that.

### Required output format
One JSON file per case:
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
Rules:
- Coordinates must be **physical millimetres** (via `SimpleITK.TransformIndexToPhysicalPoint`) — **not** voxel indices.
- `direction_xyz` = unit vector pointing from ostium into the daughter vessel.
- Each real daughter appears exactly once; each prediction needs a unique `instance_id` and `parent_instance_id = "aorta"`.
- If no eligible daughter exists in a case, return an empty `daughters` list.

### Required visual check
- For at least 3 cases: a simple image/plot showing the aorta mask, detected ostia, and daughter-direction arrows.
- This is only for sanity-checking the output — **not** meant to be a polished clinical UI.

### Edge cases the system must handle correctly
- A shorter/longer supplied aortic segment → different numbers of daughters is expected and fine.
- The flat top/bottom faces created by cropping the aorta are **not** branch origins — don't false-positive on them.
- Two separate origins close together at the aortic wall = two distinct instances, not one.
- A common trunk that splits shortly after leaving the aorta = **one** direct aortic origin (only the first branch point off the aorta counts).
- A vessel that branches off of another daughter (not directly off the aorta) does **not** count as a direct daughter.
- Never guess/hallucinate a vessel that isn't actually visible or is outside the supplied coverage.
- The aorta's terminal split into the iliac arteries is **explicitly out of the core task** (may be a separate optional extension).

> 📘 **Plain-English version**: Imagine the aorta as a section of pipe that's been cut out of a longer pipe for study — the cut ends (top/bottom) are just flat because someone cut them, not because a branch is there, so don't mistake a cut edge for a branch. If two branches genuinely come off the pipe at slightly different, distinct spots, count them separately, even if they're close together. But if it's really just one opening that immediately splits into two right after leaving the pipe, that's still just one "exit," so count it once. Also, don't count exits that belong to a branch, not the main pipe (only exits directly off the *aorta* count) — and don't guess an exit exists if you can't actually see it in the scan. Finally, the "**iliac arteries**" are just the two big vessels the aorta itself splits into at its very end (like the trunk of a tree splitting into two big branches) — that specific split is out of scope for this task.

### Minimum working prototype requirements
The submission must:
- Accept a brand-new CT + aorta mask with **no manual point placement** (fully automatic).
- Output valid JSON matching the schema above.
- Detect a *variable* number of daughters (not hardcoded).
- Preserve the physical coordinate system.
- Run on the full evaluation set with no case-specific tweaking/hacks.
- **Run on a standard laptop with no GPU.**
- Must be runnable via: `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json`

Any method is allowed: classical image processing, vessel-enhancement filters (e.g. Frangi/Hessian-based), graph/search algorithms, lightweight ML, or a hybrid — no specific technique is mandated.

### Evaluation criteria (out of 100%)
| Category | Weight | What's assessed |
|---|---|---|
| Branch discovery | 45% | Precision/recall/F1 for detecting the correct number of daughter instances |
| Ostium localisation | 25% | Physical distance between predicted and reference ostium centres |
| Daughter-instance quality | 15% | Whether the seed lies on the matched daughter, direction follows the proximal path, radius is consistent with the lumen |
| Compute efficiency | 10% | Runtime + peak memory on organiser's CPU-only system |
| Reproducibility | 5% | Valid output, documented setup, runs successfully on unseen cases |

- Predictions are matched to references **one-to-one** — duplicate detections of the same real branch count as false positives.
- Test environment: 4 CPU cores, 8 GB RAM, **no GPU, no internet access**.
- Target runtime: ~60 sec/case average (initial target; may be finalized after organiser baseline testing).

> 📘 **Plain-English version**: "**Precision**" = of all the branches you claimed to find, how many were actually real (penalizes false alarms). "**Recall**" = of all the real branches that existed, how many did you actually find (penalizes missed branches). "**F1**" is just a single combined score that balances precision and recall together, so you can't win by only optimizing one of them. The other categories just check: are your points in the right physical spot, does your branch measurement actually make sense, does your code run fast/light enough, and does it actually work reliably when someone else runs it.

### Submission checklist
- Source code
- Dependency or environment file
- Short README with one setup command + one run command
- JSON predictions for the dev set
- Required visual checks (≥3 cases)
- 5-minute demo video/explanation covering method, runtime, and known failure cases

### Explicitly out of scope
- Segmenting the parent aorta (it's given to you).
- Assigning anatomical names to branches.
- Reconstructing the full distal vascular tree (only direct daughters near the aorta matter).
- Predicting branches that aren't actually visible in the scan.

> 📘 **Plain-English version**: You're not being asked to (1) trace out the main highway yourself — it's already drawn for you; (2) label each exit with its real name (e.g., "this is the kidney exit") — just number them; (3) map every tiny side street beyond the first exit ramp — just the exits directly off the highway; or (4) guess at exits that aren't actually visible in the picture you're given.

### One-line summary of the real ask
> Find every eligible artery that directly leaves the supplied aorta, keep the daughters separate, and represent each origin accurately enough to be used as a machine-readable branch instance.

---

## 3. The Underlying Medical Problem (Background Only)

- Vascular surgeons planning interventions (e.g., aortic aneurysm repair, stent placement) need to know exactly where each side-branch artery originates so devices don't accidentally block blood flow to organs (kidneys, intestines, etc.) fed by those branches.
- Because branch anatomy varies significantly patient-to-patient, and can be partially obscured or only partially captured in a scan, this is currently a manual, expertise-dependent, and time-consuming step in surgical planning.
- Automating "where do the branches come off the aorta, and how big/oriented are they" is a foundational building block toward the broader "digital twin" surgical planning pipeline Toralis is building.

> 📘 **Plain-English version**: When a surgeon repairs a damaged section of the aorta (e.g., a bulging weak spot called an **aneurysm**) by inserting a device like a **stent** (a tube-shaped support structure), they need to make sure that device doesn't accidentally cover up/block one of the exits feeding blood to a kidney or intestine — that could cause serious harm. Right now, figuring out exactly where those exits are is manual and requires an expert eyeballing the scan. Automating this — precisely and quickly — is a small but foundational piece of Toralis's bigger goal: a full computer "twin" of a patient's vessels that helps plan surgery in advance.

---

## 4. Open Questions to Think About Before Deciding on a Solution

- What's the core technical strategy: classical vessel-enhancement (e.g. Frangi filter + connected components) vs. a lightweight learned model vs. a hybrid?
- How do we reliably distinguish a "direct aortic daughter" from a branch-of-a-branch, using only local geometry/intensity near the aorta wall?
- How do we estimate ostium centre, seed point, direction, and radius robustly from a short (≤10mm) proximal segment, especially in noisy/lower-resolution CT regions?
- How do we avoid false positives on the cropped flat ends of the aorta, and avoid merging/splitting close-together ostia incorrectly?
- Given the strict runtime/hardware constraints (CPU-only, ~60s/case, 8GB RAM), how much can lean on heavier ML vs. needs to stay classical/fast?
- What's the simplest reliable way to validate against the dev-set reference outputs before the hidden test set evaluation?
