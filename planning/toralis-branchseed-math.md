# Branchseed Challenge — The Math Behind the Solution

This doc explains the key mathematical ideas our pipeline relies on, why each one is the right tool for the job, and the intuition behind it. No code — just the math, explained so you don't need a background in medical imaging to follow.

---

## 1. The image as a 3D function

A CT scan is mathematically just a function:

```
I(x, y, z) → intensity value (in Hounsfield Units)
```

Every voxel position maps to a brightness number. All the math below is about analyzing the *shape* of this function — where it's bright, how brightness changes from point to point, and what geometric structures those changes imply.

Two coordinate systems matter:
- **Index space**: integer voxel positions `(i, j, k)` — how the data is stored.
- **Physical space**: real-world millimetre positions `(x, y, z)` — what the challenge requires.

The conversion is an **affine transformation**:

```
physical_point = origin + direction_matrix × (index × spacing)
```

where `spacing` is the physical size of a voxel along each axis, `origin` is where voxel (0,0,0) sits in the real world, and `direction_matrix` is a rotation describing how the grid axes are oriented. SimpleITK's `TransformIndexToPhysicalPoint` applies exactly this formula. **Every distance rule in the spec (5mm, 10mm) must be computed in physical space** because spacing varies per subject.

---

## 2. Vesselness: the Hessian matrix and its eigenvalues

**The problem**: distinguish "bright tube" (a vessel) from "bright blob" (an organ) or "bright sheet" (a bone surface), using only local intensity values.

**The tool**: second derivatives — how the brightness *curves* around a point.

### The Hessian matrix

At each voxel, compute the 3×3 matrix of second partial derivatives of intensity:

```
        | ∂²I/∂x²   ∂²I/∂x∂y  ∂²I/∂x∂z |
H(p) =  | ∂²I/∂y∂x  ∂²I/∂y²   ∂²I/∂y∂z |
        | ∂²I/∂z∂x  ∂²I/∂z∂y  ∂²I/∂z²  |
```

Intuition: a first derivative says "brightness is increasing/decreasing in this direction." A second derivative says "brightness forms a peak/valley/ridge in this direction." The Hessian collects this curvature information for all directions at once.

### Eigenvalues reveal shape

Diagonalizing H gives three eigenvalues λ₁, λ₂, λ₃ (sorted |λ₁| ≤ |λ₂| ≤ |λ₃|), each paired with an eigenvector (a direction). They describe brightness curvature along three perpendicular axes:

| Structure | Eigenvalue signature (bright structure on dark background) |
|---|---|
| **Tube** (vessel!) | λ₁ ≈ 0, λ₂ ≈ λ₃ ≪ 0 — flat along the tube's axis, sharply curved across it |
| **Blob** (organ) | λ₁ ≈ λ₂ ≈ λ₃ ≪ 0 — curved in every direction |
| **Sheet/plate** (bone edge) | λ₁ ≈ λ₂ ≈ 0, λ₃ ≪ 0 — flat in two directions |
| **Noise/nothing** | all λ ≈ 0 |

A tube is bright along one axis (walk along the vessel: brightness stays constant → λ₁ ≈ 0) and falls off steeply in the two perpendicular directions (step off the vessel sideways: brightness drops → λ₂, λ₃ strongly negative).

**Bonus**: the eigenvector paired with λ₁ points **along the vessel axis** — which directly gives us a local direction estimate for a branch, for free.

### The Frangi vesselness score

Frangi's filter combines the eigenvalues into a single 0-to-1 "tube-likeness" score using three ratios:

```
R_A = |λ₂| / |λ₃|          → distinguishes tubes from sheets (≈1 for tubes, ≈0 for sheets)
R_B = |λ₁| / √(|λ₂·λ₃|)    → distinguishes tubes from blobs (≈0 for tubes, ≈1 for blobs)
S   = √(λ₁² + λ₂² + λ₃²)   → overall strength; suppresses weak/noisy responses
```

```
V = (1 − exp(−R_A²/2α²)) · exp(−R_B²/2β²) · (1 − exp(−S²/2c²))
```

with tunable sensitivities α, β, c. High V = "very tube-like and strong" → likely vessel.

### Multi-scale detection

Vessels come in different thicknesses. The Hessian is computed on a Gaussian-smoothed image at several smoothing scales σ (e.g., 1mm, 2mm, 3mm), and each voxel takes the **maximum** vesselness over all scales. A scale roughly "tuned" to a vessel's radius responds strongest to it — so small σ catches thin branches, large σ catches thick ones.

---

## 3. Morphological dilation: building the search shell

**Dilation** of a binary set A by a structuring element B (e.g., a small sphere of radius r):

```
A ⊕ B = { p : (B centered at p) ∩ A ≠ ∅ }
```

In words: every point within distance r of the original shape becomes part of the dilated shape. Dilating the aorta mask by a few mm and subtracting the original mask gives a hollow **shell** hugging the aortic wall — exactly where ostia must live. This is set-theoretic geometry, cheap to compute, and shrinks our search space from the whole volume to a thin band.

The sphere's radius must be specified in **voxels converted from mm using each subject's spacing** (see §1).

---

## 4. Connected components: grouping voxels into branch candidates

Define a graph where each candidate vessel voxel is a node, and edges connect **adjacent** voxels (26-connectivity in 3D: touching by face, edge, or corner). A **connected component** is a maximal set of nodes mutually reachable through edges — i.e., one contiguous blob.

Standard algorithms (union-find, or flood-fill/BFS) label all components in roughly linear time O(n) in the number of voxels.

This is what lets us say "these voxels form *one* candidate branch, those form *another*" — and it's also how the spec's tricky rules become computable:
- **Two nearby but separate ostia** = two distinct connected components at the wall → keep separate. Proximity doesn't matter; *connectivity* does.
- **Common trunk that forks later** = one connected component touching the wall in one contact zone → one instance, regardless of what it does downstream.
- **Branch-of-a-branch** = component that touches a daughter but not the aorta mask itself → excluded by the adjacency test.

---

## 5. The distance transform: measuring vessel thickness

The **Euclidean distance transform (EDT)** of a binary vessel mask assigns each interior voxel its distance to the nearest background voxel:

```
D(p) = min { ‖p − q‖ : q ∉ vessel }
```

Key property: **at the centerline of a tube, D(p) ≈ the tube's local radius.** The center of a pipe is, by definition, the point farthest from its walls — and that distance is the radius.

So `radius_mm = D(seed_point)` (computed with anisotropic spacing so the answer is in true mm). One transform, computed in O(n) with standard algorithms, gives us radius estimates everywhere at once.

The EDT also helps find the centerline itself: the centerline is the **ridge** of the distance function (the locus of local maxima of D across the vessel cross-section).

---

## 6. Skeletonization & centerline geometry

**Skeletonization (thinning)** reduces a 3D blob to a 1-voxel-wide curve that preserves its topology — the mathematical "spine" of the shape. Formally it approximates the **medial axis**: the set of points with more than one nearest boundary point.

From the skeleton of a candidate branch we get an ordered path of points `p₀ (at the wall), p₁, p₂, …` walking outward. This 1D curve is what makes the spec's measurements well-defined:

- **Arc length**: distance along the path `s(k) = Σᵢ ‖pᵢ₊₁ − pᵢ‖` (in physical mm). The **seed** is the point where s = 5mm; **eligibility** requires the path to reach s ≥ 5mm; tracing stops at s = 10mm or at a skeleton **branch point** (a skeleton voxel with ≥3 skeleton neighbours = a bifurcation).
- **Direction (tangent)**: the local direction of the curve at the ostium. Estimated either as the normalized difference `(p_seed − p_ostium)/‖p_seed − p_ostium‖`, or more robustly by fitting a line to the first few path points via **least squares / PCA** (the first principal component of the point cloud is the dominant direction). Must be reported as a **unit vector** (length 1 — direction only, no magnitude).

---

## 7. Principal Component Analysis (PCA) — the direction-finding workhorse

Given a small cloud of 3D points (e.g., the proximal centerline points), form the 3×3 **covariance matrix**:

```
C = (1/n) Σᵢ (pᵢ − p̄)(pᵢ − p̄)ᵀ
```

Its eigenvector with the largest eigenvalue is the axis along which the points spread the most — for a stubby vessel segment, that's the **vessel's direction**. This is the same eigen-decomposition math as the Hessian analysis in §2, applied to point positions instead of image curvature. One tool, two uses.

Sign disambiguation: PCA gives an axis, not an arrow — flip the vector if needed so it points **away** from the aorta (dot product with "ostium → component centroid" should be positive).

---

## 8. Detecting the flat cropped ends (false-positive suppression)

The cropped top/bottom of the aorta form **planar** disc-shaped regions, while real vessel wall is **curved**. Two cheap mathematical tests:

1. **Slice-position test**: contact points lying in the first or last occupied z-slices of the mask are suspect.
2. **Planarity test (PCA again)**: take the local patch of boundary voxels around a contact zone and run PCA. A flat disc has variance concentrated in 2 components (λ₃ ≈ 0, thin pancake); a curved wall or protruding tube spreads variance into the third. Reject contacts whose local geometry is planar and axis-aligned with the volume's z-extremes.

---

## 9. The evaluation math (what the rubric computes)

Predictions are matched **one-to-one** to reference branches — this is an instance of **bipartite matching**: greedily (or optimally, via the Hungarian algorithm) pair each predicted ostium with the nearest unmatched reference ostium within a tolerance distance. Then:

```
Precision = TP / (TP + FP)     "of my detections, how many were real?"
Recall    = TP / (TP + FN)     "of the real branches, how many did I find?"
F1        = 2·P·R / (P + R)    harmonic mean — punishes imbalance
```

The harmonic mean matters: predicting 50 branches to catch every real one gives high recall but terrible precision, and F1 collapses. The math structurally forces a *conservative but thorough* detector — which is why the eligibility check (§6) is as important as the detection itself.

Ostium localisation (25% of score) is plain **Euclidean distance in mm** between matched ostium pairs — small errors accumulate directly, so sub-voxel care in computing the contact centroid pays off.

---

## 10. Why this math beats ML here (one paragraph)

Every step above is a **closed-form, deterministic computation** with a handful of interpretable parameters (smoothing scales, thresholds, dilation radius). It requires zero training data, runs in seconds on a CPU, behaves identically on every machine, and its failure modes are debuggable ("the threshold was too high") rather than opaque. With only ~25 cases and a hidden test set, a learned model would risk overfitting exactly where determinism is being graded (reproducibility, generalization to unseen cases). The classical math is not a fallback — for this problem size and rubric, it is the optimal tool.

---

## Cheat sheet — one line per concept

| Math | Job in the pipeline |
|---|---|
| Affine index→physical transform | Report answers in true mm; honor per-subject spacing |
| Hessian eigenvalues (Frangi) | Score every voxel for "tube-likeness" → find vessels |
| Gaussian multi-scale analysis | Catch both thin and thick branches |
| Morphological dilation | Build a thin search shell around the aortic wall |
| Connected components (graph) | Separate/merge candidates exactly per the spec's rules |
| Euclidean distance transform | Vessel radius at any point; centerline ridge |
| Skeletonization / medial axis | 1D centerline → arc length, seed at 5mm, bifurcation stops |
| PCA (covariance eigenvectors) | Branch direction; planarity test for cropped ends |
| Bipartite matching + P/R/F1 | How the rubric scores us; replicate locally on the dev set |

---

## 11. Critical evaluation — is this the right approach?

### Verdict up front

Yes — the classical approach is the right call for *this* problem under *these* constraints. But with two honest caveats: (1) the Frangi filter is probably **not the star of the show** — connectivity to the aorta is, and (2) the real difficulty won't be finding branches, it'll be **suppressing false positives**.

### Pros

- **Zero training data needed** — with ~25 cases and only a few having reference outputs, any trained model would be starved. Classical math works on case #1.
- **Fits the rubric's hard constraints perfectly** — CPU-only, 8GB RAM, ~60s/case, no internet, must run reproducibly on unseen cases. Deterministic algorithms ace the 10% efficiency + 5% reproducibility categories almost by default.
- **Debuggable** — when it fails on a dev case, you can see *why* (threshold too high, dilation too small) and fix it in minutes. A misbehaving neural net during a 24-hour hackathon is a time sink you can't afford.
- **The spec's rules are geometric, not statistical** — "5mm traceable," "one component at the wall = one instance," "flat cropped ends don't count." These translate *directly* into connectivity and arc-length math. An ML model would have to *learn* rules you can just *write*.
- **The strongest available signal is free**: daughter lumens are contrast-filled and **physically continuous with the aorta lumen** — same blood, same dye, similar brightness. Connectivity from the given mask is a nearly perfect anchor.

### Cons / real risks

- **Threshold fragility across scans**: contrast timing varies patient-to-patient, so "bright" isn't a fixed HU number. A hardcoded threshold that works on subject 1 may fail on subject 20. *Mitigation (important)*: calibrate per-case using the intensity statistics **inside the given aorta mask itself** — that's a free, per-patient reference for "what does contrast-filled blood look like in this scan."
- **False positives are everywhere near the aorta**:
  - **Calcified plaque** in the aortic wall (very common in vascular patients) is bright and sits right at the wall — looks like a branch stub.
  - **Bone**: the aorta runs directly against the spine; vertebrae are bright and inside the dilated shell.
  - **The IVC** (a large vein) runs parallel and can be contrast-bright depending on scan timing.

  This is where Frangi/tubularity, size limits, and the 5mm-traceability rule earn their keep — as *filters*, not as detectors.
- **Small branches near resolution limits**: at 1.5mm voxel spacing, a 2mm-radius branch is ~2-3 voxels wide, and the 5mm eligibility check is ~3 voxels of path. Skeletonization and radius estimates get noisy at that scale. Expect the worst dev-set errors on coarse-resolution subjects.
- **Merge errors**: two branches that arise close together (e.g., celiac trunk and SMA) can blur into one connected component via partial-volume voxels — the exact opposite of the spec's "keep separate origins separate" rule. This edge case needs deliberate handling, not luck.
- **Frangi has knobs** (α, β, c, scale range) and tuning them well against only a handful of reference cases is guesswork-adjacent. Budget time for this or reduce dependence on it.
- **No learning means no free improvement** — if the hidden test set has a systematically different character (different scanner, different contrast protocol), fixed rules can't adapt. This is the genuine advantage ML would have with enough data — which we don't have.

### One strategic adjustment

Invert the emphasis in the plan: make **region-growing from the aorta mask** (grow outward through contrast-bright voxels, and every place the growth escapes the mask surface = candidate ostium) the *primary* detector, and use vesselness/tubularity as a *secondary filter* to kill calcification/bone/vein false positives. Connectivity is a stronger, simpler, more robust signal than tube-shape scoring — and it's directly aligned with the spec's own definition of a daughter ("lumen connects directly to the parent aorta").

### Bottom line

Classical is correct for this rubric, this data volume, and this timeline. The plan lives or dies on **per-case intensity calibration** and **false-positive suppression** — not on the elegance of the vesselness math. Prioritize accordingly.

---

## 12. How we satisfy the "Ostium localisation" (25%) and "Daughter-instance quality" (15%) rubric categories

### Ostium localisation (25%)

**What it means:** The judges have the *true* location of each branch doorway (ostium) in mm. They measure the straight-line distance between your predicted point and the true point. Closer = more points. It's literally "how many mm off were you?"

**How we satisfy it:**
- Find where the candidate branch's voxels touch the aorta wall (the contact zone).
- Take the **centroid** (average position) of that contact zone — not just any touching voxel — so the point lands in the *centre* of the doorway.
- Convert to physical mm with `TransformIndexToPhysicalPoint` using each subject's own spacing.

> 📘 **Plain-English version**: the answer key says "the exit ramp starts at kilometre marker 12.4" — you get graded on how close your guess is to 12.4. Averaging the whole contact patch instead of picking one edge voxel is what gets you sub-voxel accuracy.

### Daughter-instance quality (15%)

Three sanity checks on the *measurements* you report for each branch:

1. **"Seed lies on the matched daughter"** — your seed point (5mm down the branch) must actually sit *inside* that branch's lumen, not floating in fat or inside a different vessel.
   → We satisfy this by walking exactly 5mm along the branch's **centerline** (skeleton, §6), so by construction the seed is in the middle of the tube.

2. **"Direction follows its proximal path"** — your direction arrow must point the way the branch actually goes in its first stretch.
   → We satisfy this by fitting a line (PCA, §7) to the first few centerline points, or simply normalizing (seed − ostium). Either follows the real path.

3. **"Radius consistent with the lumen"** — your reported thickness must match how thick the vessel really is there.
   → We satisfy this by reading the **distance transform** value (§5) at the seed: distance from the tube's centre to its wall *is* the radius, by definition.

**One-liner:** 25% = "is your doorway point in the right spot?" 15% = "are your three measurements (seed, arrow, thickness) physically sensible for that branch?" The centerline-based approach satisfies all three almost automatically, because every measurement is derived from the actual middle of the actual vessel.
