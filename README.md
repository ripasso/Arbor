# Arbor: branchseed challenge

Given a CT volume and a binary mask of the parent abdominal aorta only, this
finds every artery that leaves that aorta and reports each one as a separate
daughter instance: the centre of its opening, a seed 5 mm along it, its initial
direction and its local lumen radius, in physical millimetres.

Classical image processing throughout. No training data, no GPU, no network
access, no per-case parameters.

## Results for given dataset
https://ripasso.github.io/Arbor/

# Demo
https://www.youtube.com/watch?v=onATyIupc-U

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

## Inspiration
A missed branch artery during aortic stent grafting can cut off blood to a kidney. Surgeons find these branches by manually scrolling CT scans and estimating by eye. Arbor gives a complete, measured inventory of every branch in seconds, so the surgical team walks in with every vessel accounted for.

## What it does
Given a CT angiogram and a binary mask of the abdominal aorta, Arbor finds every artery that leaves that aorta, reporting each as a separate instance with the centre of its opening in physical millimetres, a seed point 5 mm along the vessel, its initial direction, and its lumen radius. It runs in about ten seconds per case on a single laptop core. Furthermore, every detection is drawn on a flattened map of the aortic wall, so a surgeon can see and audit exactly how each decision was made.

## How we built it
Arbor is built entirely with classical image processing in Python (NumPy, SciPy, scikit-image), not machine learning in the pipeline. The aortic wall is cut open and flattened into a 2D map, depth down the aorta on one axis and clock position around the wall on the other, which turns branch detection into reading marks on a map.

Since aortic brightness runs from ~85 to 580 HU across patients, a threshold ladder walks down from the supplied lumen until the most separate, plausible openings resolve, and each bright patch touching the wall becomes one branch ostium. Labels then grow outward in strict distance order, keeping branches separate and leaks contained; patches that split into two vessels a few millimetres out are divided and traced back to the wall; and each branch is followed 10 mm to its first bifurcation to measure its seed, direction, and radius. Physiological filters remove implausible candidates, with every refusal carrying a logged reason.

## Challenges we ran into
With images that were low resolution that with more noise (i.e. having low contrast), our algorithm had a difficult time in identifying the daughter branches, and often overestimated. We solved this by changing the filter for low contrast images, to add nuance and improve accuracy.
