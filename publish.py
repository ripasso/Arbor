"""Assemble site/ from out/.

The page is generated entirely from the pipeline's own outputs, so this exists
to make that literal rather than a claim: it copies the figures, wraps the atlas
as a script the page can load without a server, and reports anything the page
asks for that the pipeline did not produce.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out")
ASSETS = os.path.join(OUT, "assets")
SITE = os.path.join(ROOT, "site")

# prefix -> extension, one entry per family of figures the page loads by name
FAMILIES = {
    "ct_": ".jpg",          # patient-aligned axial slices
    "mask_": ".png",        # supplied aorta, overlaid on the above
    "unwrap_": ".png",      # flattened wall map
    "cor_": ".jpg",         # coronal slab projections
    "cormask_": ".png",
    "sag_": ".jpg",         # sagittal slab projections
    "sagmask_": ".png",
    "evid_": ".jpg",        # per-branch evidence contact sheet
}


def main():
    with open(os.path.join(OUT, "atlas.json")) as fh:
        atlas = json.load(fh)
    cases = [c["case_id"] for c in atlas["cases"]]

    os.makedirs(SITE, exist_ok=True)
    copied, missing = 0, []
    wanted = set()
    for case in cases:
        for prefix, ext in FAMILIES.items():
            name = f"{prefix}{case}{ext}"
            wanted.add(name)
            src = os.path.join(ASSETS, name)
            if not os.path.exists(src):
                missing.append(name)
                continue
            dst = os.path.join(SITE, name)
            if (not os.path.exists(dst)
                    or os.path.getmtime(src) > os.path.getmtime(dst)):
                shutil.copy2(src, dst)
                copied += 1

    with open(os.path.join(SITE, "data.js"), "w") as fh:
        fh.write("window.ATLAS=")
        json.dump(atlas, fh, separators=(",", ":"))
        fh.write(";\n")

    stale = [f for f in os.listdir(SITE)
             if any(f.startswith(p) for p in FAMILIES) and f not in wanted]
    for f in stale:
        os.remove(os.path.join(SITE, f))

    print(f"{len(cases)} cases, {copied} figures copied, "
          f"{len(stale)} stale removed")
    if missing:
        print(f"MISSING {len(missing)} figures the page will ask for:")
        for name in missing[:12]:
            print("  ", name)
        if len(missing) > 12:
            print(f"   ... and {len(missing) - 12} more")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
