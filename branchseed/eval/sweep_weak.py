"""Sweep the weak-study brightness floor against the draft reference."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score as sc  # noqa: E402

data = sc.cached()
base = sc.score(data=data, detail=True)
print(sc.line("baseline (no weak filter: fraction=0)", base))
print("  per case (tp,fp,fn):", base["per_case"])

for frac in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
    s = sc.score({"weak_hu_fraction": frac}, data=data)
    print(sc.line(f"  weak_hu_fraction={frac}", s))

for gate in (260, 280, 300, 320, 350, 400):
    s = sc.score({"weak_lumen_hu": gate}, data=data)
    print(sc.line(f"  weak_lumen_hu={gate}", s))

print("\nchosen setting, per case:")
s = sc.score({"weak_lumen_hu": 300.0, "weak_hu_fraction": 0.70},
             data=data, detail=True)
print(sc.line("  weak 300/0.70", s))
print("  per case (tp,fp,fn):", s["per_case"])
