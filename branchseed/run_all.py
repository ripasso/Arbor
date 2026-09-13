import glob, json, os, sys, time, traceback
import numpy as np
import resource
from pipeline import process_case, to_prediction
from export import export_case, write_atlas

DATA = "/Users/rastinabbaspour/Downloads/TORALIS CHALLENGE "
OUT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
ASSETS = os.path.join(OUT, "assets")
PRED = os.path.join(OUT, "predictions")
os.makedirs(PRED, exist_ok=True); os.makedirs(ASSETS, exist_ok=True)

subs = sorted(os.listdir(DATA))
if len(sys.argv) > 1: subs = sys.argv[1:]
entries = []
for s in subs:
    d = os.path.join(DATA, s)
    try:
        o = glob.glob(d+"/orig*.nii")[0]; m = glob.glob(d+"/mask*.nii")[0]
        t0=time.time()
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
        r = process_case(o, m, s)
        r["peak_rss_gb"] = round(max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6, rss0), 2)
        with open(os.path.join(PRED, f"{s}.json"), "w") as fh:
            json.dump(to_prediction(r), fh, indent=2)
        e = export_case(r, ASSETS)
        e["seconds"] = round(time.time()-t0, 2)
        entries.append(e)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6
        print(f"{s}: {e['seconds']:5.1f}s  daughters={len(e['daughters'])} extras={len(e['extras'])} "
              f"len={e['aorta_length_mm']:.0f}mm zones={len(e['landing_zones'])} peakRSS={peak:.1f}GB")
    except Exception:
        print(f"{s}: FAILED"); traceback.print_exc()
write_atlas(entries, OUT)
print("cases:", len(entries), "total branches:", sum(len(e['daughters']) for e in entries))
