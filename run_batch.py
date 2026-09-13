import os
import time
import json
import subprocess

# Locate the dataset folder safely
data_root_choices = [d for d in os.listdir(".") if "CHALLENGE" in d and os.path.isdir(d)]
if not data_root_choices:
    print("Error: Could not locate dataset folder!")
    exit(1)
root_dir = data_root_choices[0]

out_dir = "predictions"
log_dir = "logs"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

summary_file = os.path.join(out_dir, "run_summary.csv")
with open(summary_file, "w") as f:
    f.write("case_id,status,runtime_sec,daughters_found\n")

print("\n====================================================")
print("          STARTING FINAL RUNTIME SWEEP              ")
print("====================================================\n")

subjects = sorted([d for d in os.listdir(root_dir) if d.startswith("subject")])

for subj in subjects:
    num_str = "".join(filter(str.isdigit, subj))
    if not num_str: 
        continue
    num = int(num_str)
    
    img = os.path.join(root_dir, subj, f"orig{num}.nii")
    if not os.path.exists(img):
        img += ".gz"
        
    mask = os.path.join(root_dir, subj, f"mask{num}.nii")
    if not os.path.exists(mask):
        mask += ".gz"
        
    out = os.path.join(out_dir, f"{subj}.json")
    log_path = os.path.join(log_dir, f"{subj}.log")
    
    if not os.path.exists(img) or not os.path.exists(mask):
        print(f"[{subj}] MISSING INPUT — skipped")
        with open(summary_file, "a") as f:
            f.write(f"{subj},MISSING_INPUT,0.0,0\n")
        continue
        
    start_time = time.time()
    
    cmd = ["python3", "run.py", "--image", img, "--aorta-mask", mask, "--output", out]
    with open(log_path, "w") as log_f:
        res = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        
    end_time = time.time()
    rt = end_time - start_time
    
    if res.returncode == 0:
        try:
            with open(out, "r") as json_f:
                data = json.load(json_f)
                n = l(data.get("daughters", []))
        except Exception:
            n = "NA"
        status = "OK"
        print(f"[{subj}] OK  runtime={rt:.3f}s  daughters={n}")
    else:
        status = "FAILED"
        n = 0
        print(f"[{subj}] FAILED — see {log_path}")
        
    with open(summary_file, "a") as f:
        f.write(f"{subj},{status},{rt:.3f},{n}\n")

print("\n====================================================")
print("          FINAL COMPLETE BATCH RUN SUMMARY          ")
print("====================================================\n")

with open(summary_file, "r") as f:
    for line in f.readlines():
        print(line.strip().replace(",", "\t"))
