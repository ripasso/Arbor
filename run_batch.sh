#!/bin/bash
mkdir -p predictions logs

SUMMARY="predictions/run_summary.csv"
echo "case_id,status,runtime_sec,daughters_found" > "$SUMMARY"

counter=1
# Uses wildcard matching to catch trailing spaces in the folder name safely
for dir in TORALIS*CHALLENGE*/subject*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    subj=$(basename "$dir")
    i=$(echo "$subj" | tr -dc '0-9')
    if [ -z "$i" ]; then
        i=$counter
    else
        i=$((10#$i))
    fi
    
    img="${dir}orig${i}.nii"
    mask="${dir}mask${i}.nii"
    
    # Fallback to finding any matching nii file if naming varies
    if [ ! -f "$img" ]; then
        img=$(find "$dir" -maxdepth 1 -name "orig*.nii" | head -n 1)
    fi
    if [ ! -f "$mask" ]; then
        mask=$(find "$dir" -maxdepth 1 -name "mask*.nii" | head -n 1)
    fi

    out="predictions/${subj}.json"
    log="logs/${subj}.log"

    if [[ ! -f "$img" || ! -f "$mask" ]]; then
        echo "$subj,MISSING_INPUT,0,0" >> "$SUMMARY"
        echo "[$subj] MISSING INPUT — skipped"
        counter=$((counter + 1))
        continue
    fi

    start=$(python3 -c 'import time; print(time.time())')
    
    if python run.py --image "$img" --aorta-mask "$mask" --output "$out" > "$log" 2>&1; then
        end=$(python3 -c 'import time; print(time.time())')
        rt=$(python3 -c "print(f'{$end - $start:.3f}')")
        n=$(python3 -c "import json; data=json.load(open('$out')); print(len(next((v for v in data.values() if isinstance(v, list)), [])) if isinstance(data, dict) else len(data))" 2>/dev/null || echo "0")
        echo "$subj,OK,$rt,$n" >> "$SUMMARY"
        echo "[$subj] OK  runtime=${rt}s  daughters=$n"
    else
        end=$(python3 -c 'import time; print(time.time())')
        rt=$(python3 -c "print(f'{$end - $start:.3f}')")
        echo "$subj,FAILED,$rt,0" >> "$SUMMARY"
        echo "[$subj] FAILED — see $log"
    fi
    counter=$((counter + 1))
done

echo ""
echo "===================================================="
echo "           AL COMPLETE BATCH RUN SUMMARY        "
echo "===================================================="
column -s, -t "$SUMMARY"
