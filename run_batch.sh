#!/bin/bash
mkdir -p predictions logs

SUMMARY="predictions/run_summary.csv"
echo "case_id,status,runtime_sec,daughters_found" > "$SUMMARY"

counter=1
# The wildcard safely matches any directory variation on your disk
for dir in *CHALLENGE*/subject*/; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    subj=$(basename "$dir")
    i=$(echo "$subj" | tr -dc '0-9')
    if [ -z "$i" ]; then
        i=$counter
    fi
    
    img="${dir}orig${i}.nii"
    mask="${dir}mask${i}.nii"
    out="predictions/${subj}.json"
    log="logs/${subj}.log"

    if [[ ! -f "$img" || ! -f "$mask" ]]; then
        echo "$subj,MISSING_INPUT,0,0" >> "$SUMMARY"
        counter=$((counter + 1))
        continue
    fi

    start=$(python3 -c 'import time; print(time.time())')
    
    if python3 run.py --image "$img" --aorta-mask "$mask" --output "$out" > "$log" 2>&1; then
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
echo "            FINAL COMPLETE BATCH RUN SUMMARY        "
echo "===================================================="
column -s, -t "$SUMMARY"
