#Languages
TRAINING_CODES=(en_ewt fr_gsd es_gsd pt_gsd ur_udtb ug_udt vi_vtb fa_perdt)

for CODE in "${TRAINING_CODES[@]}"; do
    for CODE2 in "${TRAINING_CODES[@]}"; do
        echo "=== Evaluating $CODE on $CODE2"
        python3.12 src/eval.py "$CODE" "$CODE2"
        echo "=== Finished evaluating $CODE on $CODE2"
    done
done