source projectEnv/bin/activate

#Configs
EPOCHS=3
MAX_TRAINING_SIZE=-1

#Languages
TRAINING_CODES=(en_ewt fr_gsd es_gsd pt_gsd ur_udtb ug_udt vi_vtb fa_perdt)

for CODE in "${TRAINING_CODES[@]}"; do
    echo "=== Training on $CODE ==="
    python3 src/train.py "$CODE" "$EPOCHS" "$MAX_TRAINING_SIZE"
    echo "=== Finished $CODE ==="
done