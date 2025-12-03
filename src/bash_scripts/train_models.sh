#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Resolve project structure
# ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
VENV_DIR="$PROJECT_DIR/venv"

# ----------------------------------------
# Activate Venv
# ----------------------------------------

source "$VENV_DIR/bin/activate"

LANGS=("en" "es" "fr" "fa" "pt" "ur" "ug" "vi" "de" "zh" "he" "ar")
MODELS=("mbert" "xlmr")
EPOCHS=5
MAX_SIZE=-1

cd "$SRC_DIR"

# ----------------------------------------
# Training loops
# ----------------------------------------
for lang in "${LANGS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Training ${lang} ${model}."
        python3.12 -m python_programs.train "$model" "$lang" "$EPOCHS" "$MAX_SIZE"
        echo "${lang} ${model} trained."
    done
done
