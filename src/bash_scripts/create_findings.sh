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

MODELS=("mbert" "xlmr")

ALL_LANGS=("en" "es" "fr" "fa" "pt" "ur" "ug" "vi" "de" "zh" "he" "ar")
SUBSET_LANGS=("en" "de" "zh" "he" "ar")

cd "$SRC_DIR"

# ----------------------------------------
# Run findings
# ----------------------------------------
for model in "${MODELS[@]}"; do
    python3.12 -m python_programs.results "$model" "${ALL_LANGS[@]}"
    python3.12 -m python_programs.results "$model" "${SUBSET_LANGS[@]}"
done


python3.12 -m python_programs.results "${ALL_LANGS[@]}"
python3.12 -m python_programs.results "${SUBSET_LANGS[@]}"
