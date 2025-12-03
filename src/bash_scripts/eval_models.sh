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
TRAINED_LANGS=("en" "es" "fr" "fa" "pt" "ur" "ug" "vi" "de" "zh" "he" "ar")
EVAL_LANGS=("en" "es" "fr" "fa" "pt" "ur" "ug" "vi" "de" "zh" "he" "ar")

cd "$SRC_DIR"

# ----------------------------------------
# Eval loops
# ----------------------------------------
for trained_lang in "${TRAINED_LANGS[@]}"; do
    for eval_lang in "${EVAL_LANGS[@]}"; do
        for model in "${MODELS[@]}"; do
            echo "Evaluating $model trained using $trained_lang on $eval_lang"
            python3.12 -m python_programs.eval "$model" "$trained_lang" "$eval_lang"
        done
    done
done
