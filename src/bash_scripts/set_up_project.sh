#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------
# Resolve project structure
# ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC_DIR="$PROJECT_DIR/src"
VENV_DIR="$PROJECT_DIR/venv"
DATASET_DIR="$PROJECT_DIR/datasets"
REQ_FILE="$PROJECT_DIR/requirements.txt"
LANG2VEC_DIR="$PROJECT_DIR/src2/python_programs/lang2vec"

echo "-- Project directory: $PROJECT_DIR"

# ----------------------------------------
# Create venv
# ----------------------------------------
echo "-- Creating virtual environment"
python3.12 -m venv "$VENV_DIR"

echo "-- Activating venv"

source "$VENV_DIR/bin/activate"

echo "-- Upgrading pip"
pip install -q --upgrade pip

# ----------------------------------------
# Install pip requirements
# ----------------------------------------
if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: $REQ_FILE not found."
    exit 1
fi

echo "-- Installing Python dependencies"
pip install -q -r "$REQ_FILE"
echo "-- Requirements installed"

# ----------------------------------------
# Install lang2vec manually
# ----------------------------------------
echo "-- Installing lang2vec"

if [ -d "$LANG2VEC_DIR" ]; then
    rm -rf "$LANG2VEC_DIR"
fi

git clone -q https://github.com/antonisa/lang2vec "$LANG2VEC_DIR"

# install quietly
(cd "$LANG2VEC_DIR" && python3.12 setup.py install >/dev/null 2>&1)

echo "-- Lang2vec installed"

# ----------------------------------------
# Download UD datasets
# ----------------------------------------
echo "-- Preparing dataset directory"
rm -rf "$DATASET_DIR"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

DATASETS=(
    "https://github.com/UniversalDependencies/UD_English-EWT"
    "https://github.com/UniversalDependencies/UD_Spanish-GSD"
    "https://github.com/UniversalDependencies/UD_French-GSD"
    "https://github.com/UniversalDependencies/UD_Persian-PerDT"
    "https://github.com/UniversalDependencies/UD_Portuguese-GSD"
    "https://github.com/UniversalDependencies/UD_Urdu-UDTB"
    "https://github.com/UniversalDependencies/UD_Uyghur-UDT"
    "https://github.com/UniversalDependencies/UD_Vietnamese-VTB"
    "https://github.com/UniversalDependencies/UD_German-GSD"
    "https://github.com/UniversalDependencies/UD_Chinese-GSD"
    "https://github.com/UniversalDependencies/UD_Hebrew-HTB"
    "https://github.com/UniversalDependencies/UD_Arabic-PADT"
)

echo "-- Downloading UD datasets"

for url in "${DATASETS[@]}"; do
    repo_name=$(basename "$url")
    echo "   > $repo_name"
    git clone -q "$url"
done

echo "-- All datasets installed"

echo "-- Project setup complete."
