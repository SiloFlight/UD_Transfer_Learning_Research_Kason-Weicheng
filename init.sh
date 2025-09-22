#source this file before running python files to resolve import issues.

source "$(dirname "${BASH_SOURCE[0]}")/projectEnv/bin/activate"

#Update PYTHONPATH to be project root
export PYTHONPATH=$(dirname "${BASH_SOURCE[0]}/src")