#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-alfworld}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
DATA_DIR="${ALFWORLD_DATA:-/root/.cache/alfworld}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"

if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
elif [ -x /root/miniconda3/bin/conda ]; then
    CONDA_EXE="/root/miniconda3/bin/conda"
else
    echo "conda not found" >&2
    exit 1
fi

CONDA_BASE="$($CONDA_EXE info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if "$CONDA_EXE" run -n "$ENV_NAME" python --version >/dev/null 2>&1; then
    echo "Conda env '$ENV_NAME' already exists."
else
    "$CONDA_EXE" create -y --override-channels -c defaults -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"
python -m pip install --index-url "$PIP_INDEX_URL" --timeout 60 -r requirements-task2.txt

export ALFWORLD_DATA="$DATA_DIR"
echo "ALFWORLD_DATA=$ALFWORLD_DATA"

if [ "${DOWNLOAD_DATA:-0}" = "1" ]; then
    if [ -f "$ALFWORLD_DATA/logic/alfred.pddl" ] && find "$ALFWORLD_DATA/json_2.1.1" -name game.tw-pddl -print -quit 2>/dev/null | grep -q .; then
        echo "ALFWorld data already exists; skip download."
    else
        alfworld-download --data-dir "$ALFWORLD_DATA"
    fi
fi

echo "Task 2 environment is ready."
echo "Run: export ALFWORLD_DATA=$ALFWORLD_DATA"
echo "Run: python env_check.py --config configs/base_config.yaml --split eval_out_of_distribution"
