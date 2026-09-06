#!/usr/bin/env bash
set -euo pipefail

# Keep the vLLM serving stack separate from the later ALFWorld environment.
# vLLM and TextWorld/ALFWorld have different dependency pressure points.
ENV_PREFIX="${ENV_PREFIX:-/root/autodl-tmp/envs/qwen-vllm-clean}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_CHANNEL="${CONDA_CHANNEL:-defaults}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
PIP_TIMEOUT="${PIP_TIMEOUT:-60}"

echo "Creating/updating task1 environment at: ${ENV_PREFIX}"
if [[ ! -d "${ENV_PREFIX}" ]]; then
  conda create -y     --override-channels     -c "${CONDA_CHANNEL}"     -p "${ENV_PREFIX}"     "python=${PYTHON_VERSION}"
fi

ENV_PYTHON="${ENV_PREFIX}/bin/python"
"${ENV_PYTHON}" -m pip install --upgrade pip --index-url "${PIP_INDEX_URL}" --timeout "${PIP_TIMEOUT}"
"${ENV_PYTHON}" -m pip install -r "${PROJECT_DIR}/requirements-task1.txt" --index-url "${PIP_INDEX_URL}" --timeout "${PIP_TIMEOUT}"

echo
echo "Task1 environment is ready."
echo "Activate it with:"
echo "  conda activate ${ENV_PREFIX}"
