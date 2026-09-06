#!/usr/bin/env bash
set -euo pipefail

# One process hosts the model; evaluation scripts call it over HTTP.
# This mirrors the real Agent setup: environment loop and inference service are decoupled.
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

DEFAULT_VLLM_BIN="/root/autodl-tmp/envs/qwen-vllm-clean/bin/vllm"
if [[ -x "${DEFAULT_VLLM_BIN}" ]]; then
  VLLM_BIN="${DEFAULT_VLLM_BIN}"
elif command -v vllm >/dev/null 2>&1; then
  VLLM_BIN="$(command -v vllm)"
else
  echo "vLLM is not installed or not found." >&2
  echo "Run: bash scripts/setup_task1_env.sh" >&2
  exit 1
fi

echo "Starting vLLM OpenAI-compatible server"
echo "  model path: ${MODEL_PATH}"
echo "  served name: ${SERVED_MODEL_NAME}"
echo "  endpoint: http://${HOST}:${PORT}/v1"
echo "  tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "  max model len: ${MAX_MODEL_LEN}"
echo "  gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"

exec "${VLLM_BIN}" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"

