#!/usr/bin/env bash
set -euo pipefail

cd Mask2Former

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <config_file> <run_name> [extra detectron2 args...]"
  exit 1
fi

CONFIG="$1"
RUN_NAME="$2"
shift 2  # remove config + run_name

# Derive experiment folder from config name
CFG_NAME=$(basename "${CONFIG}" .yaml)
BASE_OUT="/home/jovyan/mask2former-weights/outputs/ade20k/${CFG_NAME}"

# Generate unique run id
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME_SAFE=$(echo "${RUN_NAME}" | tr ' /' '__')
RUN_ID="${TIMESTAMP}_${RUN_NAME_SAFE}"

OUT_DIR="${BASE_OUT}/${RUN_ID}"
mkdir -p "${OUT_DIR}"

echo "=================================="
echo "CWD        : $(pwd)"
echo "Config     : ${CONFIG}"
echo "Run name   : ${RUN_NAME_SAFE}"
echo "Run ID     : ${RUN_ID}"
echo "Output dir : ${OUT_DIR}"
echo "=================================="

python train_net.py \
  --config-file "${CONFIG}" \
  --num-gpus 2 \
  DATALOADER.NUM_WORKERS 0 \
  OUTPUT_DIR "${OUT_DIR}" \
  "$@"
