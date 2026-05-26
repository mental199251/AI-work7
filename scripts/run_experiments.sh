#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"

"${PYTHON_BIN}" collect_environment.py
"${PYTHON_BIN}" train.py --config configs/fcn_resnet18.yaml --device "${DEVICE}"
"${PYTHON_BIN}" train.py --config configs/fcn_resnet34.yaml --device "${DEVICE}"
"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet18.yaml --device "${DEVICE}"
"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet34.yaml --device "${DEVICE}"
"${PYTHON_BIN}" compare_models.py
"${PYTHON_BIN}" predict.py --compare --output-dir outputs/visualizations/comparison --device "${DEVICE}"
