#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"

# Existing baseline checkpoints are reused so only the three extension models are trained.
for checkpoint in outputs/checkpoints/fcn_resnet18_best.pth outputs/checkpoints/fcn_resnet34_best.pth; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing baseline checkpoint: ${checkpoint}. Run scripts/run_experiments.sh first." >&2
    exit 1
  fi
done

"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet18.yaml --device "${DEVICE}"
"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet34.yaml --device "${DEVICE}"

"${PYTHON_BIN}" train.py --config configs/fcn_resnet50.yaml --device "${DEVICE}"
"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet50.yaml --device "${DEVICE}"

"${PYTHON_BIN}" train.py --config configs/fcn_resnet34_deconv.yaml --device "${DEVICE}"
"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet34_deconv.yaml --device "${DEVICE}"

"${PYTHON_BIN}" train.py --config configs/fcn_resnet34_voc21.yaml --device "${DEVICE}"
"${PYTHON_BIN}" evaluate.py --config configs/fcn_resnet34_voc21.yaml --device "${DEVICE}"

"${PYTHON_BIN}" compare_models.py \
  --configs configs/fcn_resnet18.yaml configs/fcn_resnet34.yaml configs/fcn_resnet50.yaml \
  --summary-stem backbone_comparison --classes-stem backbone_class_iou_comparison \
  --title "Backbone Comparison"
"${PYTHON_BIN}" compare_models.py \
  --configs configs/fcn_resnet34.yaml configs/fcn_resnet34_deconv.yaml \
  --summary-stem upsampling_comparison --classes-stem upsampling_class_iou_comparison \
  --title "Upsampling Comparison"
"${PYTHON_BIN}" compare_models.py \
  --configs configs/fcn_resnet34_voc21.yaml \
  --summary-stem voc21_results --classes-stem voc21_class_iou \
  --title "VOC 21-Class Results"

"${PYTHON_BIN}" predict.py \
  --configs configs/fcn_resnet18.yaml configs/fcn_resnet34.yaml configs/fcn_resnet50.yaml \
  --output-dir outputs/visualizations/backbone_comparison --device "${DEVICE}"
"${PYTHON_BIN}" predict.py \
  --configs configs/fcn_resnet34.yaml configs/fcn_resnet34_deconv.yaml \
  --output-dir outputs/visualizations/upsampling_comparison --device "${DEVICE}"
"${PYTHON_BIN}" predict.py \
  --config configs/fcn_resnet34_voc21.yaml \
  --output-dir outputs/visualizations/fcn_resnet34_voc21 --device "${DEVICE}"
