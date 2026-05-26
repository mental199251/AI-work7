#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${1:-./data}"

"${PYTHON_BIN}" prepare_data.py --root "${DATA_ROOT}" --download
