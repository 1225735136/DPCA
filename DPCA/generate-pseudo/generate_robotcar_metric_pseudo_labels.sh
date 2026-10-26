#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${ROBOTCAR_DATA_ROOT:?Set ROBOTCAR_DATA_ROOT to the RobotCar md4all root}"
: "${ROBOTCAR_PSEUDO_OUTPUT:?Set ROBOTCAR_PSEUDO_OUTPUT to the pseudo-label output directory}"
: "${DAV2_METRIC_CHECKPOINT:?Set DAV2_METRIC_CHECKPOINT to a DAV2 metric checkpoint}"

python "${script_dir}/generate_metric_pseudo_labels.py" \
  --dataset robotcar \
  --data-root "${ROBOTCAR_DATA_ROOT}" \
  --output-root "${ROBOTCAR_PSEUDO_OUTPUT}" \
  --checkpoint "${DAV2_METRIC_CHECKPOINT}" \
  --preview-count 20 \
  "$@"
