#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  run_train_apptainer.sh IMAGE DATA_DIR [TRAIN_OPTIONS...]

Example:
  run_train_apptainer.sh /opt/tensorflow.sif /data/lichess \
    --epochs 10 --batch-size 1024 --checkpoint-dir outputs/checkpoints

Environment:
  APPTAINER_BIN                 Apptainer executable (default: apptainer)
  APPTAINER_PYTHON              Python executable inside image (default: python)
  TF_AUTOTUNE_THRESHOLD         TensorFlow autotune threshold (default: 1)
  TRAIN_CONTAINER_ROOT          Project mount path (default: /workspace/train)
EOF
}

if [[ $# -lt 2 ]]; then
  usage
  exit 2
fi

readonly image="$1"
readonly data_host_input="$2"
shift 2

if [[ ! -f "$image" ]]; then
  echo "Image not found: $image" >&2
  exit 1
fi
if [[ ! -d "$data_host_input" && ! -f "$data_host_input" ]]; then
  echo "Data path not found: $data_host_input" >&2
  exit 1
fi

readonly project_host="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly data_host="$(realpath "$data_host_input")"
readonly data_parent="$(dirname "$data_host")"
readonly data_name="$(basename "$data_host")"
readonly container_root="${TRAIN_CONTAINER_ROOT:-/workspace/train}"
readonly data_container="/mnt/chess-data/$data_name"
readonly apptainer_bin="${APPTAINER_BIN:-apptainer}"
readonly python_bin="${APPTAINER_PYTHON:-python}"
readonly autotune_threshold="${TF_AUTOTUNE_THRESHOLD:-1}"

bind_args=(
  --bind "$project_host:$container_root"
  --bind "$data_parent:/mnt/chess-data:ro"
)

exec "$apptainer_bin" exec --nv \
  "${bind_args[@]}" \
  --pwd "$container_root" \
  --env "PYTHONPATH=$container_root/src" \
  --env "TF_AUTOTUNE_THRESHOLD=$autotune_threshold" \
  "$image" "$python_bin" -m train.main \
  "$data_container" "$@"
