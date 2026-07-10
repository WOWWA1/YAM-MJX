#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/yam_stack_common.sh
source "$SCRIPT_DIR/yam_stack_common.sh"
yam_stack_load_env

[[ -x "$YAM_MUJOCO_VENV/bin/python" ]] || yam_stack_die "Missing MuJoCo venv: $YAM_MUJOCO_VENV. Run scripts/bootstrap_yam_stack.sh first."

if (($# == 0)); then
  set -- \
    --output /tmp/yam_tiptop_obs.h5 \
    --preview-png /tmp/yam_tiptop_obs.png \
    --fovy 24 \
    --camera-pos 0.65,-0.30,0.42 \
    --camera-target 0.45,0.0,0.025 \
    --cube-pos 0.45,0.0,0.025
fi

cd "$YAM_PLAYGROUND_DIR"
# shellcheck disable=SC1091
source "$YAM_MUJOCO_VENV/bin/activate"
exec python scripts/save_tiptop_h5_from_yam.py "$@"
