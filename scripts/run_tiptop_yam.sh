#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/yam_stack_common.sh
source "$SCRIPT_DIR/yam_stack_common.sh"
yam_stack_load_env

[[ -d "$YAM_TIPTOP_DIR" ]] || yam_stack_die "Missing TiPToP checkout: $YAM_TIPTOP_DIR. Run scripts/bootstrap_yam_stack.sh first."
[[ -n "${GOOGLE_API_KEY:-}" ]] || yam_stack_die "GOOGLE_API_KEY is required for TiPToP/Gemini perception."

if (($# == 0)); then
  set -- \
    --h5-path /tmp/yam_tiptop_obs.h5 \
    --task-instruction "pick up the red cube" \
    --output-dir /tmp/tiptop_yam_m2t2_robotiq_frame_fixed_spheres \
    --tool-frame-mode yam-robotiq-pinch-pad \
    --num-particles 512 \
    --max-planning-time 60 \
    --constraint-debug \
    --pose-debug
fi

cd "$YAM_TIPTOP_DIR"
exec pixi run python "$YAM_PLAYGROUND_DIR/scripts/run_tiptop_h5_yam_debug.py" "$@"
