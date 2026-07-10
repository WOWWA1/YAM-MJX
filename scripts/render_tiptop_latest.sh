#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/yam_stack_common.sh
source "$SCRIPT_DIR/yam_stack_common.sh"
yam_stack_load_env

RUN_OR_PARENT="${1:-/tmp/tiptop_yam_m2t2_robotiq_frame_fixed_spheres}"
VIDEO_PATH="${2:-/tmp/yam_tiptop_replay.mp4}"

if [[ -f "$RUN_OR_PARENT/tiptop_plan.json" ]]; then
  LATEST="$RUN_OR_PARENT"
else
  LATEST="$(find "$RUN_OR_PARENT" -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort | tail -1)"
fi

[[ -n "${LATEST:-}" && -f "$LATEST/tiptop_plan.json" ]] || yam_stack_die "Could not find tiptop_plan.json under $RUN_OR_PARENT"
[[ -x "$YAM_MUJOCO_VENV/bin/python" ]] || yam_stack_die "Missing MuJoCo venv: $YAM_MUJOCO_VENV. Run scripts/bootstrap_yam_stack.sh first."

printf 'Rendering %s -> %s\n' "$LATEST/tiptop_plan.json" "$VIDEO_PATH"

cd "$YAM_PLAYGROUND_DIR"
# shellcheck disable=SC1091
source "$YAM_MUJOCO_VENV/bin/activate"
exec python scripts/replay_tiptop_plan_yam.py \
  --plan "$LATEST/tiptop_plan.json" \
  --video "$VIDEO_PATH" \
  --playback-mode teleport \
  --speed 2.0 \
  --marker-sites grasp_site,tcp_site \
  --marker-fingertip-geoms lf_down,rf_down \
  --marker-finger-midpoint-bodies lf_down,rf_down \
  --marker-tool-frame-mode yam-robotiq-pinch-pad \
  --marker-radius 0.012 \
  --camera-pos 0.75,-0.55,0.5 \
  --camera-target 0.43,-0.02,0.06 \
  --fovy 32
