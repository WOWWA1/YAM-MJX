#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/yam_stack_common.sh
source "$SCRIPT_DIR/yam_stack_common.sh"
yam_stack_load_env

yam_stack_log "Path summary"
printf 'YAM_ROOT=%s\n' "$YAM_ROOT"
printf 'YAM_LOCAL=%s\n' "$YAM_LOCAL"
printf 'YAM_TIPTOP_DIR=%s\n' "$YAM_TIPTOP_DIR"
printf 'YAM_M2T2_DIR=%s\n' "$YAM_M2T2_DIR"
printf 'YAM_MUJOCO_VENV=%s\n' "$YAM_MUJOCO_VENV"

yam_stack_log "Tool checks"
command -v pixi
pixi --version
command -v uv || true

yam_stack_log "TiPToP check"
(
  cd "$YAM_TIPTOP_DIR"
  pixi run tiptop-run -h >/dev/null
  echo "tiptop ok"
)

yam_stack_log "M2T2 check"
(
  cd "$YAM_M2T2_DIR"
  pixi run python - <<'PY'
from pointnet2_ops import _ext
import m2t2
print("m2t2 ok")
PY
)

yam_stack_log "MuJoCo check"
"$YAM_MUJOCO_VENV/bin/python" - <<'PY'
import mujoco
print("mujoco ok", mujoco.__version__)
PY
