#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/yam_stack_common.sh
source "$SCRIPT_DIR/yam_stack_common.sh"
yam_stack_load_env

[[ -d "$YAM_M2T2_DIR" ]] || yam_stack_die "Missing M2T2 checkout: $YAM_M2T2_DIR. Run scripts/bootstrap_yam_stack.sh first."

cd "$YAM_M2T2_DIR"
exec pixi run server
