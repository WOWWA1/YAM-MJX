#!/usr/bin/env bash
# Shared helpers for the Linux-only YAM + TiPToP setup scripts.

set -euo pipefail

yam_stack_repo_root() {
  local script_dir
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/.." && pwd
}

yam_stack_default_local() {
  if [[ "${EUID:-$(id -u)}" == "0" || -w /opt ]]; then
    printf '%s\n' "/opt/yam-local"
  else
    printf '%s\n' "$HOME/.cache/yam-mjx"
  fi
}

yam_stack_load_env() {
  local repo_root default_local env_file
  repo_root="$(yam_stack_repo_root)"
  default_local="$(yam_stack_default_local)"

  export YAM_ROOT="${YAM_ROOT:-$repo_root}"
  export YAM_LOCAL="${YAM_LOCAL:-$default_local}"

  if [[ -n "${YAM_ENV_FILE:-}" && -f "$YAM_ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$YAM_ENV_FILE"
  else
    for env_file in "$YAM_LOCAL/yam_env.sh" "/opt/yam-local/yam_env.sh" "$HOME/.cache/yam-mjx/yam_env.sh"; do
      if [[ -f "$env_file" ]]; then
        # shellcheck disable=SC1090
        source "$env_file"
        break
      fi
    done
  fi

  # Keep wrappers tied to the checkout they live in, even if an old env file
  # from another pod/machine is still around.
  export YAM_ROOT="$repo_root"
  export YAM_LOCAL="${YAM_LOCAL:-$default_local}"
  export YAM_EXTERNAL_DIR="$YAM_ROOT/external"
  export YAM_TIPTOP_DIR="$YAM_EXTERNAL_DIR/tiptop"
  export YAM_M2T2_DIR="$YAM_EXTERNAL_DIR/M2T2"
  export YAM_PLAYGROUND_DIR="$YAM_ROOT/mujoco_playground"
  export YAM_MUJOCO_VENV="${YAM_MUJOCO_VENV:-$YAM_LOCAL/venvs/mujoco_playground}"

  export PIXI_HOME="${PIXI_HOME:-$YAM_LOCAL/pixi-home}"
  export PIXI_CACHE_DIR="${PIXI_CACHE_DIR:-$YAM_LOCAL/pixi-cache}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-$YAM_LOCAL/uv-cache}"
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

  export TIPTOP_PACKAGE_DIR="$YAM_TIPTOP_DIR"
  export YAM_TIPTOP_ASSETS_DIR="$YAM_PLAYGROUND_DIR/tiptop_yam_assets"

  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
  if [[ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
  fi
  export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
  export MAX_JOBS="${MAX_JOBS:-2}"
  export PATH="$PIXI_HOME/bin:$HOME/.local/bin:$HOME/.pixi/bin:$PATH"
}

yam_stack_die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

yam_stack_log() {
  printf '\n==> %s\n' "$*"
}

yam_stack_warn() {
  printf 'WARN: %s\n' "$*" >&2
}
