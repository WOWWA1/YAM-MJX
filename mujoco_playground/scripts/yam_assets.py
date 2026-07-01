"""Resolve local YAM assets from MuJoCo Menagerie."""

from __future__ import annotations

from pathlib import Path
import subprocess


MENAGERIE_REPO = "https://github.com/deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT_SHA = "1b86ece576591213e2b666ebf59508454200ca97"

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DEPS_DIR = REPO_ROOT / "mujoco_playground" / "external_deps"
MENAGERIE_DIR = EXTERNAL_DEPS_DIR / "mujoco_menagerie"
YAM_DIR = MENAGERIE_DIR / "i2rt_yam"


def ensure_menagerie_assets() -> None:
    """Download MuJoCo Menagerie assets if this checkout does not have them."""
    if MENAGERIE_DIR.exists():
        return

    EXTERNAL_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", MENAGERIE_REPO, str(MENAGERIE_DIR)],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(MENAGERIE_DIR), "checkout", MENAGERIE_COMMIT_SHA],
        check=True,
    )


def require_yam_file(filename: str) -> Path:
    """Return a YAM asset path, downloading Menagerie first when needed."""
    ensure_menagerie_assets()
    path = YAM_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find YAM asset {filename!r} at {path}. "
            f"Expected MuJoCo Menagerie asset folder: {YAM_DIR}"
        )
    return path
