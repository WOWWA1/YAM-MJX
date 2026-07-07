#!/usr/bin/env python3
"""Install YAM support into a local TiPToP checkout.

The TiPToP repository currently hardcodes its supported robots in a few source
files. This installer applies a small, repeatable bridge patch to the checkout
under ``external/tiptop`` and installs ``cutamp.robots.yam``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PLAYGROUND_ROOT.parent
DEFAULT_TIPTOP_ROOT = REPO_ROOT / "external" / "tiptop"
DEFAULT_ASSETS_DIR = PLAYGROUND_ROOT / "tiptop_yam_assets"
BRIDGE_SOURCE = Path(__file__).resolve().parent / "tiptop_yam_bridge" / "yam.py"

YAM_IMPORT = """from .yam import (
    load_yam_rerun,
    yam_home,
    yam_curobo_cfg,
    get_yam_gripper_spheres,
    get_yam_tool_from_ee,
    get_yam_ik_solver,
    get_yam_kinematics_model,
)"""

YAM_CONTAINER = """
def load_yam_container(tensor_args: TensorDeviceType) -> RobotContainer:
    kin_model = get_yam_kinematics_model()
    joint_limits = kin_model.kinematics_config.joint_limits.position
    assert joint_limits.shape == (2, 6), f"Invalid joint limits shape: {joint_limits.shape}"

    gripper_spheres = get_yam_gripper_spheres(tensor_args)
    tool_from_ee = get_yam_tool_from_ee(tensor_args)
    return RobotContainer("yam", kin_model, joint_limits, gripper_spheres, tool_from_ee)
"""

YAM_ROBOT_ENTRY = """    "yam": {
        "rerun": load_yam_rerun,
        "q_home": yam_home,
        "container": load_yam_container,
    },"""


def _read(path: Path) -> str:
    return path.read_text()


def _write_if_changed(path: Path, text: str) -> None:
    old = path.read_text() if path.exists() else None
    if old != text:
        path.write_text(text)
        print(f"patched {path}")


def _replace(text: str, old: str, new: str, path: Path) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find expected patch anchor in {path}: {old!r}")
    return text.replace(old, new, 1)


def _insert_after(text: str, anchor: str, block: str, path: Path) -> str:
    if block in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Could not find expected patch anchor in {path}: {anchor!r}")
    return text.replace(anchor, f"{anchor}\n{block}", 1)


def _insert_before(text: str, anchor: str, block: str, path: Path) -> str:
    if block in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Could not find expected patch anchor in {path}: {anchor!r}")
    return text.replace(anchor, f"{block}\n\n{anchor}", 1)


def _ensure_assets(args: argparse.Namespace) -> None:
    if args.skip_generate_assets and not (args.assets_dir / "yam.yml").exists():
        raise FileNotFoundError(f"Generated YAM assets are missing: {args.assets_dir / 'yam.yml'}")
    if args.skip_generate_assets and (args.assets_dir / "yam.yml").exists():
        return
    if (args.assets_dir / "yam.yml").exists() and not args.regenerate_assets:
        return

    generator = Path(__file__).resolve().parent / "generate_yam_curobo_assets.py"
    cmd = [sys.executable, str(generator), "--output-dir", str(args.assets_dir)]
    if args.validate_assets:
        cmd.append("--validate")
    subprocess.run(cmd, check=True)


def _install_yam_module(tiptop_root: Path) -> None:
    dst = tiptop_root / "cutamp" / "cutamp" / "robots" / "yam.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not BRIDGE_SOURCE.exists():
        raise FileNotFoundError(f"Missing bridge source: {BRIDGE_SOURCE}")
    source = BRIDGE_SOURCE.read_text()
    _write_if_changed(dst, source)


def _patch_cutamp_robots(tiptop_root: Path) -> None:
    path = tiptop_root / "cutamp" / "cutamp" / "robots" / "__init__.py"
    text = _read(path)
    text = _insert_after(
        text,
        "from .ur5 import load_ur5_rerun, ur5_home, get_ur5_gripper_spheres, get_ur5_ik_solver, get_ur5_kinematics_model",
        YAM_IMPORT,
        path,
    )
    text = _insert_before(text, "robot_to_fns = {", YAM_CONTAINER.strip(), path)
    text = _insert_after(text, "robot_to_fns = {", YAM_ROBOT_ENTRY, path)
    _write_if_changed(path, text)


def _patch_cutamp_config(tiptop_root: Path) -> None:
    path = tiptop_root / "cutamp" / "cutamp" / "config.py"
    text = _read(path)
    text = _replace(
        text,
        'robot: Literal["panda", "fr3_robotiq", "ur5", "panda_robotiq", "fr3_franka"] = "panda"',
        'robot: Literal["panda", "fr3_robotiq", "ur5", "panda_robotiq", "fr3_franka", "yam"] = "panda"',
        path,
    )
    text = _replace(
        text,
        'if config.robot not in {"panda", "fr3_robotiq", "ur5", "panda_robotiq", "fr3_franka"}:',
        'if config.robot not in {"panda", "fr3_robotiq", "ur5", "panda_robotiq", "fr3_franka", "yam"}:',
        path,
    )
    _write_if_changed(path, text)


def _patch_tamp_world(tiptop_root: Path) -> None:
    path = tiptop_root / "cutamp" / "cutamp" / "tamp_world.py"
    text = _read(path)
    text = _insert_after(
        text,
        "from cutamp.robots.ur5 import ur5_curobo_cfg, get_ur5_ik_solver",
        "from cutamp.robots.yam import yam_curobo_cfg, get_yam_ik_solver",
        path,
    )
    text = _replace(
        text,
        '        elif self.robot_name == "ur5":\n            self.ik_solver = get_ur5_ik_solver(self.world_cfg)\n        else:',
        '        elif self.robot_name == "ur5":\n            self.ik_solver = get_ur5_ik_solver(self.world_cfg)\n        elif self.robot_name == "yam":\n            self.ik_solver = get_yam_ik_solver(self.world_cfg)\n        else:',
        path,
    )
    text = _replace(
        text,
        '        elif self.robot_name == "ur5":\n            robot_cfg = ur5_curobo_cfg()\n        else:',
        '        elif self.robot_name == "ur5":\n            robot_cfg = ur5_curobo_cfg()\n        elif self.robot_name == "yam":\n            robot_cfg = yam_curobo_cfg()\n        else:',
        path,
    )
    _write_if_changed(path, text)


def _patch_tiptop_motion_planning(tiptop_root: Path) -> None:
    path = tiptop_root / "tiptop" / "motion_planning.py"
    text = _read(path)
    text = _insert_after(
        text,
        "from cutamp.robots.ur5 import get_ur5_ik_solver, ur5_curobo_cfg",
        "from cutamp.robots.yam import get_yam_ik_solver, yam_curobo_cfg",
        path,
    )
    text = _replace(
        text,
        "    load_ur5_container,\n    panda_robotiq_curobo_cfg,",
        "    load_ur5_container,\n    load_yam_container,\n    panda_robotiq_curobo_cfg,",
        path,
    )
    text = _replace(
        text,
        '        elif cfg.robot.type == "ur5":\n            ik_solver = get_ur5_ik_solver(world_cfg)\n            container = load_ur5_container(TensorDeviceType())\n        else:',
        '        elif cfg.robot.type == "ur5":\n            ik_solver = get_ur5_ik_solver(world_cfg)\n            container = load_ur5_container(TensorDeviceType())\n        elif cfg.robot.type == "yam":\n            ik_solver = get_yam_ik_solver(world_cfg)\n            container = load_yam_container(TensorDeviceType())\n        else:',
        path,
    )
    text = _replace(
        text,
        '    elif cfg.robot.type == "ur5":\n        robot_cfg = ur5_curobo_cfg()\n    else:',
        '    elif cfg.robot.type == "ur5":\n        robot_cfg = ur5_curobo_cfg()\n    elif cfg.robot.type == "yam":\n        robot_cfg = yam_curobo_cfg()\n    else:',
        path,
    )
    _write_if_changed(path, text)


def _patch_tiptop_workspace(tiptop_root: Path) -> None:
    path = tiptop_root / "tiptop" / "workspace.py"
    text = _read(path)
    text = _replace(
        text,
        '    elif cfg.robot.type == "ur5":\n        cuboids = ur5_workspace()\n    else:',
        '    elif cfg.robot.type == "ur5":\n        cuboids = ur5_workspace()\n    elif cfg.robot.type == "yam":\n        cuboids = ()\n    else:',
        path,
    )
    _write_if_changed(path, text)


def _patch_tiptop_utils(tiptop_root: Path) -> None:
    path = tiptop_root / "tiptop" / "utils.py"
    text = _read(path)
    text = _replace(
        text,
        "    load_ur5_rerun,\n)",
        "    load_ur5_rerun,\n    load_yam_rerun,\n)",
        path,
    )
    text = _replace(
        text,
        '    elif robot_type == "ur5":\n        return load_ur5_rerun()\n    else:',
        '    elif robot_type == "ur5":\n        return load_ur5_rerun()\n    elif robot_type == "yam":\n        return load_yam_rerun()\n    else:',
        path,
    )
    _write_if_changed(path, text)


def install(args: argparse.Namespace) -> None:
    tiptop_root = args.tiptop_root.resolve()
    if not (tiptop_root / "cutamp" / "cutamp").exists() or not (tiptop_root / "tiptop").exists():
        raise FileNotFoundError(f"Could not find TiPToP checkout at {tiptop_root}")

    _ensure_assets(args)
    _install_yam_module(tiptop_root)
    _patch_cutamp_robots(tiptop_root)
    _patch_cutamp_config(tiptop_root)
    _patch_tamp_world(tiptop_root)
    _patch_tiptop_motion_planning(tiptop_root)
    _patch_tiptop_workspace(tiptop_root)
    _patch_tiptop_utils(tiptop_root)
    print("YAM TiPToP bridge installed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiptop-root", type=Path, default=DEFAULT_TIPTOP_ROOT)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--skip-generate-assets", action="store_true")
    parser.add_argument("--regenerate-assets", action="store_true")
    parser.add_argument("--validate-assets", action="store_true")
    args = parser.parse_args()
    install(args)


if __name__ == "__main__":
    main()
