"""Replay a TiPToP plan in the MuJoCo YAM simulator.

This is simulator-only. It loads the local YAM MJCF, adds the same simple cube
scene used by save_tiptop_h5_from_yam.py, then plays a saved tiptop_plan.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np

from save_tiptop_h5_from_yam import YAM_DIR, _make_scene_xml, _parse_vec


DEFAULT_CAMERA_POS = np.array([0.65, -0.30, 0.42], dtype=np.float64)
DEFAULT_CAMERA_TARGET = np.array([0.45, 0.0, 0.025], dtype=np.float64)
DEFAULT_CUBE_POS = np.array([0.45, 0.0, 0.025], dtype=np.float64)
YAM_CUROBO_TO_MUJOCO_Q_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)


def _curobo_q_to_mujoco(q: np.ndarray, convert: bool) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * YAM_CUROBO_TO_MUJOCO_Q_SIGNS if convert else q


def _latest_plan() -> Path:
    roots = [
        Path("/tmp/tiptop_yam_sim_bootstrap"),
        Path("/tmp/tiptop_yam_debug"),
        Path("/tmp/tiptop_yam_debug_builtin_grasps"),
    ]
    plans: list[Path] = []
    for root in roots:
        if root.exists():
            plans.extend(root.glob("*/tiptop_plan.json"))

    if not plans:
        raise FileNotFoundError(
            "No tiptop_plan.json found under /tmp/tiptop_yam_sim_bootstrap or /tmp/tiptop_yam_debug*"
        )
    return max(plans, key=lambda p: p.stat().st_mtime)


def _load_plan(path: Path | None) -> tuple[Path, dict[str, Any]]:
    plan_path = path if path is not None else _latest_plan()
    with plan_path.open() as f:
        plan = json.load(f)
    return plan_path, plan


def _load_model(camera_pos: np.ndarray, camera_target: np.ndarray, cube_pos: np.ndarray, fovy: float) -> mujoco.MjModel:
    scene_xml = _make_scene_xml(
        camera_pos=camera_pos,
        camera_target=camera_target,
        cube_pos=cube_pos,
        fovy=fovy,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=YAM_DIR, delete=False) as tmp:
        tmp.write(scene_xml)
        tmp_xml = Path(tmp.name)

    try:
        return mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        tmp_xml.unlink(missing_ok=True)


def _set_cube_pose(model: mujoco.MjModel, data: mujoco.MjData, cube_pos: np.ndarray) -> None:
    cube_jid = model.joint("tiptop_cube_freejoint").id
    cube_qadr = model.jnt_qposadr[cube_jid]
    data.qpos[cube_qadr : cube_qadr + 3] = cube_pos
    data.qpos[cube_qadr + 3 : cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])


def _set_gripper(model: mujoco.MjModel, data: mujoco.MjData, value: float, teleport: bool) -> None:
    if model.nu >= 7:
        data.ctrl[6] = value
    if teleport and model.nq >= 8:
        data.qpos[6] = value
        data.qpos[7] = -value


def _reset_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_init: np.ndarray,
    cube_pos: np.ndarray,
    open_width: float,
    teleport: bool,
) -> None:
    mujoco.mj_resetData(model, data)

    home_id = model.key("home").id
    data.qpos[:] = model.key_qpos[home_id]
    data.ctrl[:] = model.key_ctrl[home_id]

    data.qpos[:6] = q_init
    data.ctrl[:6] = q_init
    _set_cube_pose(model, data, cube_pos)
    _set_gripper(model, data, open_width, teleport=True)

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _set_arm(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, teleport: bool) -> None:
    if q.shape != (6,):
        raise ValueError(f"Expected 6 arm joints, got shape {q.shape}")
    data.ctrl[:6] = q
    if teleport:
        data.qpos[:6] = q
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)


def _sync_for(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    speed: float,
    teleport: bool,
) -> bool:
    duration = max(duration / max(speed, 1e-6), 0.0)
    if teleport:
        viewer.sync()
        time.sleep(duration)
        return viewer.is_running()

    end_time = time.monotonic() + duration
    while viewer.is_running() and time.monotonic() < end_time:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    return viewer.is_running()


def _play_plan(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: dict[str, Any],
    cube_pos: np.ndarray,
    open_width: float,
    close_width: float,
    speed: float,
    teleport: bool,
    convert_curobo_to_mujoco_q: bool,
) -> None:
    q_init = _curobo_q_to_mujoco(plan.get("q_init"), convert_curobo_to_mujoco_q)
    if q_init.shape != (6,):
        raise ValueError("tiptop_plan.json must contain q_init with 6 values")

    _reset_scene(model, data, q_init, cube_pos, open_width, teleport=teleport)
    viewer.sync()
    time.sleep(0.4 / max(speed, 1e-6))

    for step in plan.get("steps", []):
        if not viewer.is_running():
            return

        step_type = step.get("type")
        if step_type == "trajectory":
            dt = float(step.get("dt", 0.04))
            positions = step.get("positions", [])
            print(f"Playing trajectory: {step.get('label', '<unnamed>')} ({len(positions)} waypoints)")
            for q in positions:
                _set_arm(
                    model,
                    data,
                    _curobo_q_to_mujoco(q, convert_curobo_to_mujoco_q),
                    teleport=teleport,
                )
                if not _sync_for(viewer, model, data, dt, speed, teleport):
                    return
        elif step_type == "gripper":
            action = step.get("action")
            target = close_width if action == "close" else open_width
            print(f"Playing gripper action: {step.get('label', '<unnamed>')} -> {action}")
            _set_gripper(model, data, target, teleport=teleport)
            _sync_for(viewer, model, data, 0.8, speed, teleport=False)
        else:
            print(f"Skipping unknown plan step type: {step_type!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=None, help="Path to tiptop_plan.json. Defaults to latest /tmp run.")
    parser.add_argument(
        "--playback-mode",
        choices=("teleport", "servo"),
        default="teleport",
        help="teleport follows saved waypoints exactly; servo drives MuJoCo position actuators.",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--open-width", type=float, default=0.03)
    parser.add_argument("--close-width", type=float, default=0.0)
    parser.add_argument("--no-curobo-to-mujoco-q-conversion", action="store_true")
    parser.add_argument("--fovy", type=float, default=24.0)
    parser.add_argument(
        "--camera-pos",
        type=lambda s: _parse_vec(s, 3, "camera-pos"),
        default=DEFAULT_CAMERA_POS,
    )
    parser.add_argument(
        "--camera-target",
        type=lambda s: _parse_vec(s, 3, "camera-target"),
        default=DEFAULT_CAMERA_TARGET,
    )
    parser.add_argument(
        "--cube-pos",
        type=lambda s: _parse_vec(s, 3, "cube-pos"),
        default=DEFAULT_CUBE_POS,
    )
    args = parser.parse_args()

    plan_path, plan = _load_plan(args.plan)
    model = _load_model(
        camera_pos=args.camera_pos,
        camera_target=args.camera_target,
        cube_pos=args.cube_pos,
        fovy=args.fovy,
    )
    data = mujoco.MjData(model)

    print(f"Loaded plan: {plan_path}")
    print(f"Plan steps: {[step.get('type') for step in plan.get('steps', [])]}")
    print(
        "YAM replay q convention: "
        f"curobo_to_mujoco_q_conversion={'false' if args.no_curobo_to_mujoco_q_conversion else YAM_CUROBO_TO_MUJOCO_Q_SIGNS.tolist()}"
    )
    print("Viewer controls: press R to replay, close the viewer window to exit.")

    replay_requested = True

    def key_callback(keycode: int) -> None:
        nonlocal replay_requested
        if chr(keycode).lower() == "r":
            replay_requested = True

    teleport = args.playback_mode == "teleport"
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if replay_requested:
                replay_requested = False
                _play_plan(
                    viewer=viewer,
                    model=model,
                    data=data,
                    plan=plan,
                    cube_pos=args.cube_pos,
                    open_width=args.open_width,
                    close_width=args.close_width,
                    speed=args.speed,
                    teleport=teleport,
                    convert_curobo_to_mujoco_q=not args.no_curobo_to_mujoco_q_conversion,
                )
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
