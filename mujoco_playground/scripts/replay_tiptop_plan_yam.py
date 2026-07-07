"""Replay a TiPToP plan in the MuJoCo YAM simulator.

This is simulator-only. It loads the local YAM MJCF, adds the same simple cube
scene used by save_tiptop_h5_from_yam.py, then plays a saved tiptop_plan.json.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import mujoco
import numpy as np

from save_tiptop_h5_from_yam import YAM_DIR, _make_scene_xml, _parse_vec


DEFAULT_CAMERA_POS = np.array([0.65, -0.30, 0.42], dtype=np.float64)
DEFAULT_CAMERA_TARGET = np.array([0.45, 0.0, 0.025], dtype=np.float64)
DEFAULT_CUBE_POS = np.array([0.45, 0.0, 0.025], dtype=np.float64)
YAM_CUROBO_TO_MUJOCO_Q_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
MARKER_COLORS = (
    np.array([0.05, 1.0, 0.15, 1.0], dtype=np.float32),
    np.array([0.0, 0.85, 1.0, 1.0], dtype=np.float32),
    np.array([1.0, 0.9, 0.0, 1.0], dtype=np.float32),
    np.array([1.0, 0.25, 0.9, 1.0], dtype=np.float32),
)
FINGERTIP_GEOM_MARKER_COLOR = np.array([1.0, 1.0, 1.0, 0.95], dtype=np.float32)
FINGER_MIDPOINT_MARKER_COLOR = np.array([1.0, 0.45, 0.0, 1.0], dtype=np.float32)
TOOL_FRAME_MARKER_COLOR = np.array([1.0, 0.9, 0.0, 1.0], dtype=np.float32)
YAM_TILTED_GRASP_FRAME_ROT = np.array(
    [
        [-0.535512, -0.034585, 0.843819],
        [0.010009, 0.998831, 0.047290],
        [-0.844468, 0.033770, -0.534540],
    ],
    dtype=np.float64,
)


def _curobo_q_to_mujoco(q: np.ndarray, convert: bool) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * YAM_CUROBO_TO_MUJOCO_Q_SIGNS if convert else q


def _parse_names(text: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in text.split(",") if name.strip())


def _parse_vec3(text: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values")
    return values


def _latest_plan() -> Path:
    roots = [
        Path("/tmp/tiptop_yam_sim_grasp_frame"),
        Path("/tmp/tiptop_yam_sim_grasp_zdown04"),
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


def _cube_qadr(model: mujoco.MjModel) -> int:
    cube_jid = model.joint("tiptop_cube_freejoint").id
    return int(model.jnt_qposadr[cube_jid])


def _cube_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return data.qpos[_cube_qadr(model) : _cube_qadr(model) + 3].copy()


def _site_position(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    return data.site_xpos[model.site(site_name).id].copy()


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = model.site(site_name).id
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site_id]
    return pose


def _tool_from_ee(mode: str, local_offset: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if mode == "yam-tilted-grasp-frame":
        transform[:3, :3] = YAM_TILTED_GRASP_FRAME_ROT
    elif mode == "identity":
        transform[:3, :3] = np.eye(3, dtype=np.float64)
    else:
        raise ValueError(f"Unknown marker tool-frame mode: {mode}")
    transform[:3, 3] = np.asarray(local_offset, dtype=np.float64)
    return transform


def _tool_origin_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ee_site: str,
    mode: str,
    local_offset: tuple[float, float, float],
) -> np.ndarray:
    world_from_ee = _site_pose(model, data, ee_site)
    world_from_tool = world_from_ee @ np.linalg.inv(_tool_from_ee(mode, local_offset))
    return world_from_tool[:3, 3].copy()


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xpos[model.body(body_name).id].copy()


def _geom_ids_for_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = model.body(body_name).id
    return [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id]


def _nearest_geom_to_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    target_pos: np.ndarray,
) -> int | None:
    geom_ids = _geom_ids_for_body(model, body_name)
    if not geom_ids:
        return None
    return min(geom_ids, key=lambda geom_id: float(np.linalg.norm(data.geom_xpos[geom_id] - target_pos)))


def _finger_midpoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: tuple[str, ...],
) -> np.ndarray | None:
    if len(body_names) < 2:
        return None
    cube_pos = _cube_position(model, data)
    selected = [
        _nearest_geom_to_point(model, data, body_name, cube_pos)
        for body_name in body_names[:2]
    ]
    if selected[0] is None or selected[1] is None:
        return None
    return np.mean([data.geom_xpos[int(geom_id)] for geom_id in selected], axis=0)


def _set_attached_cube_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    offset: np.ndarray,
) -> None:
    cube_qadr = _cube_qadr(model)
    data.qpos[cube_qadr : cube_qadr + 3] = _site_position(model, data, site_name) + offset
    data.qpos[cube_qadr + 3 : cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])


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
    viewer: Any,
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
    viewer: Any,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: dict[str, Any],
    cube_pos: np.ndarray,
    open_width: float,
    close_width: float,
    speed: float,
    teleport: bool,
    convert_curobo_to_mujoco_q: bool,
    attach_on_close: bool,
    attach_site: str,
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
            if attach_on_close and action == "close":
                offset = _cube_position(model, data) - _site_position(model, data, attach_site)
                _set_attached_cube_pose(model, data, attach_site, offset)
            _sync_for(viewer, model, data, 0.8, speed, teleport=False)
        else:
            print(f"Skipping unknown plan step type: {step_type!r}")


def _add_marker(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    radius: float,
    color: np.ndarray,
) -> None:
    if scene.ngeom >= len(scene.geoms):
        return
    marker_mat = np.eye(3, dtype=np.float64).reshape(-1)
    marker_size = np.array([radius, radius, radius], dtype=np.float64)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        marker_size,
        pos,
        marker_mat,
        color,
    )
    scene.ngeom += 1


def _add_debug_markers(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    marker_sites: tuple[str, ...],
    marker_bodies: tuple[str, ...],
    marker_fingertip_geoms: tuple[str, ...],
    marker_finger_midpoint_bodies: tuple[str, ...],
    marker_tool_frame_mode: str | None,
    marker_tool_frame_local_offset: tuple[float, float, float],
    marker_tool_frame_site: str,
    marker_radius: float,
) -> None:
    scene = renderer.scene
    for idx, site_name in enumerate(marker_sites):
        site_id = model.site(site_name).id
        _add_marker(scene, data.site_xpos[site_id], marker_radius, MARKER_COLORS[idx % len(MARKER_COLORS)])

    body_offset = len(marker_sites)
    for idx, body_name in enumerate(marker_bodies):
        _add_marker(
            scene,
            _body_position(model, data, body_name),
            marker_radius,
            MARKER_COLORS[(body_offset + idx) % len(MARKER_COLORS)],
        )

    fingertip_radius = marker_radius * 0.45
    for body_name in marker_fingertip_geoms:
        for geom_id in _geom_ids_for_body(model, body_name):
            _add_marker(scene, data.geom_xpos[geom_id], fingertip_radius, FINGERTIP_GEOM_MARKER_COLOR)

    midpoint = _finger_midpoint(model, data, marker_finger_midpoint_bodies)
    if midpoint is not None:
        _add_marker(scene, midpoint, marker_radius * 1.35, FINGER_MIDPOINT_MARKER_COLOR)

    if marker_tool_frame_mode is not None:
        tool_pos = _tool_origin_position(
            model,
            data,
            marker_tool_frame_site,
            marker_tool_frame_mode,
            marker_tool_frame_local_offset,
        )
        _add_marker(scene, tool_pos, marker_radius * 1.45, TOOL_FRAME_MARKER_COLOR)


def _capture_frame(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    marker_sites: tuple[str, ...],
    marker_bodies: tuple[str, ...],
    marker_fingertip_geoms: tuple[str, ...],
    marker_finger_midpoint_bodies: tuple[str, ...],
    marker_tool_frame_mode: str | None,
    marker_tool_frame_local_offset: tuple[float, float, float],
    marker_tool_frame_site: str,
    marker_radius: float,
) -> np.ndarray:
    renderer.update_scene(data, camera="tiptop_cam")
    _add_debug_markers(
        renderer,
        model,
        data,
        marker_sites,
        marker_bodies,
        marker_fingertip_geoms,
        marker_finger_midpoint_bodies,
        marker_tool_frame_mode,
        marker_tool_frame_local_offset,
        marker_tool_frame_site,
        marker_radius,
    )
    return renderer.render().copy()


def _render_for(
    frames: list[np.ndarray],
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    speed: float,
    fps: float,
    teleport: bool,
    marker_sites: tuple[str, ...],
    marker_bodies: tuple[str, ...],
    marker_fingertip_geoms: tuple[str, ...],
    marker_finger_midpoint_bodies: tuple[str, ...],
    marker_tool_frame_mode: str | None,
    marker_tool_frame_local_offset: tuple[float, float, float],
    marker_tool_frame_site: str,
    marker_radius: float,
) -> None:
    duration = max(duration / max(speed, 1e-6), 0.0)
    frame_count = max(1, int(round(duration * fps)))
    for _ in range(frame_count):
        if not teleport:
            mujoco.mj_step(model, data)
        frames.append(
            _capture_frame(
                renderer,
                model,
                data,
                marker_sites,
                marker_bodies,
                marker_fingertip_geoms,
                marker_finger_midpoint_bodies,
                marker_tool_frame_mode,
                marker_tool_frame_local_offset,
                marker_tool_frame_site,
                marker_radius,
            )
        )


def _write_video(output: Path, frames: list[np.ndarray], fps: float) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".gif":
        import imageio.v2 as imageio

        imageio.mimsave(output, frames, duration=1.0 / fps)
        return output

    try:
        import mediapy as media

        media.write_video(str(output), frames, fps=fps)
        return output
    except Exception as exc:
        import imageio.v2 as imageio

        fallback = output.with_suffix(".gif")
        imageio.mimsave(fallback, frames, duration=1.0 / fps)
        print(f"Could not write {output.suffix or 'video'} via mediapy ({exc}); wrote {fallback}")
        return fallback


def _render_plan_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    plan: dict[str, Any],
    cube_pos: np.ndarray,
    open_width: float,
    close_width: float,
    speed: float,
    teleport: bool,
    convert_curobo_to_mujoco_q: bool,
    output: Path,
    width: int,
    height: int,
    fps: float,
    attach_on_close: bool,
    attach_site: str,
    marker_sites: tuple[str, ...],
    marker_bodies: tuple[str, ...],
    marker_fingertip_geoms: tuple[str, ...],
    marker_finger_midpoint_bodies: tuple[str, ...],
    marker_tool_frame_mode: str | None,
    marker_tool_frame_local_offset: tuple[float, float, float],
    marker_tool_frame_site: str,
    marker_radius: float,
) -> Path:
    q_init = _curobo_q_to_mujoco(plan.get("q_init"), convert_curobo_to_mujoco_q)
    if q_init.shape != (6,):
        raise ValueError("tiptop_plan.json must contain q_init with 6 values")

    frames: list[np.ndarray] = []
    _reset_scene(model, data, q_init, cube_pos, open_width, teleport=teleport)

    with mujoco.Renderer(model, width=width, height=height) as renderer:
        _render_for(
            frames,
            renderer,
            model,
            data,
            0.4,
            speed,
            fps,
            teleport=teleport,
            marker_sites=marker_sites,
            marker_bodies=marker_bodies,
            marker_fingertip_geoms=marker_fingertip_geoms,
            marker_finger_midpoint_bodies=marker_finger_midpoint_bodies,
            marker_tool_frame_mode=marker_tool_frame_mode,
            marker_tool_frame_local_offset=marker_tool_frame_local_offset,
            marker_tool_frame_site=marker_tool_frame_site,
            marker_radius=marker_radius,
        )

        attached_offset: np.ndarray | None = None
        for step in plan.get("steps", []):
            step_type = step.get("type")
            if step_type == "trajectory":
                dt = float(step.get("dt", 0.04))
                positions = step.get("positions", [])
                print(f"Rendering trajectory: {step.get('label', '<unnamed>')} ({len(positions)} waypoints)")
                for q in positions:
                    _set_arm(
                        model,
                        data,
                        _curobo_q_to_mujoco(q, convert_curobo_to_mujoco_q),
                        teleport=teleport,
                    )
                    if attached_offset is not None:
                        _set_attached_cube_pose(model, data, attach_site, attached_offset)
                    _render_for(
                        frames,
                        renderer,
                        model,
                        data,
                        dt,
                        speed,
                        fps,
                        teleport=teleport,
                        marker_sites=marker_sites,
                        marker_bodies=marker_bodies,
                        marker_fingertip_geoms=marker_fingertip_geoms,
                        marker_finger_midpoint_bodies=marker_finger_midpoint_bodies,
                        marker_tool_frame_mode=marker_tool_frame_mode,
                        marker_tool_frame_local_offset=marker_tool_frame_local_offset,
                        marker_tool_frame_site=marker_tool_frame_site,
                        marker_radius=marker_radius,
                    )
            elif step_type == "gripper":
                action = step.get("action")
                target = close_width if action == "close" else open_width
                print(f"Rendering gripper action: {step.get('label', '<unnamed>')} -> {action}")
                _set_gripper(model, data, target, teleport=teleport)
                if attach_on_close and action == "close":
                    attached_offset = _cube_position(model, data) - _site_position(model, data, attach_site)
                    _set_attached_cube_pose(model, data, attach_site, attached_offset)
                _render_for(
                    frames,
                    renderer,
                    model,
                    data,
                    0.8,
                    speed,
                    fps,
                    teleport=False,
                    marker_sites=marker_sites,
                    marker_bodies=marker_bodies,
                    marker_fingertip_geoms=marker_fingertip_geoms,
                    marker_finger_midpoint_bodies=marker_finger_midpoint_bodies,
                    marker_tool_frame_mode=marker_tool_frame_mode,
                    marker_tool_frame_local_offset=marker_tool_frame_local_offset,
                    marker_tool_frame_site=marker_tool_frame_site,
                    marker_radius=marker_radius,
                )
            else:
                print(f"Skipping unknown plan step type: {step_type!r}")

    return _write_video(output, frames, fps)


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
    parser.add_argument("--video", type=Path, default=None, help="Render headless replay video instead of opening the viewer.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--attach-on-close", action="store_true")
    parser.add_argument("--attach-site", default="grasp_site")
    parser.add_argument(
        "--marker-sites",
        type=_parse_names,
        default=(),
        help="Comma-separated MuJoCo site names to draw as colored spheres in headless video.",
    )
    parser.add_argument(
        "--marker-bodies",
        type=_parse_names,
        default=(),
        help="Comma-separated MuJoCo body origins to draw as colored spheres in headless video.",
    )
    parser.add_argument(
        "--marker-fingertip-geoms",
        type=_parse_names,
        default=(),
        help="Comma-separated body names whose direct geom centers should be drawn as small white spheres.",
    )
    parser.add_argument(
        "--marker-finger-midpoint-bodies",
        type=_parse_names,
        default=(),
        help="Two body names; draw the midpoint between each body's geom nearest the cube as an orange sphere.",
    )
    parser.add_argument(
        "--marker-tool-frame-mode",
        choices=("yam-tilted-grasp-frame", "identity"),
        default=None,
        help="Draw the virtual TiPToP tool/grasp origin implied by this tool_from_ee mode as a yellow sphere.",
    )
    parser.add_argument(
        "--marker-tool-frame-local-offset",
        type=_parse_vec3,
        default=(0.0, 0.0, 0.0),
        help="Current local offset used with --marker-tool-frame-mode.",
    )
    parser.add_argument("--marker-tool-frame-site", default="grasp_site")
    parser.add_argument("--marker-radius", type=float, default=0.01)
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
    if args.marker_sites:
        for idx, site_name in enumerate(args.marker_sites):
            # Validate up front so a misspelled site does not fail halfway through rendering.
            model.site(site_name)
            color = MARKER_COLORS[idx % len(MARKER_COLORS)]
            print(f"Marker {idx + 1}: site={site_name} rgba={color.tolist()}")
    if args.marker_bodies:
        for idx, body_name in enumerate(args.marker_bodies):
            model.body(body_name)
            color = MARKER_COLORS[(len(args.marker_sites) + idx) % len(MARKER_COLORS)]
            print(f"Marker {len(args.marker_sites) + idx + 1}: body={body_name} rgba={color.tolist()}")
    if args.marker_fingertip_geoms:
        for body_name in args.marker_fingertip_geoms:
            model.body(body_name)
            print(f"Fingertip geom markers: body={body_name} rgba={FINGERTIP_GEOM_MARKER_COLOR.tolist()}")
    if args.marker_finger_midpoint_bodies:
        if len(args.marker_finger_midpoint_bodies) != 2:
            raise ValueError("--marker-finger-midpoint-bodies expects exactly two body names")
        for body_name in args.marker_finger_midpoint_bodies:
            model.body(body_name)
        print(
            "Finger midpoint marker: "
            f"bodies={list(args.marker_finger_midpoint_bodies)} "
            f"rgba={FINGER_MIDPOINT_MARKER_COLOR.tolist()}"
        )
    if args.marker_tool_frame_mode is not None:
        model.site(args.marker_tool_frame_site)
        print(
            "Virtual tool marker: "
            f"mode={args.marker_tool_frame_mode} "
            f"offset={args.marker_tool_frame_local_offset} "
            f"ee_site={args.marker_tool_frame_site} "
            f"rgba={TOOL_FRAME_MARKER_COLOR.tolist()}"
        )

    if args.video is not None:
        video_path = _render_plan_video(
            model=model,
            data=data,
            plan=plan,
            cube_pos=args.cube_pos,
            open_width=args.open_width,
            close_width=args.close_width,
            speed=args.speed,
            teleport=args.playback_mode == "teleport",
            convert_curobo_to_mujoco_q=not args.no_curobo_to_mujoco_q_conversion,
            output=args.video,
            width=args.width,
            height=args.height,
            fps=args.fps,
            attach_on_close=args.attach_on_close,
            attach_site=args.attach_site,
            marker_sites=args.marker_sites,
            marker_bodies=args.marker_bodies,
            marker_fingertip_geoms=args.marker_fingertip_geoms,
            marker_finger_midpoint_bodies=args.marker_finger_midpoint_bodies,
            marker_tool_frame_mode=args.marker_tool_frame_mode,
            marker_tool_frame_local_offset=args.marker_tool_frame_local_offset,
            marker_tool_frame_site=args.marker_tool_frame_site,
            marker_radius=args.marker_radius,
        )
        print(f"Wrote replay video: {video_path}")
        return

    mujoco_viewer = importlib.import_module("mujoco.viewer")

    print("Viewer controls: press R to replay, close the viewer window to exit.")

    replay_requested = True

    def key_callback(keycode: int) -> None:
        nonlocal replay_requested
        if chr(keycode).lower() == "r":
            replay_requested = True

    teleport = args.playback_mode == "teleport"
    with mujoco_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
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
                    attach_on_close=args.attach_on_close,
                    attach_site=args.attach_site,
                )
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
