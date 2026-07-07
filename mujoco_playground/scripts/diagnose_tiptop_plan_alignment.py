#!/usr/bin/env python3
"""Diagnose where a saved TiPToP YAM plan places the gripper in MuJoCo.

This is simulator-only. It loads a saved ``tiptop_plan.json``, jumps to the
last arm waypoint before the first gripper close, and reports distances from
MuJoCo gripper frames to the cube. If cuRobo/TiPToP are importable, it also
compares cuRobo's YAM end-effector frame against the MuJoCo sites.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import mujoco
import numpy as np

from save_tiptop_h5_from_yam import _make_scene_xml, _parse_vec
from yam_assets import YAM_DIR, require_yam_file


DEFAULT_CAMERA_POS = np.array([0.65, -0.30, 0.42], dtype=np.float64)
DEFAULT_CAMERA_TARGET = np.array([0.45, 0.0, 0.025], dtype=np.float64)
DEFAULT_CUBE_POS = np.array([0.45, 0.0, 0.025], dtype=np.float64)
YAM_CUROBO_TO_MUJOCO_Q_SIGNS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0], dtype=np.float64)
YAM_TILTED_GRASP_FRAME_ROT = np.array(
    [
        [-0.535512, -0.034585, 0.843819],
        [0.010009, 0.998831, 0.047290],
        [-0.844468, 0.033770, -0.534540],
    ],
    dtype=np.float64,
)
YAM_SIDE_PINCH_Y_UP_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
YAM_SIDE_PINCH_Y_DOWN_ROT = np.array(
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
YAM_SIDE_PINCH_FINGER_MIDPOINT = np.array([0.017, 0.0, -0.0212], dtype=np.float64)


def _latest_plan() -> Path:
    roots = [
        Path("/tmp/tiptop_yam_sim_grasp_frame"),
        Path("/tmp/tiptop_yam_sim_grasp_zdown04"),
        Path("/tmp/tiptop_yam_sim_bootstrap"),
        Path("/tmp/tiptop_yam_debug"),
    ]
    plans: list[Path] = []
    for root in roots:
        if root.exists():
            plans.extend(root.glob("*/tiptop_plan.json"))
    if not plans:
        raise FileNotFoundError("No tiptop_plan.json found under known /tmp TiPToP run dirs")
    return max(plans, key=lambda p: p.stat().st_mtime)


def _load_plan(path: Path | None) -> tuple[Path, dict[str, Any]]:
    plan_path = path if path is not None else _latest_plan()
    with plan_path.open() as f:
        return plan_path, json.load(f)


def _pre_close_q(plan: dict[str, Any]) -> tuple[np.ndarray, str]:
    q = np.asarray(plan.get("q_init"), dtype=np.float64)
    label = "q_init"
    for step in plan.get("steps", []):
        if step.get("type") == "trajectory":
            positions = step.get("positions") or []
            if positions:
                q = np.asarray(positions[-1], dtype=np.float64)
                label = str(step.get("label", "trajectory"))
        elif step.get("type") == "gripper" and step.get("action") == "close":
            break
    if q.shape != (6,):
        raise ValueError(f"Expected 6 arm joints, got {q.shape}")
    return q, label


def _curobo_q_to_mujoco(q: np.ndarray, convert: bool) -> np.ndarray:
    return np.asarray(q, dtype=np.float64) * YAM_CUROBO_TO_MUJOCO_Q_SIGNS if convert else np.asarray(q, dtype=np.float64)


def _load_model(camera_pos: np.ndarray, camera_target: np.ndarray, cube_pos: np.ndarray, fovy: float) -> mujoco.MjModel:
    require_yam_file("yam.xml")
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


def _set_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_mujoco: np.ndarray,
    cube_pos: np.ndarray,
    open_width: float,
) -> None:
    mujoco.mj_resetData(model, data)
    home_id = model.key("home").id
    data.qpos[:] = model.key_qpos[home_id]
    data.ctrl[:] = model.key_ctrl[home_id]
    data.qpos[:6] = q_mujoco
    data.ctrl[:6] = q_mujoco
    _set_cube_pose(model, data, cube_pos)
    if model.nq >= 8:
        data.qpos[6] = open_width
        data.qpos[7] = -open_width
    if model.nu >= 7:
        data.ctrl[6] = open_width
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray | None:
    try:
        return data.site_xpos[model.site(name).id].copy()
    except KeyError:
        return None


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray | None:
    try:
        return data.xpos[model.body(name).id].copy()
    except KeyError:
        return None


def _body_rot(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray | None:
    try:
        return data.xmat[model.body(name).id].reshape(3, 3).copy()
    except KeyError:
        return None


def _site_pose(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray | None:
    try:
        site_id = model.site(name).id
    except KeyError:
        return None
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site_id]
    return pose


def _tool_from_ee(mode: str, local_offset: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if mode == "yam-tilted-grasp-frame":
        transform[:3, :3] = YAM_TILTED_GRASP_FRAME_ROT
    elif mode == "identity":
        transform[:3, :3] = np.eye(3, dtype=np.float64)
    elif mode == "yam-side-pinch-y-up":
        transform[:3, :3] = YAM_SIDE_PINCH_Y_UP_ROT
        transform[:3, 3] = -(YAM_SIDE_PINCH_Y_UP_ROT @ YAM_SIDE_PINCH_FINGER_MIDPOINT)
    elif mode == "yam-side-pinch-y-down":
        transform[:3, :3] = YAM_SIDE_PINCH_Y_DOWN_ROT
        transform[:3, 3] = -(YAM_SIDE_PINCH_Y_DOWN_ROT @ YAM_SIDE_PINCH_FINGER_MIDPOINT)
    else:
        raise ValueError(
            "Diagnostic virtual tool marker currently supports "
            "'yam-tilted-grasp-frame', 'yam-side-pinch-y-up', "
            "'yam-side-pinch-y-down', and 'identity'."
        )
    transform[:3, 3] += local_offset
    return transform


def _world_from_tool(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    mode: str,
    local_offset: np.ndarray,
) -> np.ndarray | None:
    world_from_ee = _site_pose(model, data, site_name)
    if world_from_ee is None:
        return None
    return world_from_ee @ np.linalg.inv(_tool_from_ee(mode, local_offset))


def _geom_ids_for_body(model: mujoco.MjModel, body_name: str) -> list[int]:
    try:
        body_id = model.body(body_name).id
    except KeyError:
        return []
    return [geom_id for geom_id in range(model.ngeom) if int(model.geom_bodyid[geom_id]) == body_id]


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return name if name else f"geom_{geom_id}"


def _geom_type_name(model: mujoco.MjModel, geom_id: int) -> str:
    try:
        return mujoco.mjtGeom(int(model.geom_type[geom_id])).name.replace("mjGEOM_", "").lower()
    except ValueError:
        return str(int(model.geom_type[geom_id]))


def _print_point_delta(label: str, pos: np.ndarray, cube_center: np.ndarray, cube_half_extent: float) -> None:
    delta_center = pos - cube_center
    xy = float(np.linalg.norm(delta_center[:2]))
    center_dist = float(np.linalg.norm(delta_center))
    z_above_center = float(delta_center[2])
    z_above_top = float(pos[2] - (cube_center[2] + cube_half_extent))
    print(
        f"{label:18s} pos={np.round(pos, 6).tolist()} "
        f"dist_center={center_dist:.6f} xy={xy:.6f} "
        f"z_above_center={z_above_center:.6f} z_above_cube_top={z_above_top:.6f}"
    )


def _select_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    target_pos: np.ndarray,
    mode: str,
) -> int | None:
    geom_ids = _geom_ids_for_body(model, body_name)
    if not geom_ids:
        return None
    if mode == "nearest-target":
        return min(geom_ids, key=lambda geom_id: float(np.linalg.norm(data.geom_xpos[geom_id] - target_pos)))
    if mode == "distal-local-z":
        return max(geom_ids, key=lambda geom_id: float(model.geom_pos[geom_id, 2]))
    raise ValueError(f"Unknown geom selection mode: {mode}")


def _print_geom_body_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: list[str],
    cube_center: np.ndarray,
    cube_half_extent: float,
) -> None:
    for body_name in body_names:
        geom_ids = _geom_ids_for_body(model, body_name)
        if not geom_ids:
            print(f"\nNo geoms found directly attached to body:{body_name}")
            continue

        print(f"\nGeoms directly attached to body:{body_name} (sorted by local z high-to-low):")
        for geom_id in sorted(geom_ids, key=lambda gid: float(model.geom_pos[gid, 2]), reverse=True):
            pos = data.geom_xpos[geom_id].copy()
            label = f"geom:{body_name}/{_geom_name(model, geom_id)}"
            _print_point_delta(label, pos, cube_center, cube_half_extent)
            print(
                "  "
                f"type={_geom_type_name(model, geom_id)} "
                f"local={np.round(model.geom_pos[geom_id], 6).tolist()} "
                f"size={np.round(model.geom_size[geom_id], 6).tolist()}"
            )


def _print_finger_midpoint_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: list[str],
    cube_center: np.ndarray,
    cube_half_extent: float,
    virtual_tool_pos: np.ndarray | None,
    virtual_tool_rot: np.ndarray | None,
    current_tool_offset: np.ndarray,
) -> None:
    if len(body_names) < 2:
        return

    target_points = {
        "nearest cube center": cube_center,
        "nearest cube top center": cube_center + np.array([0.0, 0.0, cube_half_extent], dtype=np.float64),
    }
    modes: list[tuple[str, str, np.ndarray | None]] = [
        ("distal local z", "distal-local-z", None),
        *[(name, "nearest-target", target) for name, target in target_points.items()],
    ]

    grasp_pos = _site_pos(model, data, "grasp_site")
    for label, mode, target_pos in modes:
        selected: list[int] = []
        for body_name in body_names[:2]:
            geom_id = _select_geom(model, data, body_name, cube_center if target_pos is None else target_pos, mode)
            if geom_id is not None:
                selected.append(geom_id)
        if len(selected) != 2:
            continue

        midpoint = np.mean([data.geom_xpos[geom_id] for geom_id in selected], axis=0)
        print(
            f"\nFinger midpoint ({label}) from "
            f"{body_names[0]}/{_geom_name(model, selected[0])} and "
            f"{body_names[1]}/{_geom_name(model, selected[1])}:"
        )
        _print_point_delta("finger_midpoint", midpoint, cube_center, cube_half_extent)
        if grasp_pos is not None:
            delta = grasp_pos - midpoint
            print(
                "  "
                f"grasp_site - finger_midpoint={np.round(delta, 6).tolist()} "
                f"norm={np.linalg.norm(delta):.6f}"
            )
        if virtual_tool_pos is not None:
            delta = virtual_tool_pos - midpoint
            print(
                "  "
                f"virtual_tool - finger_midpoint={np.round(delta, 6).tolist()} "
                f"norm={np.linalg.norm(delta):.6f}"
            )
            if virtual_tool_rot is not None:
                suggested_offset = current_tool_offset + virtual_tool_rot.T @ delta
                print(
                    "  "
                    "suggested --tool-frame-local-offset "
                    f"{','.join(f'{v:.6f}' for v in suggested_offset.tolist())}"
                )


def _print_gripper_orientation_report(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_center: np.ndarray,
    cube_half_extent: float,
    geom_bodies: list[str],
) -> None:
    link6_pos = _body_pos(model, data, "link_6")
    link6_rot = _body_rot(model, data, "link_6")
    if link6_pos is None or link6_rot is None or len(geom_bodies) < 2:
        return

    world_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    opening_axis = link6_rot[:, 0]
    finger_forward_axis = link6_rot[:, 2]
    palm_side_axis = link6_rot[:, 1]

    selected = [
        _select_geom(model, data, geom_bodies[0], cube_center, "nearest-target"),
        _select_geom(model, data, geom_bodies[1], cube_center, "nearest-target"),
    ]

    print("\nGripper orientation sanity check:")
    print(
        "  "
        f"opening_axis/link6_x={np.round(opening_axis, 6).tolist()} "
        f"dot_world_z={float(np.dot(opening_axis, world_z)):.6f}"
    )
    print(
        "  "
        f"finger_forward/link6_z={np.round(finger_forward_axis, 6).tolist()} "
        f"dot_world_z={float(np.dot(finger_forward_axis, world_z)):.6f}"
    )
    print(
        "  "
        f"palm_side/link6_y={np.round(palm_side_axis, 6).tolist()} "
        f"dot_world_z={float(np.dot(palm_side_axis, world_z)):.6f}"
    )

    if selected[0] is None or selected[1] is None:
        return

    left_pos = data.geom_xpos[int(selected[0])].copy()
    right_pos = data.geom_xpos[int(selected[1])].copy()
    midpoint = 0.5 * (left_pos + right_pos)
    pair_vec = right_pos - left_pos
    pair_dist = float(np.linalg.norm(pair_vec))
    pair_dir = pair_vec / max(pair_dist, 1e-9)
    cube_full_width = cube_half_extent * 2.0
    center_error = midpoint - cube_center
    print(
        "  "
        f"nearest-center fingertip pair={_geom_name(model, int(selected[0]))}/"
        f"{_geom_name(model, int(selected[1]))} center_dist={pair_dist:.6f} "
        f"cube_width={cube_full_width:.6f}"
    )
    print(
        "  "
        f"pair_dir={np.round(pair_dir, 6).tolist()} "
        f"alignment_with_opening_axis={abs(float(np.dot(pair_dir, opening_axis))):.6f}"
    )
    print(
        "  "
        f"midpoint_minus_cube_center={np.round(center_error, 6).tolist()} "
        f"xy={np.linalg.norm(center_error[:2]):.6f} z={center_error[2]:.6f}"
    )
    if abs(float(np.dot(finger_forward_axis, world_z))) > 0.65:
        print("  warning: finger forward axis is mostly vertical; this looks more like a top/down press than a side pinch.")
    if np.linalg.norm(center_error[:2]) > 0.02:
        print("  warning: fingertip midpoint is more than 2 cm off the cube center in XY.")


def _install_tiptop_paths() -> None:
    tiptop_dir = os.environ.get("TIPTOP_PACKAGE_DIR")
    if not tiptop_dir:
        return
    root = Path(tiptop_dir).expanduser().resolve()
    for path in reversed((root / "cutamp", root / "curobo" / "src", root)):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _try_curobo_ee(q_curobo: np.ndarray) -> np.ndarray | None:
    _install_tiptop_paths()
    try:
        import torch
        from curobo.types.base import TensorDeviceType
        from cutamp.robots import load_yam_container
    except Exception as exc:
        print(f"\ncuRobo comparison skipped: {type(exc).__name__}: {exc}")
        return None

    tensor_args = TensorDeviceType()
    container = load_yam_container(tensor_args)
    q_t = tensor_args.to_device(q_curobo).view(1, -1)
    with torch.no_grad():
        ee_pose = container.kin_model.get_state(q_t).ee_pose.get_numpy_matrix()[0]
    return np.asarray(ee_pose, dtype=np.float64)


def _print_curobo_comparison(
    ee_pose: np.ndarray | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    compare_sites: list[str],
    compare_bodies: list[str],
) -> None:
    if ee_pose is None:
        return
    ee_pos = ee_pose[:3, 3]
    print(f"\ncuRobo EE/grasp_frame pos={np.round(ee_pos, 6).tolist()}")
    for name in compare_sites:
        pos = _site_pos(model, data, name)
        if pos is not None:
            print(f"  cuRobo EE - site {name:10s}: {np.round(ee_pos - pos, 6).tolist()} norm={np.linalg.norm(ee_pos - pos):.6f}")
    for name in compare_bodies:
        pos = _body_pos(model, data, name)
        if pos is not None:
            print(f"  cuRobo EE - body {name:10s}: {np.round(ee_pos - pos, 6).tolist()} norm={np.linalg.norm(ee_pos - pos):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--cube-pos", type=lambda s: _parse_vec(s, 3, "cube-pos"), default=DEFAULT_CUBE_POS)
    parser.add_argument("--cube-half-extent", type=float, default=0.025)
    parser.add_argument("--open-width", type=float, default=0.03)
    parser.add_argument("--no-curobo-to-mujoco-q-conversion", action="store_true")
    parser.add_argument("--fovy", type=float, default=24.0)
    parser.add_argument("--camera-pos", type=lambda s: _parse_vec(s, 3, "camera-pos"), default=DEFAULT_CAMERA_POS)
    parser.add_argument("--camera-target", type=lambda s: _parse_vec(s, 3, "camera-target"), default=DEFAULT_CAMERA_TARGET)
    parser.add_argument("--sites", default="grasp_site,tcp_site")
    parser.add_argument("--bodies", default="link_6")
    parser.add_argument(
        "--geom-bodies",
        default="lf_down,rf_down",
        help="Comma-separated body names whose direct geom centers should be compared with the cube.",
    )
    parser.add_argument(
        "--tool-frame-mode",
        choices=("yam-tilted-grasp-frame", "yam-side-pinch-y-up", "yam-side-pinch-y-down", "identity"),
        default=None,
        help="If set, print the virtual TiPToP tool/grasp origin implied by this tool_from_ee mode.",
    )
    parser.add_argument(
        "--tool-frame-local-offset",
        type=lambda s: _parse_vec(s, 3, "tool-frame-local-offset"),
        default=np.zeros(3, dtype=np.float64),
        help="Current local offset used with --tool-frame-mode.",
    )
    parser.add_argument("--tool-frame-ee-site", default="grasp_site")
    args = parser.parse_args()

    plan_path, plan = _load_plan(args.plan)
    q_curobo, q_label = _pre_close_q(plan)
    q_mujoco = _curobo_q_to_mujoco(q_curobo, convert=not args.no_curobo_to_mujoco_q_conversion)

    model = _load_model(args.camera_pos, args.camera_target, args.cube_pos, args.fovy)
    data = mujoco.MjData(model)
    _set_state(model, data, q_mujoco, args.cube_pos, args.open_width)

    sites = [name.strip() for name in args.sites.split(",") if name.strip()]
    bodies = [name.strip() for name in args.bodies.split(",") if name.strip()]
    geom_bodies = [name.strip() for name in args.geom_bodies.split(",") if name.strip()]

    print(f"Loaded plan: {plan_path}")
    print(f"Using final pre-close q from: {q_label}")
    print(f"q_curobo={np.round(q_curobo, 6).tolist()}")
    print(f"q_mujoco={np.round(q_mujoco, 6).tolist()}")
    print(f"cube_center={np.round(args.cube_pos, 6).tolist()} cube_top_z={args.cube_pos[2] + args.cube_half_extent:.6f}")

    print("\nMuJoCo frame distances to cube:")
    for name in sites:
        pos = _site_pos(model, data, name)
        if pos is None:
            print(f"site {name!r} not found")
            continue
        _print_point_delta(f"site:{name}", pos, args.cube_pos, args.cube_half_extent)
    for name in bodies:
        pos = _body_pos(model, data, name)
        if pos is None:
            print(f"body {name!r} not found")
            continue
        _print_point_delta(f"body:{name}", pos, args.cube_pos, args.cube_half_extent)

    virtual_tool_pos = None
    virtual_tool_rot = None
    if args.tool_frame_mode is not None:
        world_from_tool = _world_from_tool(
            model,
            data,
            args.tool_frame_ee_site,
            args.tool_frame_mode,
            args.tool_frame_local_offset,
        )
        if world_from_tool is None:
            print(f"tool-frame ee site {args.tool_frame_ee_site!r} not found")
        else:
            virtual_tool_pos = world_from_tool[:3, 3].copy()
            virtual_tool_rot = world_from_tool[:3, :3].copy()
            print("\nVirtual TiPToP tool/grasp origin:")
            _print_point_delta("virtual_tool", virtual_tool_pos, args.cube_pos, args.cube_half_extent)

    if geom_bodies:
        _print_gripper_orientation_report(model, data, args.cube_pos, args.cube_half_extent, geom_bodies)
        _print_geom_body_report(model, data, geom_bodies, args.cube_pos, args.cube_half_extent)
        _print_finger_midpoint_report(
            model,
            data,
            geom_bodies,
            args.cube_pos,
            args.cube_half_extent,
            virtual_tool_pos,
            virtual_tool_rot,
            args.tool_frame_local_offset,
        )

    ee_pose = _try_curobo_ee(q_curobo)
    _print_curobo_comparison(ee_pose, model, data, sites, bodies)


if __name__ == "__main__":
    main()
