#!/usr/bin/env python3
"""Simulator-only TiPToP H5 runner for testing YAM tool-frame variants.

This wrapper does not talk to the robot daemon or execute a real arm. It reuses
TiPToP's offline H5 path and monkeypatches YAM planning settings in memory.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
from pathlib import Path
import sys


TIPTOP_PACKAGE_DIR = os.environ.get("TIPTOP_PACKAGE_DIR")
YAM_MUJOCO_TO_CUROBO_Q_SIGNS = (1.0, 1.0, 1.0, 1.0, 1.0, -1.0)
YAM_HOME_QPOS = (
    -0.0368123903,
    0.8585107195,
    1.4494163424,
    -1.1568245975,
    -0.0783932250,
    -0.0097276265,
)


def _parse_vec3(text: str) -> tuple[float, float, float]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values")
    return values


def _mujoco_q_to_curobo(q):
    """Convert simulator joint vectors to the YAM cuRobo/URDF convention."""
    import numpy as np

    try:
        import torch
    except Exception:  # pragma: no cover - torch is available in normal runs.
        torch = None

    if torch is not None and torch.is_tensor(q):
        signs = torch.tensor(YAM_MUJOCO_TO_CUROBO_Q_SIGNS, device=q.device, dtype=q.dtype)
        return q * signs

    arr = np.asarray(q)
    signs = np.asarray(YAM_MUJOCO_TO_CUROBO_Q_SIGNS, dtype=arr.dtype)
    return arr * signs


def _prepare_tiptop_package_dir() -> None:
    if TIPTOP_PACKAGE_DIR:
        tiptop_path = Path(TIPTOP_PACKAGE_DIR).expanduser().resolve()
    else:
        tiptop_path = Path.cwd().resolve()

    if not tiptop_path.exists():
        raise FileNotFoundError(
            f"TIPTOP_PACKAGE_DIR points to a missing path: {tiptop_path}"
        )

    if (tiptop_path / "tiptop" / "__init__.py").exists():
        tiptop_root = tiptop_path
        tiptop_package_dir = tiptop_path / "tiptop"
    elif (tiptop_path / "__init__.py").exists() and tiptop_path.name == "tiptop":
        tiptop_root = tiptop_path.parent
        tiptop_package_dir = tiptop_path
    elif TIPTOP_PACKAGE_DIR:
        raise FileNotFoundError(
            "TIPTOP_PACKAGE_DIR must point to a TiPToP checkout root or package "
            f"directory, got: {tiptop_path}"
        )
    else:
        return

    # Avoid the checkout root's namespace-style cutamp/ directory shadowing the
    # real cuTAMP package. The package source lives one level deeper.
    for path in reversed((
        tiptop_root / "cutamp",
        tiptop_root / "curobo" / "src",
        tiptop_root,
    )):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.chdir(tiptop_package_dir)


def _check_tiptop_dependencies() -> None:
    missing = [
        name
        for name in ("torch", "tiptop", "cutamp")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise ModuleNotFoundError(
            "Missing TiPToP planner dependencies: "
            + ", ".join(missing)
            + ". Install TiPToP/cuTAMP in this venv, or set TIPTOP_PACKAGE_DIR "
            "to a local TiPToP checkout that Python can import."
        )


def _tiptop_h5_module():
    """Return TiPToP's offline H5 module across old/new module names."""
    try:
        return importlib.import_module("tiptop.tiptop_h5")
    except ModuleNotFoundError as exc:
        if exc.name != "tiptop.tiptop_h5":
            raise
        return importlib.import_module("tiptop.tiptop_offline")


def _install_gemini_json_repair_patch() -> None:
    """Repair a common malformed Gemini bbox response seen in debug runs."""
    import json
    import re

    import tiptop.perception.gemini as gemini

    original_parse_response = gemini._parse_response

    def clean_response_text(response_text: str) -> str:
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "")
        elif cleaned.startswith("```"):
            cleaned = cleaned.replace("```", "")
        return cleaned.strip()

    def repair_bbox_arrays(cleaned: str) -> str:
        # Occasionally Gemini returns:
        # {"box_2d": [ymin, xmin, "label": "...", "ymax": ymax, "xmax": xmax]}
        # instead of the requested:
        # {"box_2d": [ymin, xmin, ymax, xmax], "label": "..."}
        pattern = re.compile(
            r'\{\s*"box_2d"\s*:\s*\[\s*'
            r"(-?\d+)\s*,\s*(-?\d+)\s*,\s*"
            r'"label"\s*:\s*"([^"]+)"\s*,\s*'
            r'"ymax"\s*:\s*(-?\d+)\s*,\s*'
            r'"xmax"\s*:\s*(-?\d+)\s*'
            r"\]\s*\}"
        )

        def repl(match: re.Match[str]) -> str:
            ymin, xmin, label, ymax, xmax = match.groups()
            return json.dumps(
                {
                    "box_2d": [int(ymin), int(xmin), int(ymax), int(xmax)],
                    "label": label,
                }
            )

        return pattern.sub(repl, cleaned)

    def parse_response_with_bbox_repair(response_text: str):
        try:
            return original_parse_response(response_text)
        except ValueError:
            cleaned = clean_response_text(response_text)
            repaired = repair_bbox_arrays(cleaned)
            if repaired == cleaned:
                raise
            result = json.loads(repaired)
            bboxes = result.get("bboxes", [])
            grounded_atoms = [
                {"predicate": spec["name"], "args": spec["args"]}
                for spec in result.get("predicates", [])
                if spec.get("name") and spec.get("args")
            ]
            print("YAM debug run: repaired malformed Gemini bbox JSON", flush=True)
            return bboxes, grounded_atoms

    gemini._parse_response = parse_response_with_bbox_repair


def _install_cached_perception_patch(run_dir: Path) -> None:
    """Reuse a saved TiPToP perception scene instead of calling Gemini/SAM2/M2T2."""
    import json
    import shutil

    import dill
    import torch
    from tiptop.tiptop_run import ProcessedScene

    tiptop_h5 = _tiptop_h5_module()
    run_dir = run_dir.expanduser().resolve()
    perception_dir = run_dir / "perception"
    env_path = perception_dir / "cutamp_env.pkl"
    grasps_path = perception_dir / "grasps.pt"
    metadata_path = run_dir / "metadata.json"
    required = (env_path, grasps_path, metadata_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot reuse perception; missing cached artifact(s): " + ", ".join(missing)
        )

    def copy_cached_artifacts(save_dir: Path) -> None:
        save_dir.mkdir(parents=True, exist_ok=True)
        dst_perception = save_dir / "perception"
        dst_perception.mkdir(exist_ok=True)
        for name in ("rgb.png", "bboxes_viz.png", "masks_viz.png"):
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, save_dir / name)
        for src in perception_dir.iterdir():
            if src.is_file():
                shutil.copy2(src, dst_perception / src.name)

    def surfaces_from_env(env):
        type_to_objects = getattr(env, "type_to_objects", {}) or {}
        surfaces = list(type_to_objects.get("Surface", []))
        if surfaces:
            return surfaces
        return [
            obj
            for obj in getattr(env, "statics", [])
            if getattr(obj, "name", None) != "debug_far_static"
        ]

    def rehydrate_tamp_atoms(env):
        # Pickled Atom objects can carry process-local hash/cache state. Rebuild
        # them so cuTAMP's skeleton generator can match goals in this process.
        from cutamp.tamp_domain import all_tamp_fluents

        fluent_by_name = {fluent.name: fluent for fluent in all_tamp_fluents}

        def rehydrate_state(state):
            return frozenset(
                fluent_by_name[atom.name].ground(*atom.values) for atom in state
            )

        if hasattr(env, "initial_state"):
            env.initial_state = rehydrate_state(env.initial_state)
        if hasattr(env, "goal_state"):
            env.goal_state = rehydrate_state(env.goal_state)
        return env

    async def cached_run_perception(
        session,
        observation,
        task_instruction,
        save_dir,
        depth_estimator=None,
        gripper_mask=None,
        include_workspace=True,
        log_to_rerun=True,
    ):
        del session, observation, task_instruction, depth_estimator, gripper_mask
        del include_workspace, log_to_rerun
        save_dir = Path(save_dir)
        copy_cached_artifacts(save_dir)
        with env_path.open("rb") as f:
            env = rehydrate_tamp_atoms(dill.load(f))
        map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
        grasps = torch.load(grasps_path, weights_only=False, map_location=map_location)
        with metadata_path.open() as f:
            metadata = json.load(f)
        grounded_atoms = metadata.get("perception", {}).get("grounded_atoms", [])
        all_surfaces = surfaces_from_env(env)
        table_cuboid = all_surfaces[0] if all_surfaces else None
        processed_scene = ProcessedScene(
            table_cuboid=table_cuboid,
            object_meshes={obj.name: obj for obj in getattr(env, "movables", [])},
            object_pcds={},
            grasps=grasps,
        )
        print(f"YAM debug run: reused cached perception from {run_dir}", flush=True)
        return env, all_surfaces, processed_scene, grounded_atoms

    tiptop_h5.run_perception = cached_run_perception


def _install_yam_tiptop_config() -> None:
    """Force the TiPToP runtime config to the simulator YAM embodiment."""
    from omegaconf import OmegaConf
    from tiptop.config import tiptop_cfg

    cfg = tiptop_cfg()
    OmegaConf.update(cfg, "robot.type", "yam", merge=False)
    OmegaConf.update(cfg, "robot.dof", 6, merge=False)
    OmegaConf.update(cfg, "robot.q_home", list(YAM_HOME_QPOS), merge=False)
    OmegaConf.update(cfg, "robot.q_capture", list(YAM_HOME_QPOS), merge=False)
    if not OmegaConf.select(cfg, "robot.time_dilation_factor"):
        OmegaConf.update(cfg, "robot.time_dilation_factor", 0.2, merge=False, force_add=True)


def _rz(theta, *, device, dtype):
    import torch

    c = torch.cos(torch.tensor(theta, device=device, dtype=dtype))
    s = torch.sin(torch.tensor(theta, device=device, dtype=dtype))
    z = torch.zeros((), device=device, dtype=dtype)
    one = torch.ones((), device=device, dtype=dtype)
    return torch.stack(
        (
            torch.stack((c, -s, z)),
            torch.stack((s, c, z)),
            torch.stack((z, z, one)),
        )
    )


def _make_tool_from_ee(
    mode: str,
    ref,
    local_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """Return a candidate transform from grasp/tool frame to cuRobo EE frame."""
    import torch

    device = ref.device
    dtype = ref.dtype
    current = ref.clone()
    offset = torch.tensor(local_offset, device=device, dtype=dtype)

    def with_offset(transform):
        transform = transform.clone()
        transform[:3, 3] += offset
        return transform

    def from_rotation_and_finger_midpoint(rotation_rows, ee_from_finger):
        transform = torch.eye(4, device=device, dtype=dtype)
        rotation = torch.tensor(rotation_rows, device=device, dtype=dtype)
        finger = torch.tensor(ee_from_finger, device=device, dtype=dtype)
        transform[:3, :3] = rotation
        # cuTAMP uses world_from_ee = world_from_tool @ tool_from_ee.
        # Choose the translation so the measured YAM fingertip midpoint lands
        # at the virtual TiPToP grasp/tool origin.
        transform[:3, 3] = -(rotation @ finger)
        return transform

    mujoco_site = torch.eye(4, device=device, dtype=dtype)
    mujoco_site[:3, :3] = _rz(-torch.pi / 2, device=device, dtype=dtype)
    mujoco_site[:3, 3] = torch.tensor([0.0, 0.0, 0.1347], device=device, dtype=dtype)

    measured_grasp_site = torch.eye(4, device=device, dtype=dtype)
    measured_grasp_site[:3, :3] = torch.tensor(
        [
            [0.999811, 0.019455, 0.000001],
            [0.019455, -0.999811, 0.000002],
            [0.000001, -0.000002, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    measured_grasp_site[:3, 3] = torch.tensor(
        [-0.00701567, 0.00414988, -0.13591795],
        device=device,
        dtype=dtype,
    )
    measured_grasp_site_ee_above = measured_grasp_site.clone()
    measured_grasp_site_ee_above[:3, 3] = torch.tensor(
        [-0.00701567, 0.00414988, 0.13591795],
        device=device,
        dtype=dtype,
    )
    measured_grasp_site_ee_above_z_up = torch.eye(4, device=device, dtype=dtype)
    measured_grasp_site_ee_above_z_up[:3, 3] = torch.tensor(
        [-0.00701567, 0.00414988, 0.13591795],
        device=device,
        dtype=dtype,
    )
    measured_grasp_site_ee_above_z_up_yaw_pi = torch.eye(4, device=device, dtype=dtype)
    measured_grasp_site_ee_above_z_up_yaw_pi[:3, :3] = _rz(torch.pi, device=device, dtype=dtype)
    measured_grasp_site_ee_above_z_up_yaw_pi[:3, 3] = torch.tensor(
        [-0.00701567, 0.00414988, 0.13591795],
        device=device,
        dtype=dtype,
    )

    # Canonical YAM top-down TiPToP grasp frames. These preserve cuTAMP's
    # 4-DOF cuboid convention where world_from_grasp has +Z upward and only yaw
    # changes. The translations are derived from the measured MuJoCo
    # grasp_site<->cuRobo ee_link relation so the physical grasp_site origin
    # lands on the virtual TiPToP grasp origin while the ee_link stays above it.
    canonical_topdown_yaw_0 = torch.eye(4, device=device, dtype=dtype)
    canonical_topdown_yaw_0[:3, 3] = torch.tensor(
        [-0.00654240, -0.00370353, 0.13415721],
        device=device,
        dtype=dtype,
    )
    canonical_topdown_yaw_pi = torch.eye(4, device=device, dtype=dtype)
    canonical_topdown_yaw_pi[:3, :3] = _rz(torch.pi, device=device, dtype=dtype)
    canonical_topdown_yaw_pi[:3, 3] = torch.tensor(
        [0.00654240, 0.00370353, 0.13415721],
        device=device,
        dtype=dtype,
    )

    mujoco_grasp_site_calibrated = torch.eye(4, device=device, dtype=dtype)
    mujoco_grasp_site_calibrated[:3, :3] = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    mujoco_grasp_site_calibrated[:3, 3] = torch.tensor(
        [-0.00654240, 0.00370353, -0.13415721],
        device=device,
        dtype=dtype,
    )

    # Empirically reachable YAM EE orientation for grasps near the simulator
    # cube. Unlike the canonical top-down frames, this tilted wrist pose passes
    # cuRobo IK with self-collision enabled.
    yam_tilted_reachable = torch.eye(4, device=device, dtype=dtype)
    yam_tilted_reachable[:3, :3] = torch.tensor(
        [
            [-0.535512, -0.034585, 0.843819],
            [0.010009, 0.998831, 0.047290],
            [-0.844468, 0.033770, -0.534540],
        ],
        device=device,
        dtype=dtype,
    )
    yam_tilted_reachable[:3, 3] = torch.tensor(
        [-0.00654240, -0.00370353, 0.13415721],
        device=device,
        dtype=dtype,
    )
    yam_tilted_grasp_frame = yam_tilted_reachable.clone()
    yam_tilted_grasp_frame[:3, 3] = 0.0

    # Side-pinch candidates. In the YAM MJCF, link_6 local X is the gripper
    # opening/closing axis. These keep that axis horizontal and put link_6 local
    # Y vertical, so link_6 local Z approaches the object from the side instead
    # of pressing downward onto the cube. The fingertip midpoint is estimated
    # from the central inner fingertip geom pair and expressed in the cuRobo
    # EE/grasp_site frame, not the raw link_6 frame.
    side_pinch_ee_from_finger_midpoint = (0.017, 0.0, -0.0212)
    side_pinch_y_up = from_rotation_and_finger_midpoint(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ],
        side_pinch_ee_from_finger_midpoint,
    )
    side_pinch_y_down = from_rotation_and_finger_midpoint(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        side_pinch_ee_from_finger_midpoint,
    )
    # Calibrated from the actual MuJoCo fingertip pad centers, not from a hand
    # guessed offset. The pinch frame origin is the midpoint of the lf_down/rf_down
    # inner box pad centers, +X is the real finger open/close axis, and +Z is the
    # selected link_6 palm-side direction. These are T_pinch_ee constants for
    # cuTAMP's convention: world_from_ee = world_from_grasp @ tool_from_ee.
    yam_pinch_pad_y_up = torch.eye(4, device=device, dtype=dtype)
    yam_pinch_pad_y_up[:3, :3] = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    yam_pinch_pad_y_up[:3, 3] = torch.tensor(
        [-0.00370353, -0.11295721, 0.02354240],
        device=device,
        dtype=dtype,
    )
    yam_pinch_pad_y_down = torch.eye(4, device=device, dtype=dtype)
    yam_pinch_pad_y_down[:3, :3] = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        device=device,
        dtype=dtype,
    )
    yam_pinch_pad_y_down[:3, 3] = torch.tensor(
        [-0.00370353, 0.11295721, -0.02354240],
        device=device,
        dtype=dtype,
    )
    yam_robotiq_pinch_pad = torch.eye(4, device=device, dtype=dtype)
    yam_robotiq_pinch_pad[:3, :3] = torch.tensor(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        device=device,
        dtype=dtype,
    )
    yam_robotiq_pinch_pad[:3, 3] = torch.tensor(
        [0.00370353, 0.02354240, -0.11295721],
        device=device,
        dtype=dtype,
    )

    if mode == "current":
        return with_offset(current)
    if mode == "current-inverse":
        return with_offset(torch.linalg.inv(current))
    if mode == "mujoco-site":
        return with_offset(mujoco_site)
    if mode == "mujoco-site-inverse":
        return with_offset(torch.linalg.inv(mujoco_site))
    if mode == "measured-grasp-site":
        return with_offset(measured_grasp_site)
    if mode == "measured-grasp-site-ee-above":
        return with_offset(measured_grasp_site_ee_above)
    if mode == "measured-grasp-site-ee-above-z-up":
        return with_offset(measured_grasp_site_ee_above_z_up)
    if mode == "measured-grasp-site-ee-above-z-up-yaw-pi":
        return with_offset(measured_grasp_site_ee_above_z_up_yaw_pi)
    if mode == "canonical-topdown-yaw-0":
        return with_offset(canonical_topdown_yaw_0)
    if mode == "canonical-topdown-yaw-pi":
        return with_offset(canonical_topdown_yaw_pi)
    if mode == "mujoco-grasp-site-calibrated":
        return with_offset(mujoco_grasp_site_calibrated)
    if mode == "yam-tilted-reachable":
        return with_offset(yam_tilted_reachable)
    if mode == "yam-tilted-grasp-frame":
        return with_offset(yam_tilted_grasp_frame)
    if mode == "yam-side-pinch-y-up":
        return with_offset(side_pinch_y_up)
    if mode == "yam-side-pinch-y-down":
        return with_offset(side_pinch_y_down)
    if mode == "yam-pinch-pad-y-up":
        return with_offset(yam_pinch_pad_y_up)
    if mode == "yam-pinch-pad-y-down":
        return with_offset(yam_pinch_pad_y_down)
    if mode == "yam-robotiq-pinch-pad":
        return with_offset(yam_robotiq_pinch_pad)

    raise ValueError(f"Unknown tool-frame mode: {mode}")


def _retarget_gripper_spheres(gripper_spheres, source_tool_from_ee, target_tool_from_ee):
    """Express gripper spheres in the target TiPToP tool frame."""
    source_from_target = target_tool_from_ee @ source_tool_from_ee.inverse()
    retargeted = gripper_spheres.clone()
    retargeted[:, :3] = (
        gripper_spheres[:, :3] @ source_from_target[:3, :3].T
        + source_from_target[:3, 3]
    )
    return retargeted


def _install_yam_debug_patches(
    tool_frame_mode: str,
    m2t2_grasps: bool | None,
    tool_frame_local_offset: tuple[float, float, float],
) -> None:
    import cutamp.algorithm as cutamp_algorithm
    import cutamp.robots as cutamp_robots
    import tiptop.motion_planning as motion_planning
    import tiptop.planning as tiptop_planning

    tiptop_h5 = _tiptop_h5_module()

    original_yam_loader = cutamp_robots.robot_to_fns["yam"]["container"]

    def load_yam_container_debug(tensor_args):
        container = original_yam_loader(tensor_args)
        tool_from_ee = _make_tool_from_ee(
            tool_frame_mode,
            container.tool_from_ee,
            local_offset=tool_frame_local_offset,
        )
        gripper_spheres = _retarget_gripper_spheres(
            container.gripper_spheres,
            container.tool_from_ee,
            tool_from_ee,
        )
        return cutamp_robots.RobotContainer(
            name=container.name,
            kin_model=container.kin_model,
            joint_limits=container.joint_limits,
            gripper_spheres=gripper_spheres,
            tool_from_ee=tool_from_ee,
        )

    def load_robot_container_debug(robot: str, tensor_args):
        if robot == "yam":
            return load_yam_container_debug(tensor_args)
        return original_load_robot_container(robot, tensor_args)

    original_load_robot_container = cutamp_robots.load_robot_container
    cutamp_robots.load_yam_container = load_yam_container_debug
    cutamp_robots.robot_to_fns["yam"]["container"] = load_yam_container_debug
    cutamp_robots.load_robot_container = load_robot_container_debug
    cutamp_algorithm.load_robot_container = load_robot_container_debug
    motion_planning.load_yam_container = load_yam_container_debug

    if m2t2_grasps is not None:
        original_build_tamp_config = tiptop_planning.build_tamp_config

        def build_tamp_config_debug(*args, **kwargs):
            config = original_build_tamp_config(*args, **kwargs)
            return config.__class__(
                **{
                    **config.__dict__,
                    "m2t2_grasps": m2t2_grasps,
                }
            )

        tiptop_planning.build_tamp_config = build_tamp_config_debug
        tiptop_h5.build_tamp_config = build_tamp_config_debug

        if m2t2_grasps is False:
            original_run_planning = tiptop_h5.run_planning

            def run_planning_debug(
                env,
                config,
                q_init,
                ik_solver,
                grasps,
                motion_gen,
                all_surfaces,
                experiment_dir=None,
            ):
                return original_run_planning(
                    env,
                    config,
                    q_init,
                    ik_solver,
                    None,
                    motion_gen,
                    all_surfaces,
                    experiment_dir=experiment_dir,
                )

            tiptop_h5.run_planning = run_planning_debug


def _install_yam_q_conversion_patch() -> None:
    from dataclasses import replace

    tiptop_h5 = _tiptop_h5_module()

    original_load_h5_observation = tiptop_h5.load_h5_observation

    def load_h5_observation_yam_mujoco_q(h5_path):
        observation = original_load_h5_observation(h5_path)
        q_init = _mujoco_q_to_curobo(observation.q_init).astype(observation.q_init.dtype, copy=False)
        print(
            "YAM sim q convention: converted MuJoCo q_init to cuRobo q_init "
            f"with signs={YAM_MUJOCO_TO_CUROBO_Q_SIGNS}"
        )
        return replace(observation, q_init=q_init)

    tiptop_h5.load_h5_observation = load_h5_observation_yam_mujoco_q


def _install_constraint_debug() -> None:
    import cutamp.constraint_checker as constraint_checker_mod

    calls = {"n": 0}

    def get_mask_debug(self, cost_dict, verbose=True):
        calls["n"] += 1
        overall_mask = None
        rows = []
        for cost_type, cost_info in cost_dict.items():
            if cost_info["type"] != "constraint":
                continue
            for name, values in cost_info["values"].items():
                tol = self._get_tol(cost_type, name)
                mask_full = values <= tol
                mask = mask_full.all(dim=1) if mask_full.ndim == 2 else mask_full
                remaining = mask if overall_mask is None else (overall_mask & mask)
                worst = values.max(dim=1).values if values.ndim == 2 else values
                rows.append(
                    (
                        cost_type,
                        name,
                        int(mask.sum().item()),
                        int(remaining.sum().item()),
                        float(worst.min().detach().cpu()),
                        float(worst.median().detach().cpu()),
                        float(worst.max().detach().cpu()),
                        float(tol),
                        int(mask.shape[0]),
                    )
                )
                overall_mask = remaining

        if rows and (calls["n"] <= 3 or calls["n"] % 100 == 0):
            print(f"CONSTRAINT_DEBUG call={calls['n']}")
            for cost_type, name, passed, remaining, vmin, vmed, vmax, tol, total in rows:
                print(
                    f"  {cost_type}/{name}: pass={passed}/{total} remaining={remaining} "
                    f"tol={tol:.6g} min={vmin:.6g} median={vmed:.6g} max={vmax:.6g}"
                )

        assert overall_mask is not None, "No constraints found in cost dict"
        return overall_mask

    constraint_checker_mod.ConstraintChecker.get_mask = get_mask_debug


def _install_pose_debug(
    relax_approach_orientation: bool = False,
    ignore_pick_target_collision: bool = False,
) -> None:
    """Log final grasp and approach poses tried by cuRobo."""
    import torch
    import cutamp.motion_solver as motion_solver
    from curobo.rollout.cost.pose_cost import PoseCostMetric
    from curobo.types.math import Pose
    from curobo.wrap.reacher.motion_gen import MotionGen

    def pose_summary(mat):
        arr = mat.detach().cpu()
        xyz = arr[:3, 3].numpy().tolist()
        x_axis = arr[:3, 0].numpy().tolist()
        y_axis = arr[:3, 1].numpy().tolist()
        z_axis = arr[:3, 2].numpy().tolist()
        return {
            "xyz": [round(float(v), 6) for v in xyz],
            "x_axis": [round(float(v), 6) for v in x_axis],
            "y_axis": [round(float(v), 6) for v in y_axis],
            "z_axis": [round(float(v), 6) for v in z_axis],
        }

    def pose_arg_summary(pose):
        try:
            mat = pose.get_matrix()
            if mat.ndim == 3:
                mat = mat[0]
            return pose_summary(mat)
        except Exception as exc:  # pragma: no cover - debug-only path.
            return {"unprintable_pose": type(exc).__name__}

    def result_success(result) -> bool:
        success = getattr(result, "success", False)
        if torch.is_tensor(success):
            return bool(success.detach().any().cpu().item())
        return bool(success)

    original_plan_single = MotionGen.plan_single
    plan_single_calls = {"n": 0}

    def plan_single_debug(self, start_state, goal_pose, *args, **kwargs):
        plan_single_calls["n"] += 1
        call_idx = plan_single_calls["n"]
        try:
            start_q = start_state.position.detach().cpu().reshape(-1).numpy().tolist()
            start_q = [round(float(v), 6) for v in start_q]
        except Exception:
            start_q = "<unprintable>"
        print(
            f"POSE_DEBUG plan_single call={call_idx} "
            f"start_q={start_q} target={pose_arg_summary(goal_pose)}",
            flush=True,
        )
        result = original_plan_single(self, start_state, goal_pose, *args, **kwargs)
        print(
            f"POSE_DEBUG plan_single call={call_idx} "
            f"success={result_success(result)} status={getattr(result, 'status', None)}",
            flush=True,
        )
        return result

    MotionGen.plan_single = plan_single_debug

    def try_approach_offsets_debug(
        *,
        motion_gen,
        start_js,
        world_from_ee,
        approach_offsets,
        plan_config,
        op_name,
        stage_name,
    ):
        print(f"POSE_DEBUG {stage_name} {op_name} final_ee={pose_summary(world_from_ee)}")
        if ignore_pick_target_collision and stage_name == "Pick approach" and op_name.startswith("Pick("):
            target_obj = op_name.split("(", 1)[1].split(",", 1)[0].strip()
            motion_gen.world_coll_checker.enable_obstacle(enable=False, name=target_obj)
            print(f"POSE_DEBUG {stage_name} {op_name} disabled_target_obstacle={target_obj}")

        last_result = None
        world_from_approaches = world_from_ee @ approach_offsets
        attempt_plan_config = plan_config
        if relax_approach_orientation and "approach" in stage_name.lower():
            attempt_plan_config = plan_config.clone()
            attempt_plan_config.pose_cost_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.06,
                linear_axis=2,
                tstep_fraction=0.8,
                project_to_goal_frame=True,
                tensor_args=motion_gen.tensor_args,
            )
            print(f"POSE_DEBUG {stage_name} {op_name} relax_approach_orientation=True")

        for app_idx, world_from_approach in enumerate(world_from_approaches):
            offset_xyz = approach_offsets[app_idx, :3, 3].detach().cpu().numpy().tolist()
            result = motion_gen.plan_single(
                start_js,
                Pose.from_matrix(world_from_approach),
                attempt_plan_config,
            )
            last_result = result

            success = motion_solver._is_success(result)
            print(
                f"POSE_DEBUG {stage_name} {op_name} "
                f"attempt={app_idx + 1}/{len(world_from_approaches)} "
                f"offset={[round(float(v), 6) for v in offset_xyz]} "
                f"approach={pose_summary(world_from_approach)} "
                f"success={success} status={result.status}"
            )
            if success:
                return result

        status = None if last_result is None else last_result.status
        raise motion_solver.MotionPlanningError(
            f"Failed {stage_name} for {op_name}. "
            f"Tried {len(approach_offsets)} offsets. Last status: {status}"
        )

    motion_solver._try_approach_offsets = try_approach_offsets_debug


def _install_yam_sim_bootstrap(
    rot_tol: float,
    joint_space_fallback: bool,
    ignore_robot_world_collision: bool,
    ignore_robot_movable_collision: bool,
    robot_movable_collision_tol: float | None,
    drop_static_world: bool,
    num_resampling_attempts: int | None,
    max_motion_refine_attempts: int | None,
) -> None:
    """Patch tiptop_h5.run_planning for simulator bootstrap runs."""
    import time

    import torch
    from curobo.geom.types import Cuboid
    from cutamp.algorithm import run_cutamp, setup_cutamp
    from cutamp.envs import TAMPEnvironment
    from cutamp.constraint_checker import ConstraintChecker
    from cutamp.cost_function import CostFunction
    from cutamp.cost_reduction import CostReducer
    from cutamp.particle_initialization import ParticleInitializer
    from cutamp.rollout import RolloutFunction
    from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol
    from cutamp.tamp_domain import Pick, all_tamp_operators
    from cutamp.task_planning import task_plan_generator

    tiptop_h5 = _tiptop_h5_module()

    original_particle_initializer_call = ParticleInitializer.__call__

    def particle_initializer_without_none_metadata(self, *args, **kwargs):
        particles = original_particle_initializer_call(self, *args, **kwargs)
        if particles is None:
            return None
        # Heuristic grasps do not have M2T2 confidence scores. cuTAMP already
        # ignores None metadata in one ranking path, but the sampling baseline
        # assumes every particle entry is indexable when saving best_particle.
        return {key: value for key, value in particles.items() if value is not None}

    ParticleInitializer.__call__ = particle_initializer_without_none_metadata

    class SimpleJointPlan:
        def __init__(self, position: torch.Tensor, dt: float):
            self.position = position
            vel = torch.zeros_like(position)
            if position.shape[0] > 1:
                vel[1:] = (position[1:] - position[:-1]) / dt
                vel[0] = vel[1]
            self.velocity = vel

    def _constraint_tolerances(all_surfaces):
        constraint_to_tol = default_constraint_to_tol.copy()
        constraint_to_tol["Collision"] = constraint_to_tol["Collision"].copy()
        if ignore_robot_world_collision:
            constraint_to_tol["Collision"]["robot_to_world"] = 1e6
        if ignore_robot_movable_collision:
            constraint_to_tol["Collision"]["robot_to_movables"] = 1e6
        elif robot_movable_collision_tol is not None:
            constraint_to_tol["Collision"]["robot_to_movables"] = robot_movable_collision_tol
        constraint_to_tol["KinematicConstraint"] = constraint_to_tol["KinematicConstraint"].copy()
        constraint_to_tol["KinematicConstraint"]["rot_err"] = rot_tol
        constraint_to_mult = default_constraint_to_mult.copy()
        for surface in all_surfaces:
            constraint_to_tol["StablePlacement"][f"{surface.name}_in_xy"] = 1e-2
            constraint_to_tol["StablePlacement"][f"{surface.name}_support"] = 1e-2
            constraint_to_mult["StablePlacement"][f"{surface.name}_support"] = 1.0
        return constraint_to_tol, constraint_to_mult

    def _without_static_world(env):
        if not drop_static_world:
            return env
        if (
            len(env.statics) == 1
            and getattr(env.statics[0], "name", None) == "debug_far_static"
        ):
            return env
        type_to_objects = {
            obj_type: list(objects)
            for obj_type, objects in env.type_to_objects.items()
            if obj_type != "Surface"
        }
        dummy_static = Cuboid(
            name="debug_far_static",
            dims=[0.001, 0.001, 0.001],
            pose=[100.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.0],
            color=[128, 128, 128],
        )
        return TAMPEnvironment(
            name=f"{env.name}_no_static_world",
            movables=list(env.movables),
            statics=[dummy_static],
            type_to_objects=type_to_objects,
            goal_state=env.goal_state,
        )

    def _joint_space_fallback(env, config, q_init, ik_solver, all_surfaces):
        env = _without_static_world(env)
        fallback_config = config.__class__(
            **{
                **config.__dict__,
                "m2t2_grasps": False,
                "approach": "sampling",
                "cache_subgraphs": False,
                "curobo_plan": False,
                "enable_experiment_logging": False,
                "num_resampling_attempts": max(
                    config.num_resampling_attempts,
                    num_resampling_attempts or 40,
                ),
            }
        )
        constraint_to_tol, constraint_to_mult = _constraint_tolerances(all_surfaces)
        checker = ConstraintChecker(constraint_to_tol)
        reducer = CostReducer(constraint_to_mult)
        _exp_logger, _visualizer, timer, world = setup_cutamp(
            env,
            fallback_config,
            q_init=q_init,
            ik_solver=ik_solver,
            experiment_dir=None,
        )
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_tamp_operators,
            explored_state_check=fallback_config.explored_state_check,
        )
        plan_skeleton = next(plan_gen)
        initializer = ParticleInitializer(world, fallback_config, grasps=None)
        rollout_fn = RolloutFunction(plan_skeleton, world, fallback_config)
        cost_fn = CostFunction(plan_skeleton, world, fallback_config)

        best_idx = None
        best_particles = None
        best_cost = None
        max_attempts = fallback_config.num_resampling_attempts + 1
        for attempt in range(max_attempts):
            particles = initializer(plan_skeleton)
            if particles is None:
                continue
            with torch.no_grad():
                rollout = rollout_fn(particles)
                cost_dict = cost_fn(rollout)
                mask = checker.get_mask(cost_dict, verbose=False)
                if not mask.any():
                    continue
                hard_costs = reducer.hard_costs(cost_dict)
                satisfying_indices = torch.arange(mask.shape[0], device=mask.device)[mask]
                local_best = hard_costs[mask].argmin()
                idx = satisfying_indices[local_best]
                cost = hard_costs[idx]
                if best_idx is None or cost < best_cost:
                    best_idx = idx
                    best_particles = particles
                    best_cost = cost
                    break

        if best_idx is None or best_particles is None:
            return None

        q_key = None
        pick_label = "Pick"
        move_label = "MoveFree"
        for op in plan_skeleton:
            if op.operator.name == Pick.name:
                _obj, _grasp, q_key = op.values
                pick_label = op.name
            else:
                move_label = op.name
        if q_key is None:
            return None

        q0 = world.q_init.detach()
        q1 = best_particles[q_key][best_idx].detach()
        steps = 80
        alpha = torch.linspace(0.0, 1.0, steps, device=q0.device, dtype=q0.dtype)[:, None]
        positions = q0[None, :] * (1.0 - alpha) + q1[None, :] * alpha
        dt = 0.04
        return [
            {"type": "trajectory", "label": move_label, "plan": SimpleJointPlan(positions, dt), "dt": dt},
            {"type": "gripper", "label": pick_label, "action": "close"},
        ]

    def run_planning_yam_sim(env, config, q_init, ik_solver, grasps, motion_gen, all_surfaces, experiment_dir=None):
        env = _without_static_world(env)
        sim_config = config.__class__(
            **{
                **config.__dict__,
                "m2t2_grasps": False,
                "approach": "sampling",
                "cache_subgraphs": False,
                "num_resampling_attempts": max(
                    config.num_resampling_attempts,
                    num_resampling_attempts or 20,
                ),
                "max_motion_refine_attempts": max(
                    config.max_motion_refine_attempts or 0,
                    max_motion_refine_attempts or 16,
                ),
            }
        )
        constraint_to_tol, constraint_to_mult = _constraint_tolerances(all_surfaces)
        cost_reducer = CostReducer(constraint_to_mult)
        constraint_checker = ConstraintChecker(constraint_to_tol)

        start = time.perf_counter()
        cutamp_plan, _num_satisfying, failure_reason = run_cutamp(
            env,
            sim_config,
            cost_reducer,
            constraint_checker,
            q_init=q_init,
            ik_solver=ik_solver,
            grasps=None,
            motion_gen=motion_gen,
            experiment_dir=experiment_dir,
        )
        elapsed = time.perf_counter() - start
        if cutamp_plan is not None:
            return cutamp_plan, elapsed, failure_reason

        if joint_space_fallback:
            fallback = _joint_space_fallback(env, sim_config, q_init, ik_solver, all_surfaces)
            if fallback is not None:
                print(
                    "YAM sim bootstrap: cuRobo refinement failed, "
                    "saved a direct joint-space fallback plan for MuJoCo testing."
                )
                return fallback, elapsed, None

        return None, elapsed, failure_reason

    tiptop_h5.run_planning = run_planning_yam_sim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", required=True)
    parser.add_argument("--task-instruction", required=True)
    parser.add_argument("--output-dir", default="/tmp/tiptop_yam_debug")
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--opt-steps-per-skeleton", type=int, default=600)
    parser.add_argument("--max-planning-time", type=float, default=60.0)
    parser.add_argument(
        "--tool-frame-mode",
        choices=(
            "current",
            "current-inverse",
            "mujoco-site",
            "mujoco-site-inverse",
            "measured-grasp-site",
            "measured-grasp-site-ee-above",
            "measured-grasp-site-ee-above-z-up",
            "measured-grasp-site-ee-above-z-up-yaw-pi",
            "canonical-topdown-yaw-0",
            "canonical-topdown-yaw-pi",
            "mujoco-grasp-site-calibrated",
            "yam-tilted-reachable",
            "yam-tilted-grasp-frame",
            "yam-side-pinch-y-up",
            "yam-side-pinch-y-down",
            "yam-pinch-pad-y-up",
            "yam-pinch-pad-y-down",
            "yam-robotiq-pinch-pad",
        ),
        default="current",
    )
    parser.add_argument("--disable-m2t2-grasps", action="store_true")
    parser.add_argument(
        "--tool-frame-local-offset",
        type=_parse_vec3,
        default=(0.0, 0.0, 0.0),
        help="Extra x,y,z translation added to the selected tool_from_ee frame.",
    )
    parser.add_argument("--rr-spawn", action="store_true")
    parser.add_argument("--constraint-debug", action="store_true")
    parser.add_argument("--yam-sim-bootstrap", action="store_true")
    parser.add_argument("--yam-rot-tol", type=float, default=0.8)
    parser.add_argument("--joint-space-fallback", action="store_true")
    parser.add_argument("--ignore-robot-world-collision", action="store_true")
    parser.add_argument("--ignore-robot-movable-collision", action="store_true")
    parser.add_argument(
        "--robot-movable-collision-tol",
        type=float,
        default=None,
        help=(
            "Override cuTAMP robot_to_movables collision tolerance. Useful for "
            "allowing small fingertip contact/penetration without fully ignoring "
            "robot-object collision."
        ),
    )
    parser.add_argument("--drop-static-world", action="store_true")
    parser.add_argument("--yam-num-resampling-attempts", type=int, default=None)
    parser.add_argument("--yam-max-motion-refine-attempts", type=int, default=None)
    parser.add_argument("--pose-debug", action="store_true")
    parser.add_argument("--relax-approach-orientation", action="store_true")
    parser.add_argument("--ignore-pick-target-collision", action="store_true")
    parser.add_argument(
        "--reuse-perception-from",
        type=Path,
        help="Saved TiPToP run directory whose perception/cutamp_env.pkl and grasps.pt should be reused.",
    )
    parser.add_argument("--no-yam-q-sign-conversion", action="store_true")
    args = parser.parse_args()

    _prepare_tiptop_package_dir()
    try:
        _check_tiptop_dependencies()
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from None

    _install_yam_tiptop_config()
    _install_gemini_json_repair_patch()
    _install_yam_debug_patches(
        tool_frame_mode=args.tool_frame_mode,
        m2t2_grasps=False if args.disable_m2t2_grasps else None,
        tool_frame_local_offset=args.tool_frame_local_offset,
    )
    if not args.no_yam_q_sign_conversion:
        _install_yam_q_conversion_patch()
    if args.reuse_perception_from is not None:
        _install_cached_perception_patch(args.reuse_perception_from)
    if args.constraint_debug:
        _install_constraint_debug()
    if args.pose_debug or args.ignore_pick_target_collision:
        _install_pose_debug(
            relax_approach_orientation=args.relax_approach_orientation,
            ignore_pick_target_collision=args.ignore_pick_target_collision,
        )
    if args.yam_sim_bootstrap:
        _install_yam_sim_bootstrap(
            rot_tol=args.yam_rot_tol,
            joint_space_fallback=args.joint_space_fallback,
            ignore_robot_world_collision=args.ignore_robot_world_collision,
            ignore_robot_movable_collision=args.ignore_robot_movable_collision,
            robot_movable_collision_tol=args.robot_movable_collision_tol,
            drop_static_world=args.drop_static_world,
            num_resampling_attempts=args.yam_num_resampling_attempts,
            max_motion_refine_attempts=args.yam_max_motion_refine_attempts,
        )

    run_tiptop_h5 = _tiptop_h5_module().run_tiptop_h5

    print(
        "YAM debug run: "
        f"tool_frame_mode={args.tool_frame_mode}, "
        f"tool_frame_local_offset={args.tool_frame_local_offset}, "
        f"robot_movable_collision_tol={args.robot_movable_collision_tol}, "
        f"m2t2_grasps={'false' if args.disable_m2t2_grasps else 'tiptop-default'}, "
        f"q_sign_conversion={'false' if args.no_yam_q_sign_conversion else YAM_MUJOCO_TO_CUROBO_Q_SIGNS}"
    )
    run_tiptop_h5(
        h5_path=args.h5_path,
        task_instruction=args.task_instruction,
        output_dir=args.output_dir,
        max_planning_time=args.max_planning_time,
        opt_steps_per_skeleton=args.opt_steps_per_skeleton,
        num_particles=args.num_particles,
        cutamp_visualize=False,
        rr_spawn=args.rr_spawn,
    )


if __name__ == "__main__":
    main()
