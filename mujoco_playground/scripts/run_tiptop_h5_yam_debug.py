#!/usr/bin/env python3
"""Simulator-only TiPToP H5 runner for testing YAM tool-frame variants.

This wrapper does not talk to the robot daemon or execute a real arm. It reuses
TiPToP's offline H5 path and monkeypatches YAM planning settings in memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys


TIPTOP_PACKAGE_DIR = os.environ.get("TIPTOP_PACKAGE_DIR")
YAM_MUJOCO_TO_CUROBO_Q_SIGNS = (1.0, 1.0, 1.0, 1.0, 1.0, -1.0)


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
    if not TIPTOP_PACKAGE_DIR:
        return

    tiptop_package_dir = Path(TIPTOP_PACKAGE_DIR)
    if not tiptop_package_dir.exists():
        raise FileNotFoundError(
            f"TIPTOP_PACKAGE_DIR points to a missing path: {tiptop_package_dir}"
        )

    if (tiptop_package_dir / "__init__.py").exists():
        sys.path.insert(0, str(tiptop_package_dir.parent))
    else:
        sys.path.insert(0, str(tiptop_package_dir))
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


def _make_tool_from_ee(mode: str, ref):
    """Return a candidate transform from grasp/tool frame to cuRobo EE frame."""
    import torch

    device = ref.device
    dtype = ref.dtype
    current = ref.clone()

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

    if mode == "current":
        return current
    if mode == "current-inverse":
        return torch.linalg.inv(current)
    if mode == "mujoco-site":
        return mujoco_site
    if mode == "mujoco-site-inverse":
        return torch.linalg.inv(mujoco_site)
    if mode == "measured-grasp-site":
        return measured_grasp_site
    if mode == "measured-grasp-site-ee-above":
        return measured_grasp_site_ee_above
    if mode == "measured-grasp-site-ee-above-z-up":
        return measured_grasp_site_ee_above_z_up
    if mode == "measured-grasp-site-ee-above-z-up-yaw-pi":
        return measured_grasp_site_ee_above_z_up_yaw_pi
    if mode == "canonical-topdown-yaw-0":
        return canonical_topdown_yaw_0
    if mode == "canonical-topdown-yaw-pi":
        return canonical_topdown_yaw_pi
    if mode == "mujoco-grasp-site-calibrated":
        return mujoco_grasp_site_calibrated

    raise ValueError(f"Unknown tool-frame mode: {mode}")


def _install_yam_debug_patches(tool_frame_mode: str, m2t2_grasps: bool | None) -> None:
    import cutamp.algorithm as cutamp_algorithm
    import cutamp.robots as cutamp_robots
    import tiptop.motion_planning as motion_planning
    import tiptop.planning as tiptop_planning
    import tiptop.tiptop_h5 as tiptop_h5

    original_yam_loader = cutamp_robots.robot_to_fns["yam"]["container"]

    def load_yam_container_debug(tensor_args):
        container = original_yam_loader(tensor_args)
        tool_from_ee = _make_tool_from_ee(tool_frame_mode, container.tool_from_ee)
        return cutamp_robots.RobotContainer(
            name=container.name,
            kin_model=container.kin_model,
            joint_limits=container.joint_limits,
            gripper_spheres=container.gripper_spheres,
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

    import tiptop.tiptop_h5 as tiptop_h5

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
    drop_static_world: bool,
) -> None:
    """Patch tiptop_h5.run_planning for simulator bootstrap runs."""
    import time

    import torch
    import tiptop.tiptop_h5 as tiptop_h5
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
                "num_resampling_attempts": max(config.num_resampling_attempts, 40),
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
                "num_resampling_attempts": max(config.num_resampling_attempts, 20),
                "max_motion_refine_attempts": max(config.max_motion_refine_attempts or 0, 16),
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
        ),
        default="canonical-topdown-yaw-pi",
    )
    parser.add_argument("--disable-m2t2-grasps", action="store_true")
    parser.add_argument("--rr-spawn", action="store_true")
    parser.add_argument("--constraint-debug", action="store_true")
    parser.add_argument("--yam-sim-bootstrap", action="store_true")
    parser.add_argument("--yam-rot-tol", type=float, default=0.8)
    parser.add_argument("--joint-space-fallback", action="store_true")
    parser.add_argument("--ignore-robot-world-collision", action="store_true")
    parser.add_argument("--drop-static-world", action="store_true")
    parser.add_argument("--pose-debug", action="store_true")
    parser.add_argument("--relax-approach-orientation", action="store_true")
    parser.add_argument("--ignore-pick-target-collision", action="store_true")
    parser.add_argument("--no-yam-q-sign-conversion", action="store_true")
    args = parser.parse_args()

    _prepare_tiptop_package_dir()
    try:
        _check_tiptop_dependencies()
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from None

    _install_yam_debug_patches(
        tool_frame_mode=args.tool_frame_mode,
        m2t2_grasps=False if args.disable_m2t2_grasps else None,
    )
    if not args.no_yam_q_sign_conversion:
        _install_yam_q_conversion_patch()
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
            drop_static_world=args.drop_static_world,
        )

    from tiptop.tiptop_h5 import run_tiptop_h5

    print(
        "YAM debug run: "
        f"tool_frame_mode={args.tool_frame_mode}, "
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
