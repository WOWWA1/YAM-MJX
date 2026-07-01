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

    if mode == "current":
        return current
    if mode == "current-inverse":
        return torch.linalg.inv(current)
    if mode == "mujoco-site":
        return mujoco_site
    if mode == "mujoco-site-inverse":
        return torch.linalg.inv(mujoco_site)

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


def _install_yam_sim_bootstrap(rot_tol: float, joint_space_fallback: bool) -> None:
    """Patch tiptop_h5.run_planning for simulator bootstrap runs."""
    import time

    import torch
    import tiptop.tiptop_h5 as tiptop_h5
    from cutamp.algorithm import run_cutamp, setup_cutamp
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
        constraint_to_tol["KinematicConstraint"] = constraint_to_tol["KinematicConstraint"].copy()
        constraint_to_tol["KinematicConstraint"]["rot_err"] = rot_tol
        constraint_to_mult = default_constraint_to_mult.copy()
        for surface in all_surfaces:
            constraint_to_tol["StablePlacement"][f"{surface.name}_in_xy"] = 1e-2
            constraint_to_tol["StablePlacement"][f"{surface.name}_support"] = 1e-2
            constraint_to_mult["StablePlacement"][f"{surface.name}_support"] = 1.0
        return constraint_to_tol, constraint_to_mult

    def _joint_space_fallback(env, config, q_init, ik_solver, all_surfaces):
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
        choices=("current", "current-inverse", "mujoco-site", "mujoco-site-inverse"),
        default="current",
    )
    parser.add_argument("--disable-m2t2-grasps", action="store_true")
    parser.add_argument("--rr-spawn", action="store_true")
    parser.add_argument("--constraint-debug", action="store_true")
    parser.add_argument("--yam-sim-bootstrap", action="store_true")
    parser.add_argument("--yam-rot-tol", type=float, default=0.8)
    parser.add_argument("--joint-space-fallback", action="store_true")
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
    if args.constraint_debug:
        _install_constraint_debug()
    if args.yam_sim_bootstrap:
        _install_yam_sim_bootstrap(
            rot_tol=args.yam_rot_tol,
            joint_space_fallback=args.joint_space_fallback,
        )

    from tiptop.tiptop_h5 import run_tiptop_h5

    print(
        "YAM debug run: "
        f"tool_frame_mode={args.tool_frame_mode}, "
        f"m2t2_grasps={'false' if args.disable_m2t2_grasps else 'tiptop-default'}"
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
