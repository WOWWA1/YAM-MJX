#!/usr/bin/env python3
"""Planning-only TiPToP/YAM debug runner from cached perception outputs.

This is simulator/planner-only. It loads a previously saved TiPToP run directory
containing perception/cutamp_env.pkl and metadata.json, then reruns cuTAMP with
the YAM debug patches without calling Gemini, SAM, M2T2, or robot hardware.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import dill
import numpy as np
import torch

from run_tiptop_h5_yam_debug import (
    YAM_MUJOCO_TO_CUROBO_Q_SIGNS,
    _check_tiptop_dependencies,
    _install_constraint_debug,
    _install_pose_debug,
    _install_yam_debug_patches,
    _install_yam_sim_bootstrap,
    _install_yam_tiptop_config,
    _mujoco_q_to_curobo,
    _prepare_tiptop_package_dir,
    _tiptop_h5_module,
)


def _load_q_at_capture(run_dir: Path) -> np.ndarray:
    metadata_path = run_dir / "metadata.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return np.asarray(metadata["observation"]["q_at_capture"], dtype=np.float32)


def _load_cached_env(run_dir: Path):
    env_path = run_dir / "perception" / "cutamp_env.pkl"
    with env_path.open("rb") as f:
        env = dill.load(f)

    # Pickled Atom objects carry a cached Python hash from the process that
    # created them. Rebuild goal atoms so set membership works in this process.
    from cutamp.tamp_domain import all_tamp_fluents

    fluent_by_name = {fluent.name: fluent for fluent in all_tamp_fluents}
    env.goal_state = frozenset(
        fluent_by_name[atom.name].ground(*atom.values) for atom in env.goal_state
    )
    return env


def _load_cached_grasps(run_dir: Path):
    grasps_path = run_dir / "perception" / "grasps.pt"
    if not grasps_path.exists():
        return None
    return torch.load(grasps_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached-run-dir", required=True)
    parser.add_argument("--output-dir", default="/tmp/tiptop_yam_cached_debug")
    parser.add_argument("--num-particles", type=int, default=256)
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
        ),
        default="canonical-topdown-yaw-pi",
    )
    parser.add_argument(
        "--tool-frame-local-offset",
        type=lambda s: tuple(float(part.strip()) for part in s.split(",")),
        default=(0.0, 0.0, 0.0),
        help="Extra x,y,z translation added to the selected tool_from_ee frame.",
    )
    parser.add_argument("--disable-m2t2-grasps", action="store_true")
    parser.add_argument("--constraint-debug", action="store_true")
    parser.add_argument("--pose-debug", action="store_true")
    parser.add_argument("--relax-approach-orientation", action="store_true")
    parser.add_argument("--ignore-pick-target-collision", action="store_true")
    parser.add_argument("--yam-sim-bootstrap", action="store_true")
    parser.add_argument("--yam-rot-tol", type=float, default=0.8)
    parser.add_argument("--joint-space-fallback", action="store_true")
    parser.add_argument("--ignore-robot-world-collision", action="store_true")
    parser.add_argument("--drop-static-world", action="store_true")
    parser.add_argument("--use-env-surfaces", action="store_true")
    parser.add_argument("--no-yam-q-sign-conversion", action="store_true")
    args = parser.parse_args()

    cached_run_dir = Path(args.cached_run_dir)
    save_dir = Path(args.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    _prepare_tiptop_package_dir()
    _check_tiptop_dependencies()
    _install_yam_tiptop_config()

    _install_yam_debug_patches(
        tool_frame_mode=args.tool_frame_mode,
        m2t2_grasps=False if args.disable_m2t2_grasps else None,
        tool_frame_local_offset=args.tool_frame_local_offset,
    )
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

    from tiptop.config import tiptop_cfg
    from tiptop.motion_planning import build_curobo_solvers
    from tiptop.planning import build_tamp_config, save_tiptop_plan, serialize_plan
    from tiptop.utils import check_cutamp_version, print_tiptop_banner, setup_logging

    run_planning = _tiptop_h5_module().run_planning

    print_tiptop_banner()
    check_cutamp_version()
    setup_logging()

    cfg = tiptop_cfg()
    config = build_tamp_config(
        num_particles=args.num_particles,
        max_planning_time=args.max_planning_time,
        opt_steps=args.opt_steps_per_skeleton,
        robot_type=cfg.robot.type,
        time_dilation_factor=cfg.robot.time_dilation_factor,
        collision_activation_distance=0.0,
        enable_visualizer=False,
    )

    q_init = _load_q_at_capture(cached_run_dir)
    if not args.no_yam_q_sign_conversion:
        q_init = _mujoco_q_to_curobo(q_init).astype(q_init.dtype, copy=False)
        print(
            "YAM sim q convention: converted cached MuJoCo q_at_capture to cuRobo q_init "
            f"with signs={YAM_MUJOCO_TO_CUROBO_Q_SIGNS}"
        )
    env = _load_cached_env(cached_run_dir)
    grasps = None if args.disable_m2t2_grasps else _load_cached_grasps(cached_run_dir)
    all_surfaces = list(env.type_to_objects.get("Surface", [])) if args.use_env_surfaces else []

    ik_solver, motion_gen, _ = build_curobo_solvers(
        args.num_particles,
        config.coll_n_spheres,
        include_workspace=False,
    )

    print(
        "YAM cached debug run: "
        f"cached_run_dir={cached_run_dir}, "
        f"tool_frame_mode={args.tool_frame_mode}, "
        f"m2t2_grasps={'false' if args.disable_m2t2_grasps else 'cached'}, "
        f"q_sign_conversion={'false' if args.no_yam_q_sign_conversion else YAM_MUJOCO_TO_CUROBO_Q_SIGNS}"
    )

    start = time.perf_counter()
    cutamp_plan, planning_duration, failure_reason = run_planning(
        env,
        config,
        q_init,
        ik_solver,
        grasps,
        motion_gen,
        all_surfaces,
        experiment_dir=save_dir / "cutamp",
    )
    wall_duration = time.perf_counter() - start

    if cutamp_plan is not None:
        plan_path = save_dir / "tiptop_plan.json"
        save_tiptop_plan(serialize_plan(cutamp_plan, q_init), plan_path)
        print(f"YAM cached debug: saved TiPToP plan to {plan_path}")
    else:
        print(f"YAM cached debug: no plan found: {failure_reason}")

    metadata = {
        "cached_run_dir": str(cached_run_dir),
        "planning_success": cutamp_plan is not None,
        "planning_failure_reason": failure_reason,
        "planning_duration": planning_duration,
        "wall_duration": wall_duration,
        "q_at_capture": q_init.tolist(),
        "tool_frame_mode": args.tool_frame_mode,
        "q_sign_conversion": None if args.no_yam_q_sign_conversion else YAM_MUJOCO_TO_CUROBO_Q_SIGNS,
    }
    with (save_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"YAM cached debug: saved outputs to {save_dir}")


if __name__ == "__main__":
    main()
