#!/usr/bin/env python3
"""Check YAM grasp/tool frame candidates against TiPToP's 4-DOF grasp convention.

Simulator/planner only. This script does not talk to robot hardware.

cuTAMP uses:
    world_from_ee = world_from_grasp @ tool_from_ee

For TiPToP's built-in 4-DOF cuboid grasp sampler, world_from_grasp is a simple
top-down yaw frame near the object's top. This script checks which
tool_from_ee candidates keep the physical MuJoCo grasp site at that TiPToP
grasp origin while leaving the cuRobo EE pose IK-reachable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import numpy as np


TIPTOP_ROOT = Path(os.environ.get("TIPTOP_ROOT", "/home/drosakis/yam-tamp/tiptop/tiptop"))
for _path in (TIPTOP_ROOT / "cutamp", TIPTOP_ROOT, TIPTOP_ROOT / "curobo" / "src"):
    if _path.exists():
        sys.path.insert(0, str(_path))


PHYSICAL_GRASP_FROM_EE = np.array(
    [
        [1.0, 0.0, 0.0, -0.0065424022],
        [0.0, -1.0, 0.0, 0.0037035275],
        [0.0, 0.0, -1.0, -0.1341572070],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _parse_csv_floats(text: str, expected: int | None = None) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if expected is not None and len(values) != expected:
        raise argparse.ArgumentTypeError(f"Expected {expected} comma-separated values, got {len(values)}")
    return values


def _rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _mat_from_rz_t(yaw: float, t: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = _rz(yaw)
    out[:3, 3] = np.asarray(t, dtype=np.float32)
    return out


def _candidate_tool_from_ee() -> dict[str, np.ndarray]:
    """Return candidate T_grasp_ee transforms used by cuTAMP."""
    ee_from_physical_grasp = np.linalg.inv(PHYSICAL_GRASP_FROM_EE)
    physical_grasp_in_ee = ee_from_physical_grasp[:3, 3]

    # Preserve TiPToP's top-down 4-DOF grasp convention, but choose translation
    # so the physical MuJoCo grasp site origin lands at the virtual TiPToP
    # grasp origin:  T_virtual_physical = T_virtual_ee @ T_ee_physical.
    yaw0_R = _rz(0.0)
    yawpi_R = _rz(np.pi)
    yaw0_t = -(yaw0_R @ physical_grasp_in_ee)
    yawpi_t = -(yawpi_R @ physical_grasp_in_ee)

    candidates = {
        "current-placeholder": _mat_from_rz_t(0.0, (0.0, 0.0, 0.135)),
        "previous-yaw-pi": _mat_from_rz_t(np.pi, (-0.00701567, 0.00414988, 0.13591795)),
        "canonical-yaw-0": _mat_from_rz_t(0.0, yaw0_t),
        "canonical-yaw-pi": _mat_from_rz_t(np.pi, yawpi_t),
        "physical-mujoco-grasp-site": PHYSICAL_GRASP_FROM_EE.copy(),
        "yam-pinch-pad-y-up": np.array(
            [
                [0.0, 1.0, 0.0, -0.00370353],
                [0.0, 0.0, -1.0, -0.11295721],
                [-1.0, 0.0, 0.0, 0.02354240],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        "yam-pinch-pad-y-down": np.array(
            [
                [0.0, 1.0, 0.0, -0.00370353],
                [0.0, 0.0, 1.0, 0.11295721],
                [1.0, 0.0, 0.0, -0.02354240],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    }
    return candidates


def _make_world_grasps(
    cube_pos: np.ndarray,
    cube_dims: np.ndarray,
    yaw_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    # This mirrors grasp_4dof_sampler() for a Cuboid:
    # obj_half_z = obj.dims[2] / 2 - 0.02.
    obj_from_grasp_z = max(0.0, float(cube_dims[2]) / 2.0 - 0.02)
    yaws = np.linspace(-np.pi, np.pi, yaw_count, endpoint=False, dtype=np.float32)

    world_from_grasps: list[np.ndarray] = []
    for yaw in yaws:
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = _rz(float(yaw))
        mat[:3, 3] = cube_pos + np.array([0.0, 0.0, obj_from_grasp_z], dtype=np.float32)
        world_from_grasps.append(mat)
    return np.stack(world_from_grasps, axis=0), yaws


def _dummy_world_cfg():
    from curobo.geom.types import Cuboid, WorldConfig

    return WorldConfig(
        cuboid=[
            Cuboid(
                name="dummy_far_obstacle",
                dims=[0.01, 0.01, 0.01],
                pose=[99.9, 99.9, 99.9, 1.0, 0.0, 0.0, 0.0],
            )
        ]
    )


def _solve_ik(mats: np.ndarray, self_collision_check: bool) -> np.ndarray:
    import torch
    from curobo.types.math import Pose
    from cutamp.robots.yam import get_yam_ik_solver

    ik_solver = get_yam_ik_solver(
        _dummy_world_cfg(),
        self_collision_check=self_collision_check,
        self_collision_opt=False,
    )
    pose = Pose.from_matrix(torch.as_tensor(mats, device="cuda", dtype=torch.float32))
    torch.cuda.synchronize()
    result = ik_solver.solve_batch(pose)
    torch.cuda.synchronize()
    return result.success.detach().cpu().numpy().reshape(-1).astype(bool)


def _score_candidate(
    name: str,
    tool_from_ee: np.ndarray,
    world_from_grasps: np.ndarray,
    approach_offsets: list[float],
    self_collision_check: bool,
) -> None:
    ee_from_physical_grasp = np.linalg.inv(PHYSICAL_GRASP_FROM_EE)
    virtual_from_physical = tool_from_ee @ ee_from_physical_grasp
    physical_origin_error = float(np.linalg.norm(virtual_from_physical[:3, 3]))

    world_from_ee = world_from_grasps @ tool_from_ee
    final_success = _solve_ik(world_from_ee, self_collision_check=self_collision_check)

    approach_rows = []
    approach_successes = []
    for offset in approach_offsets:
        approach_offset = np.eye(4, dtype=np.float32)
        approach_offset[2, 3] = offset
        world_from_approach = world_from_ee @ approach_offset
        success = _solve_ik(world_from_approach, self_collision_check=self_collision_check)
        approach_rows.append((offset, success))
        approach_successes.append(success)

    any_approach_success = np.stack(approach_successes, axis=0).any(axis=0)
    both_success = final_success & any_approach_success

    ee_z = world_from_ee[:, 2, 3]
    print(f"\n{name}")
    print(f"  tool_from_ee translation: {tool_from_ee[:3, 3].tolist()}")
    print(f"  physical grasp origin error in virtual frame: {physical_origin_error:.6f} m")
    print(f"  final EE z range: {float(ee_z.min()):.6f} .. {float(ee_z.max()):.6f}")
    print(f"  final IK success: {int(final_success.sum())}/{len(final_success)}")
    for offset, success in approach_rows:
        print(f"  approach offset {offset:+.3f} IK success: {int(success.sum())}/{len(success)}")
    print(f"  final + any approach success: {int(both_success.sum())}/{len(both_success)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-pos", type=lambda s: _parse_csv_floats(s, 3), default=[0.45, 0.0, 0.025])
    parser.add_argument("--cube-dims", type=lambda s: _parse_csv_floats(s, 3), default=[0.05, 0.05, 0.05])
    parser.add_argument("--yaw-count", type=int, default=16)
    parser.add_argument("--approach-offsets", type=_parse_csv_floats, default=[0.06, 0.05, 0.04, 0.03, 0.02, 0.015])
    parser.add_argument("--disable-self-collision-check", action="store_true")
    args = parser.parse_args()

    cube_pos = np.asarray(args.cube_pos, dtype=np.float32)
    cube_dims = np.asarray(args.cube_dims, dtype=np.float32)
    world_from_grasps, _yaws = _make_world_grasps(cube_pos, cube_dims, args.yaw_count)
    self_collision_check = not args.disable_self_collision_check

    print("YAM TiPToP grasp-frame candidate check")
    print("hardware: not used")
    print(f"TIPTOP_ROOT={TIPTOP_ROOT}")
    print(f"cube_pos={cube_pos.tolist()} cube_dims={cube_dims.tolist()}")
    print(f"yaws={args.yaw_count} self_collision_check={self_collision_check}")
    print("cuTAMP convention: world_from_ee = world_from_grasp @ tool_from_ee")

    for name, candidate in _candidate_tool_from_ee().items():
        _score_candidate(
            name,
            candidate,
            world_from_grasps,
            approach_offsets=[float(v) for v in args.approach_offsets],
            self_collision_check=self_collision_check,
        )


if __name__ == "__main__":
    main()
