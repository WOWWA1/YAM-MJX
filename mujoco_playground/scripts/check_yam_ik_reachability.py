#!/usr/bin/env python3
"""Simulator/planner-only YAM IK reachability sweep near the TiPToP cube.

This does not connect to, read from, or command robot hardware. It asks cuRobo
whether simple end-effector poses around the simulated cube are IK-solvable.
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


def _make_targets(
    cube_pos: np.ndarray,
    xy_offsets: list[tuple[float, float]],
    ee_z_offsets: list[float],
    yaw_count: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    mats: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    yaws = np.linspace(-np.pi, np.pi, yaw_count, endpoint=False)
    for xy_index, (dx, dy) in enumerate(xy_offsets):
        for z_offset in ee_z_offsets:
            for yaw in yaws:
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = _rz(float(yaw))
                mat[:3, 3] = cube_pos + np.array([dx, dy, z_offset], dtype=np.float32)
                mats.append(mat)
                rows.append(
                    {
                        "xy_index": float(xy_index),
                        "dx": float(dx),
                        "dy": float(dy),
                        "ee_z_offset": float(z_offset),
                        "yaw": float(yaw),
                        "x": float(mat[0, 3]),
                        "y": float(mat[1, 3]),
                        "z": float(mat[2, 3]),
                    }
                )
    return np.stack(mats, axis=0), rows


def _solve_targets(mats: np.ndarray, self_collision_check: bool):
    import torch
    from curobo.types.math import Pose
    from cutamp.robots.yam import get_yam_ik_solver

    world_cfg = _dummy_world_cfg()
    ik_solver = get_yam_ik_solver(
        world_cfg,
        self_collision_check=self_collision_check,
        self_collision_opt=False,
    )
    pose = Pose.from_matrix(torch.as_tensor(mats, device="cuda", dtype=torch.float32))
    torch.cuda.synchronize()
    result = ik_solver.solve_batch(pose)
    torch.cuda.synchronize()
    return result.success.detach().cpu().numpy().reshape(-1).astype(bool), result


def _print_grouped(rows: list[dict[str, float]], success: np.ndarray, key: str) -> None:
    values = sorted({row[key] for row in rows})
    print(f"\nSuccess by {key}:")
    for value in values:
        idx = np.array([row[key] == value for row in rows], dtype=bool)
        print(f"  {key}={value: .6f}: {int(success[idx].sum())}/{int(idx.sum())}")


def _print_summary(label: str, rows: list[dict[str, float]], success: np.ndarray, result) -> None:
    print(f"\n=== {label} ===")
    print(f"successes: {int(success.sum())}/{len(success)} ({success.mean() * 100.0:.1f}%)")
    if hasattr(result, "status"):
        print(f"status: {result.status}")
    _print_grouped(rows, success, "ee_z_offset")
    _print_grouped(rows, success, "xy_index")

    yaw_values = sorted({row["yaw"] for row in rows})
    yaw_counts = []
    for yaw in yaw_values:
        idx = np.array([row["yaw"] == yaw for row in rows], dtype=bool)
        yaw_counts.append((int(success[idx].sum()), int(idx.sum()), yaw))
    print("\nBest yaw buckets:")
    for ok, total, yaw in sorted(yaw_counts, reverse=True)[:8]:
        print(f"  yaw={yaw: .6f}: {ok}/{total}")

    print("\nFirst successful targets:")
    shown = 0
    for row, ok in zip(rows, success):
        if not ok:
            continue
        print(
            "  "
            f"pos=({row['x']:.4f}, {row['y']:.4f}, {row['z']:.4f}) "
            f"z_offset={row['ee_z_offset']:.4f} "
            f"xy_offset=({row['dx']:.4f}, {row['dy']:.4f}) "
            f"yaw={row['yaw']:.4f}"
        )
        shown += 1
        if shown >= 10:
            break
    if shown == 0:
        print("  none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube-pos", type=lambda s: _parse_csv_floats(s, 3), default=[0.45, 0.0, 0.025])
    parser.add_argument(
        "--ee-z-offsets",
        type=_parse_csv_floats,
        default=[0.10, 0.13, 0.16, 0.19, 0.22, 0.25],
        help="End-effector z offsets above cube-pos z, not absolute z values.",
    )
    parser.add_argument("--yaw-count", type=int, default=16)
    parser.add_argument(
        "--xy-offsets",
        default="0,0;-0.007,0.004;-0.02,0;0.02,0;0,-0.02;0,0.02",
        help="Semicolon-separated dx,dy offsets from cube position.",
    )
    parser.add_argument("--skip-self-collision-check", action="store_true")
    args = parser.parse_args()

    cube_pos = np.asarray(args.cube_pos, dtype=np.float32)
    xy_offsets = []
    for pair in args.xy_offsets.split(";"):
        dx, dy = _parse_csv_floats(pair, 2)
        xy_offsets.append((dx, dy))

    mats, rows = _make_targets(
        cube_pos=cube_pos,
        xy_offsets=xy_offsets,
        ee_z_offsets=[float(v) for v in args.ee_z_offsets],
        yaw_count=args.yaw_count,
    )

    print("YAM IK reachability sweep near simulated cube")
    print("hardware: not used")
    print(f"cube_pos={cube_pos.tolist()}")
    print(f"targets={len(rows)}")
    print(f"TIPTOP_ROOT={TIPTOP_ROOT}")

    if not args.skip_self_collision_check:
        success, result = _solve_targets(mats, self_collision_check=True)
        _print_summary("self_collision_check=True", rows, success, result)

    success, result = _solve_targets(mats, self_collision_check=False)
    _print_summary("self_collision_check=False", rows, success, result)


if __name__ == "__main__":
    main()
