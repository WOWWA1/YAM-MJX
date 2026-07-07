#!/usr/bin/env python3
"""Generate first-pass cuRobo assets for YAM from I2RT meshes.

This is simulator/planner-only tooling. It builds a repo-local asset folder for
TiPToP/cuRobo without baking in Mac or RunPod absolute paths.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np


I2RT_REPO = "https://github.com/i2rt-robotics/i2rt.git"
I2RT_COMMIT_SHA = "c40b41258e3898cf925713126bd3210877809ef5"

PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DEPS_DIR = PLAYGROUND_ROOT / "mujoco_playground" / "external_deps"
I2RT_DIR = EXTERNAL_DEPS_DIR / "i2rt"
I2RT_YAM_DIR = I2RT_DIR / "i2rt" / "robot_models" / "arm" / "yam"
OUTPUT_DIR = PLAYGROUND_ROOT / "tiptop_yam_assets"

JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
COLLISION_SPHERE_COUNTS = {
    "base_link": 3,
    "link_1": 4,
    "link_2": 8,
    "link_3": 8,
    "link_4": 6,
    "link_5": 5,
}

# MuJoCo Menagerie has link_6 and gripper collision primitives, but I2RT's URDF
# currently references missing link_6 STL files. These spheres approximate that
# fixed wrist/open-gripper envelope until we port the linear gripper as real URDF.
LINK_6_PROXY_SPHERES = [
    {"center": [0.0, 0.0, 0.030], "radius": 0.031},
    {"center": [0.0, 0.039, 0.052], "radius": 0.020},
    {"center": [0.0, -0.039, 0.052], "radius": 0.020},
    {"center": [0.0, 0.0, 0.082], "radius": 0.025},
    {"center": [0.037, 0.039, 0.098], "radius": 0.014},
    {"center": [0.037, 0.039, 0.122], "radius": 0.012},
    {"center": [-0.037, -0.039, 0.098], "radius": 0.014},
    {"center": [-0.037, -0.039, 0.122], "radius": 0.012},
]

GRASP_FRAME_FROM_LINK_6 = [
    0.0,
    0.0,
    0.1347,
    math.sqrt(0.5),
    0.0,
    0.0,
    -math.sqrt(0.5),
]


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_i2rt_assets() -> None:
    """Clone the small I2RT model subset needed by the generator."""
    EXTERNAL_DEPS_DIR.mkdir(parents=True, exist_ok=True)
    if not I2RT_DIR.exists():
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                I2RT_REPO,
                str(I2RT_DIR),
            ]
        )

    _run(
        [
            "git",
            "-C",
            str(I2RT_DIR),
            "sparse-checkout",
            "set",
            "i2rt/robot_models/arm/yam",
            "i2rt/robot_models/gripper/linear_4310",
        ]
    )
    try:
        _run(["git", "-C", str(I2RT_DIR), "checkout", I2RT_COMMIT_SHA])
    except subprocess.CalledProcessError:
        _run(["git", "-C", str(I2RT_DIR), "fetch", "--depth", "1", "origin", I2RT_COMMIT_SHA])
        _run(["git", "-C", str(I2RT_DIR), "checkout", I2RT_COMMIT_SHA])

    if not (I2RT_YAM_DIR / "yam.urdf").exists():
        raise FileNotFoundError(f"Could not find I2RT YAM URDF at {I2RT_YAM_DIR / 'yam.urdf'}")


def _parse_vector(text: str | None, expected: int) -> np.ndarray:
    if not text:
        return np.zeros(expected, dtype=np.float64)
    values = [float(part) for part in text.split()]
    if len(values) != expected:
        raise ValueError(f"Expected {expected} values in {text!r}, got {len(values)}")
    return np.array(values, dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _origin_transform(origin: ET.Element | None) -> np.ndarray:
    xyz = _parse_vector(origin.get("xyz") if origin is not None else None, 3)
    rpy = _parse_vector(origin.get("rpy") if origin is not None else None, 3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def _resolve_package_mesh(filename: str, yam_dir: Path) -> Path:
    if filename.startswith("package://assets/"):
        return yam_dir / filename.removeprefix("package://")
    if filename.startswith("package://"):
        return yam_dir / filename.removeprefix("package://")
    return yam_dir / filename


def _sanitize_urdf(src_urdf: Path, dst_urdf: Path) -> list[str]:
    """Copy the URDF and remove mesh references missing from the I2RT checkout."""
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    removed: list[str] = []

    for link in root.findall("link"):
        for child in list(link):
            if child.tag not in {"visual", "collision"}:
                continue
            mesh = child.find("./geometry/mesh")
            if mesh is None:
                continue
            filename = mesh.get("filename", "")
            mesh_path = _resolve_package_mesh(filename, src_urdf.parent)
            if not mesh_path.exists():
                removed.append(f"{link.get('name')}/{child.tag}: {filename}")
                link.remove(child)
                continue
            if filename.startswith("package://assets/"):
                mesh.set("filename", filename.removeprefix("package://"))

    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)
    return removed


def _collision_mesh_specs(urdf_path: Path, yam_dir: Path) -> dict[str, list[tuple[Path, np.ndarray]]]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    specs: dict[str, list[tuple[Path, np.ndarray]]] = defaultdict(list)
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            continue
        for collision in link.findall("collision"):
            mesh = collision.find("./geometry/mesh")
            if mesh is None:
                continue
            mesh_path = _resolve_package_mesh(mesh.get("filename", ""), yam_dir)
            if not mesh_path.exists():
                continue
            specs[link_name].append((mesh_path, _origin_transform(collision.find("origin"))))
    return dict(specs)


def _load_transformed_mesh(mesh_path: Path, transform: np.ndarray):
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit(
            "This generator needs trimesh. Run it inside the TiPToP pixi environment on RunPod."
        ) from exc

    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    mesh = loaded.copy()
    mesh.apply_transform(transform)
    return mesh


def _fit_link_spheres(
    link_name: str,
    specs: list[tuple[Path, np.ndarray]],
    n_spheres: int,
    surface_radius: float,
    voxelize_method: str,
    refine_iters: int,
    max_cover_points: int,
):
    try:
        import trimesh
        from curobo.geom.sphere_fit import SphereFitType, fit_spheres_to_mesh
    except ImportError as exc:
        raise SystemExit(
            "This generator needs cuRobo and trimesh. Run it with pixi from external/tiptop on RunPod."
        ) from exc

    meshes = [_load_transformed_mesh(mesh_path, transform) for mesh_path, transform in specs]
    combined = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
    try:
        centers, radii = fit_spheres_to_mesh(
            combined,
            n_spheres=n_spheres,
            surface_sphere_radius=surface_radius,
            fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            voxelize_method=voxelize_method,
        )
    except Exception as exc:  # pragma: no cover - depends on mesh voxelizer details.
        print(
            f"Warning: voxel fit failed for {link_name} ({exc}); falling back to surface sampling.",
            file=sys.stderr,
        )
        centers, radii = fit_spheres_to_mesh(
            combined,
            n_spheres=n_spheres,
            surface_sphere_radius=surface_radius,
            fit_type=SphereFitType.SAMPLE_SURFACE,
            voxelize_method=voxelize_method,
        )
    centers = _cover_mesh_with_centers(
        combined,
        centers=np.asarray(centers, dtype=np.float64),
        n_spheres=n_spheres,
        padding=surface_radius,
        refine_iters=refine_iters,
        max_cover_points=max_cover_points,
    )
    return centers


def _mesh_cover_points(mesh, max_cover_points: int) -> np.ndarray:
    points = np.vstack([np.asarray(mesh.vertices), np.asarray(mesh.triangles_center)])
    if len(points) <= max_cover_points:
        return points

    rng = np.random.default_rng(0)
    idx = rng.choice(len(points), size=max_cover_points, replace=False)
    return points[idx]


def _nearest_labels(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - centers[None, :, :]
    return np.argmin(np.einsum("ijk,ijk->ij", diff, diff), axis=1)


def _augment_centers(points: np.ndarray, centers: np.ndarray, n_spheres: int) -> np.ndarray:
    if centers.ndim != 2 or centers.shape[1] != 3:
        centers = np.empty((0, 3), dtype=np.float64)
    if len(centers) == 0:
        centers = np.mean(points, axis=0, keepdims=True)

    centers = centers[:n_spheres].astype(np.float64)
    while len(centers) < n_spheres:
        diff = points[:, None, :] - centers[None, :, :]
        nearest_d2 = np.min(np.einsum("ijk,ijk->ij", diff, diff), axis=1)
        centers = np.vstack([centers, points[int(np.argmax(nearest_d2))]])
    return centers


def _cover_mesh_with_centers(
    mesh,
    centers: np.ndarray,
    n_spheres: int,
    padding: float,
    refine_iters: int,
    max_cover_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use cuRobo centers as seeds, then size spheres to cover mesh samples."""
    points = _mesh_cover_points(mesh, max_cover_points=max_cover_points)
    centers = _augment_centers(points, centers, n_spheres)

    for _ in range(refine_iters):
        labels = _nearest_labels(points, centers)
        for i in range(len(centers)):
            assigned = points[labels == i]
            if len(assigned) > 0:
                centers[i] = np.mean(assigned, axis=0)

    labels = _nearest_labels(points, centers)
    radii = np.empty(len(centers), dtype=np.float64)
    for i in range(len(centers)):
        assigned = points[labels == i]
        if len(assigned) == 0:
            radii[i] = padding
        else:
            radii[i] = np.max(np.linalg.norm(assigned - centers[i], axis=1)) + padding
    return centers, radii


def _clean_float(value: float, precision: int) -> float:
    rounded = round(float(value), precision)
    return 0.0 if rounded == -0.0 else rounded


def _format_spheres(centers, radii, precision: int) -> list[dict[str, object]]:
    spheres = []
    for center, radius in zip(np.asarray(centers), np.ravel(radii)):
        spheres.append(
            {
                "center": [_clean_float(v, precision) for v in center.tolist()],
                "radius": _clean_float(max(float(radius), 0.001), precision),
            }
        )
    return spheres


def _format_proxy_spheres(precision: int) -> list[dict[str, object]]:
    return [
        {
            "center": [_clean_float(v, precision) for v in sphere["center"]],
            "radius": _clean_float(float(sphere["radius"]), precision),
        }
        for sphere in LINK_6_PROXY_SPHERES
    ]


def _robot_config(collision_spheres: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    collision_link_names = list(collision_spheres) + ["attached_object"]
    return {
        "robot_cfg": {
            "kinematics": {
                "use_usd_kinematics": False,
                "urdf_path": "yam.urdf",
                "asset_root_path": "",
                "base_link": "base_link",
                "ee_link": "grasp_frame",
                "link_names": None,
                "lock_joints": {},
                "extra_links": {
                    "grasp_frame": {
                        "parent_link_name": "link_6",
                        "link_name": "grasp_frame",
                        "fixed_transform": GRASP_FRAME_FROM_LINK_6,
                        "joint_type": "FIXED",
                        "joint_name": "grasp_frame_joint",
                    },
                    "attached_object": {
                        "parent_link_name": "grasp_frame",
                        "link_name": "attached_object",
                        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                        "joint_type": "FIXED",
                        "joint_name": "attach_joint",
                    },
                },
                "extra_collision_spheres": {"attached_object": 4},
                "collision_link_names": collision_link_names,
                "collision_spheres": collision_spheres,
                "collision_sphere_buffer": 0.005,
                "self_collision_ignore": {
                    "base_link": ["link_1", "link_2"],
                    "link_1": ["base_link", "link_2", "link_3"],
                    "link_2": ["base_link", "link_1", "link_3", "link_4"],
                    "link_3": ["link_1", "link_2", "link_4", "link_5"],
                    "link_4": ["link_2", "link_3", "link_5", "link_6"],
                    "link_5": ["link_3", "link_4", "link_6", "attached_object"],
                    "link_6": ["link_4", "link_5", "attached_object"],
                },
                "self_collision_buffer": {
                    "base_link": 0.0,
                    "link_1": 0.0,
                    "link_2": 0.0,
                    "link_3": 0.0,
                    "link_4": 0.0,
                    "link_5": 0.0,
                    "link_6": 0.0,
                    "attached_object": 0.0,
                },
                "cspace": {
                    "joint_names": JOINT_NAMES,
                    "retract_config": [
                        -0.0368123903,
                        0.8585107195,
                        1.4494163424,
                        -1.1568245975,
                        -0.0783932250,
                        -0.0097276265,
                    ],
                    "null_space_weight": [1.0] * len(JOINT_NAMES),
                    "cspace_distance_weight": [1.0] * len(JOINT_NAMES),
                    "max_jerk": 500.0,
                    "max_acceleration": 12.0,
                    "position_limit_clip": 0.0,
                },
            }
        }
    }


def _write_yaml(path: Path, data: dict[str, object]) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("This generator needs PyYAML. Run it in the TiPToP pixi environment.") from exc

    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))


def _write_readme(path: Path, removed_urdf_meshes: list[str]) -> None:
    removed = "\n".join(f"- {item}" for item in removed_urdf_meshes) or "- None"
    path.write_text(
        f"""# Generated YAM cuRobo Assets

Generated from:

- I2RT repository: `{I2RT_REPO}`
- I2RT commit: `{I2RT_COMMIT_SHA}`

The arm collision spheres are generated with cuRobo's `fit_spheres_to_mesh`
from I2RT collision STL meshes. `link_6` and the linear gripper currently use
conservative proxy spheres because this I2RT checkout references missing
`link_6` mesh files and the gripper is supplied as MJCF rather than a cuRobo
URDF branch.

Removed missing URDF mesh references:

{removed}
"""
    )


def _validate_curobo(config_path: Path, asset_dir: Path) -> None:
    import torch
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.types.robot import RobotConfig
    from curobo.util_file import load_yaml

    cfg = load_yaml(str(config_path))["robot_cfg"]
    cfg["kinematics"]["external_asset_path"] = str(asset_dir)
    cfg["kinematics"]["external_robot_configs_path"] = str(asset_dir)
    robot_cfg = RobotConfig.from_dict(cfg)
    model = CudaRobotModel(robot_cfg.kinematics)
    q = torch.tensor(cfg["kinematics"]["cspace"]["retract_config"], device="cuda", dtype=torch.float32)[None]
    state = model.get_state(q)
    torch.cuda.synchronize()
    print("cuRobo validation: ok")
    print(f"  joints: {model.joint_names}")
    print(f"  ee position: {state.ee_pose.position.detach().cpu().numpy().round(6).tolist()[0]}")
    print(f"  spheres: {tuple(state.link_spheres_tensor.shape)}")


def generate(args: argparse.Namespace) -> Path:
    ensure_i2rt_assets()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assets_dst = output_dir / "assets"
    if assets_dst.exists():
        shutil.rmtree(assets_dst)
    shutil.copytree(I2RT_YAM_DIR / "assets", assets_dst)

    removed_urdf_meshes = _sanitize_urdf(I2RT_YAM_DIR / "yam.urdf", output_dir / "yam.urdf")
    specs = _collision_mesh_specs(I2RT_YAM_DIR / "yam.urdf", I2RT_YAM_DIR)

    collision_spheres: dict[str, list[dict[str, object]]] = {}
    for link_name, n_spheres in COLLISION_SPHERE_COUNTS.items():
        if link_name not in specs:
            raise RuntimeError(f"No collision mesh available for {link_name}")
        centers, radii = _fit_link_spheres(
            link_name,
            specs[link_name],
            n_spheres=n_spheres,
            surface_radius=args.surface_radius,
            voxelize_method=args.voxelize_method,
            refine_iters=args.refine_iters,
            max_cover_points=args.max_cover_points,
        )
        collision_spheres[link_name] = _format_spheres(centers, radii, args.precision)
        print(f"{link_name}: generated {len(collision_spheres[link_name])} spheres")

    collision_spheres["link_6"] = _format_proxy_spheres(args.precision)
    print(f"link_6: generated {len(collision_spheres['link_6'])} proxy spheres")

    _write_yaml(output_dir / "yam.yml", _robot_config(collision_spheres))
    _write_readme(output_dir / "README.md", removed_urdf_meshes)

    print(f"Wrote {output_dir / 'yam.yml'}")
    if args.validate:
        _validate_curobo(output_dir / "yam.yml", output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--surface-radius", type=float, default=0.005)
    parser.add_argument("--voxelize-method", default="ray")
    parser.add_argument("--refine-iters", type=int, default=8)
    parser.add_argument("--max-cover-points", type=int, default=12000)
    parser.add_argument("--precision", type=int, default=6)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
