"""YAM robot helpers for a local TiPToP/cuTAMP checkout.

This file is installed into ``external/tiptop/cutamp/cutamp/robots/yam.py`` by
``install_tiptop_yam_bridge.py``. It intentionally resolves the generated YAM
assets from the surrounding YAM-MJX checkout instead of hardcoding a machine path.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from jaxtyping import Float
from yourdfpy import URDF

from cutamp.robots.utils import RerunRobot


yam_home = (
    -0.0368123903,
    0.8585107195,
    1.4494163424,
    -1.1568245975,
    -0.0783932250,
    -0.0097276265,
)


def _yam_assets_dir() -> Path:
    env_path = os.environ.get("YAM_TIPTOP_ASSETS_DIR")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if (path / "yam.yml").exists():
            return path
        raise FileNotFoundError(f"YAM_TIPTOP_ASSETS_DIR does not contain yam.yml: {path}")

    for parent in Path(__file__).resolve().parents:
        candidate = parent / "mujoco_playground" / "tiptop_yam_assets"
        if (candidate / "yam.yml").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find generated YAM cuRobo assets. Run "
        "`python mujoco_playground/scripts/generate_yam_curobo_assets.py` "
        "from the YAM-MJX checkout, or set YAM_TIPTOP_ASSETS_DIR."
    )


@lru_cache(maxsize=1)
def yam_curobo_cfg() -> dict:
    assets_dir = _yam_assets_dir()
    cfg = load_yaml(str(assets_dir / "yam.yml"))
    kinematics = cfg["robot_cfg"]["kinematics"]
    kinematics["external_asset_path"] = str(assets_dir)
    kinematics["external_robot_configs_path"] = str(assets_dir)
    return cfg


def _yam_cfg_dict() -> dict:
    return yam_curobo_cfg()["robot_cfg"]


def get_yam_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model for YAM."""
    robot_cfg = RobotConfig.from_dict(_yam_cfg_dict())
    return CudaRobotModel(robot_cfg.kinematics)


def get_yam_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """cuRobo IK solver for YAM."""
    ik_config = IKSolverConfig.load_from_robot_config(
        _yam_cfg_dict(),
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    return IKSolver(ik_config)


def get_yam_tool_from_ee(tensor_args: TensorDeviceType = TensorDeviceType()) -> Float[torch.Tensor, "4 4"]:
    """Transform from TiPToP's grasp/tool frame to YAM's cuRobo EE frame.

    TiPToP/cuTAMP uses ``world_from_ee = world_from_grasp @ tool_from_ee``.
    This calibration puts the TiPToP grasp origin at the midpoint of YAM's
    inner fingertip pads instead of at the wrist/link_6 frame.
    """
    return tensor_args.to_device(
        [
            [0.0, 1.0, 0.0, -0.00370353],
            [0.0, 0.0, -1.0, -0.11295721],
            [-1.0, 0.0, 0.0, 0.02354240],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _transform_sphere_centers(transform: torch.Tensor, spheres: torch.Tensor) -> torch.Tensor:
    transformed = spheres.clone()
    transformed[:, :3] = spheres[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    return transformed


def get_yam_gripper_spheres(tensor_args: TensorDeviceType = TensorDeviceType()) -> Float[torch.Tensor, "num_spheres 4"]:
    """Approximate YAM gripper spheres in the TiPToP tool frame."""
    ee_spheres = tensor_args.to_device(
        [
            [0.0, 0.0, -0.055, 0.026],
            [0.0, 0.0, -0.025, 0.020],
            [0.032, 0.0, -0.018, 0.012],
            [-0.032, 0.0, -0.018, 0.012],
            [0.032, 0.0, 0.010, 0.010],
            [-0.032, 0.0, 0.010, 0.010],
        ]
    )
    return _transform_sphere_centers(get_yam_tool_from_ee(tensor_args), ee_spheres)



def load_yam_rerun(load_mesh: bool = True) -> RerunRobot:
    """Load a Rerun-friendly URDF representation for visualization."""
    assets_dir = _yam_assets_dir()
    urdf_path = assets_dir / yam_curobo_cfg()["robot_cfg"]["kinematics"]["urdf_path"]

    def _locate_asset(fname: str) -> str:
        if fname.startswith("package://"):
            fname = fname.removeprefix("package://")
        return str(assets_dir / fname)

    urdf = URDF.load(str(urdf_path), filename_handler=_locate_asset)
    return RerunRobot("yam", urdf, q_neutral=yam_home, load_mesh=load_mesh)
