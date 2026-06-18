"""Export a MuJoCo YAM observation in TiPToP H5 format.

This script is simulator-only. It loads the local YAM MJCF, adds a simple cube
and a fixed RGB-D camera, then writes the fields consumed by `tiptop-h5`:

  rgb, depth, intrinsic_matrix, pos_w, quat_w_ros, q_init
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import tempfile

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
YAM_DIR = (
    REPO_ROOT
    / "mujoco_playground"
    / "external_deps"
    / "mujoco_menagerie"
    / "i2rt_yam"
)
YAM_XML = YAM_DIR / "yam.xml"

DEFAULT_Q_CAPTURE = np.array(
    [
        -0.036812390325776434,
        0.8585107194628829,
        1.4494163424124515,
        -1.156824597543297,
        -0.07839322499428114,
        -0.009727626459143934,
    ],
    dtype=np.float64,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError("Cannot normalize a zero vector")
    return v / norm


def _camera_xyaxes(pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return MuJoCo camera x/y axes for a camera looking at target.

    MuJoCo cameras use an OpenGL-style local frame: +x right, +y up, -z forward.
    """
    forward = _normalize(target - pos)
    z_axis = -forward
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = _normalize(np.cross(up_hint, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return x_axis, y_axis


def _make_scene_xml(
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    cube_pos: np.ndarray,
    fovy: float,
) -> str:
    x_axis, y_axis = _camera_xyaxes(camera_pos, camera_target)
    xyaxes = " ".join(f"{v:.9g}" for v in np.concatenate([x_axis, y_axis]))
    cam_pos = " ".join(f"{v:.9g}" for v in camera_pos)
    cube_pos_s = " ".join(f"{v:.9g}" for v in cube_pos)

    return f"""<mujoco model="yam tiptop h5 scene">
  <include file="yam.xml"/>

  <statistic center="0.2 -0.1 0.3" extent="0.65"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <material name="tiptop_red" rgba="0.9 0.08 0.06 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1" pos="0 0 -0.01"/>
    <camera name="tiptop_cam" pos="{cam_pos}" xyaxes="{xyaxes}" fovy="{fovy:.9g}"/>
    <body name="tiptop_cube" pos="{cube_pos_s}">
      <freejoint name="tiptop_cube_freejoint"/>
      <geom name="tiptop_cube_geom" type="box" size="0.025 0.025 0.025" material="tiptop_red" mass="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def _intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> np.ndarray:
    fovy = math.radians(fovy_deg)
    fy = 0.5 * height / math.tan(0.5 * fovy)
    fx = fy
    return np.array(
        [
            [fx, 0.0, (width - 1.0) * 0.5],
            [0.0, fy, (height - 1.0) * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _world_from_cv_camera(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str) -> np.ndarray:
    cam_id = model.camera(camera_name).id
    world_from_mj_cam = data.cam_xmat[cam_id].reshape(3, 3)

    # MuJoCo/OpenGL camera: +x right, +y up, -z forward.
    # TiPToP/OpenCV projection: +x right, +y down, +z forward.
    mj_from_cv = np.diag([1.0, -1.0, -1.0])
    world_from_cv = np.eye(4, dtype=np.float32)
    world_from_cv[:3, :3] = world_from_mj_cam @ mj_from_cv
    world_from_cv[:3, 3] = data.cam_xpos[cam_id]
    return world_from_cv


def _quat_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def save_h5(
    output: Path,
    preview_png: Path | None,
    width: int,
    height: int,
    q_capture: np.ndarray,
    camera_pos: np.ndarray,
    camera_target: np.ndarray,
    cube_pos: np.ndarray,
    fovy: float,
) -> None:
    if not YAM_XML.exists():
        raise FileNotFoundError(f"Could not find YAM XML at {YAM_XML}")

    scene_xml = _make_scene_xml(camera_pos, camera_target, cube_pos, fovy)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=YAM_DIR, delete=False) as tmp:
        tmp.write(scene_xml)
        tmp_xml = Path(tmp.name)

    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        tmp_xml.unlink(missing_ok=True)

    data = mujoco.MjData(model)

    home_id = model.key("home").id
    data.qpos[:] = model.key_qpos[home_id]
    data.ctrl[:] = model.key_ctrl[home_id]
    data.qpos[:6] = q_capture
    data.ctrl[:6] = q_capture

    cube_jid = model.joint("tiptop_cube_freejoint").id
    cube_qadr = model.jnt_qposadr[cube_jid]
    data.qpos[cube_qadr : cube_qadr + 3] = cube_pos
    data.qpos[cube_qadr + 3 : cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    # Open the coupled gripper a bit so it does not dominate the depth image.
    if model.nq >= 8:
        data.qpos[6] = 0.03
        data.qpos[7] = -0.03
    if model.nu >= 7:
        data.ctrl[6] = 0.03

    mujoco.mj_forward(model, data)

    K = _intrinsics_from_fovy(width, height, fovy)
    world_from_cam = _world_from_cv_camera(model, data, "tiptop_cam")
    quat_w_ros = _quat_wxyz_from_matrix(world_from_cam[:3, :3])

    with mujoco.Renderer(model, height=height, width=width) as renderer:
        renderer.update_scene(data, camera="tiptop_cam")
        rgb = renderer.render()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="tiptop_cam")
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()

    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as f:
        f.create_dataset("rgb", data=rgb.astype(np.uint8))
        f.create_dataset("depth", data=depth.astype(np.float32))
        f.create_dataset("intrinsic_matrix", data=K)
        f.create_dataset("pos_w", data=world_from_cam[:3, 3].astype(np.float32))
        f.create_dataset("quat_w_ros", data=quat_w_ros)
        f.create_dataset("q_init", data=q_capture.astype(np.float32))

    if preview_png is not None:
        import imageio.v3 as iio

        preview_png.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(preview_png, rgb)

    print(f"Wrote TiPToP H5 observation: {output}")
    print(f"  rgb: {rgb.shape} {rgb.dtype}")
    print(f"  depth: {depth.shape} min={float(depth[depth > 0].min()):.4f} max={float(depth.max()):.4f}")
    print(f"  q_init: {q_capture.tolist()}")
    print(f"  pos_w: {world_from_cam[:3, 3].tolist()}")
    print(f"  quat_w_ros: {quat_w_ros.tolist()}")
    if preview_png is not None:
        print(f"Wrote preview PNG: {preview_png}")


def _parse_vec(text: str, expected: int, name: str) -> np.ndarray:
    values = np.array([float(x.strip()) for x in text.split(",") if x.strip()], dtype=np.float64)
    if values.shape != (expected,):
        raise argparse.ArgumentTypeError(f"{name} must have {expected} comma-separated values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/yam_tiptop_obs.h5"))
    parser.add_argument("--preview-png", type=Path, default=Path("/tmp/yam_tiptop_obs.png"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fovy", type=float, default=58.0)
    parser.add_argument(
        "--q-capture",
        type=lambda s: _parse_vec(s, 6, "q-capture"),
        default=DEFAULT_Q_CAPTURE,
        help="Six comma-separated YAM arm joint positions.",
    )
    parser.add_argument(
        "--camera-pos",
        type=lambda s: _parse_vec(s, 3, "camera-pos"),
        default=np.array([0.34, -0.58, 0.58], dtype=np.float64),
    )
    parser.add_argument(
        "--camera-target",
        type=lambda s: _parse_vec(s, 3, "camera-target"),
        default=np.array([0.25, 0.0, 0.07], dtype=np.float64),
    )
    parser.add_argument(
        "--cube-pos",
        type=lambda s: _parse_vec(s, 3, "cube-pos"),
        default=np.array([0.32, 0.0, 0.025], dtype=np.float64),
    )
    args = parser.parse_args()

    save_h5(
        output=args.output,
        preview_png=args.preview_png,
        width=args.width,
        height=args.height,
        q_capture=args.q_capture,
        camera_pos=args.camera_pos,
        camera_target=args.camera_target,
        cube_pos=args.cube_pos,
        fovy=args.fovy,
    )


if __name__ == "__main__":
    main()
