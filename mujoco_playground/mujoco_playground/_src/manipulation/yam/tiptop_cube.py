"""YAM arm with a cube scene for TiPToP-style simulation."""

from __future__ import annotations

import atexit
from pathlib import Path
import tempfile
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.005,
      episode_length=500,
      action_repeat=1,
      impl="jax",
      naconmax=512,
      naccdmax=512,
      njmax=256,
      cube_pos=(0.32, 0.0, 0.025),
      camera_pos=(0.34, -0.58, 0.58),
      camera_target=(0.25, 0.0, 0.07),
      fovy=58.0,
      default_arm_qpos=(
          -0.036812390325776434,
          0.8585107194628829,
          1.4494163424124515,
          -1.156824597543297,
          -0.07839322499428114,
          -0.009727626459143934,
      ),
      default_gripper_ctrl=0.03,
  )


def _normalize(v: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(v)
  if norm == 0:
    raise ValueError("Cannot normalize a zero vector")
  return v / norm


def _camera_xyaxes(
    pos: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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

  return f"""<mujoco model="yam tiptop cube">
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


def _yam_dir() -> Path:
  mjx_env.ensure_menagerie_exists()
  path = Path(mjx_env.MENAGERIE_PATH) / "i2rt_yam"
  if not (path / "yam.xml").exists():
    raise FileNotFoundError(f"Could not find YAM assets in {path}")
  return path


def _write_scene_xml(config: config_dict.ConfigDict) -> Path:
  yam_dir = _yam_dir()
  scene_xml = _make_scene_xml(
      camera_pos=np.array(config.camera_pos, dtype=np.float64),
      camera_target=np.array(config.camera_target, dtype=np.float64),
      cube_pos=np.array(config.cube_pos, dtype=np.float64),
      fovy=float(config.fovy),
  )
  with tempfile.NamedTemporaryFile(
      "w", suffix=".xml", dir=yam_dir, delete=False
  ) as tmp:
    tmp.write(scene_xml)
    path = Path(tmp.name)

  atexit.register(lambda p=path: p.unlink(missing_ok=True))
  return path


class TiptopCube(mjx_env.MjxEnv):
  """A scripted-control YAM cube scene inside the Playground env API."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides=config_overrides)
    self._xml_path = _write_scene_xml(self._config)
    self._mj_model = mujoco.MjModel.from_xml_path(str(self._xml_path))
    self._mj_model.opt.timestep = self.sim_dt
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    self._post_init()

  def _post_init(self) -> None:
    self._home_key = self._mj_model.key("home").id
    self._cube_joint = self._mj_model.joint("tiptop_cube_freejoint").id
    self._cube_qadr = self._mj_model.jnt_qposadr[self._cube_joint]
    self._cube_body = self._mj_model.body("tiptop_cube").id
    self._grasp_site = self._mj_model.site("grasp_site").id
    self._left_finger_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.joint("left_finger").id
    ]
    self._right_finger_qadr = self._mj_model.jnt_qposadr[
        self._mj_model.joint("right_finger").id
    ]
    self._ctrl_low = jp.array(self._mj_model.actuator_ctrlrange[:, 0])
    self._ctrl_high = jp.array(self._mj_model.actuator_ctrlrange[:, 1])

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = np.array(self._mj_model.key_qpos[self._home_key], dtype=np.float64)
    qvel = np.zeros(self._mj_model.nv, dtype=np.float64)
    ctrl = np.array(self._mj_model.key_ctrl[self._home_key], dtype=np.float64)

    arm_qpos = np.array(self._config.default_arm_qpos, dtype=np.float64)
    qpos[:6] = arm_qpos
    ctrl[:6] = arm_qpos

    cube_pos = np.array(self._config.cube_pos, dtype=np.float64)
    qpos[self._cube_qadr : self._cube_qadr + 3] = cube_pos
    qpos[self._cube_qadr + 3 : self._cube_qadr + 7] = np.array(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float64
    )

    gripper = float(self._config.default_gripper_ctrl)
    qpos[self._left_finger_qadr] = gripper
    qpos[self._right_finger_qadr] = -gripper
    ctrl[6] = gripper

    data = mjx_env.make_data(
        self._mj_model,
        qpos=jp.array(qpos),
        qvel=jp.array(qvel),
        ctrl=jp.array(ctrl),
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self._mjx_model, data)

    del rng  # Deterministic reset for now; callers still get Playground shape.
    metrics = {"gripper_cube_distance": jp.array(0.0)}
    info = {"time_out": jp.array(0.0)}
    obs = self._get_obs(data)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    ctrl = jp.clip(action, self._ctrl_low, self._ctrl_high)
    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    obs = self._get_obs(data)

    gripper_pos = data.site_xpos[self._grasp_site]
    cube_pos = data.xpos[self._cube_body]
    distance = jp.linalg.norm(gripper_pos - cube_pos)
    reward = -distance
    done = (jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()).astype(float)
    metrics = {"gripper_cube_distance": distance}
    return mjx_env.State(data, obs, reward, done, metrics, state.info)

  def _get_obs(self, data: mjx.Data) -> jax.Array:
    gripper_pos = data.site_xpos[self._grasp_site]
    cube_pos = data.xpos[self._cube_body]
    cube_quat = data.xquat[self._cube_body]
    return jp.concatenate([
        data.qpos,
        data.qvel,
        data.ctrl,
        gripper_pos,
        cube_pos,
        cube_quat,
        cube_pos - gripper_pos,
    ])

  @property
  def xml_path(self) -> str:
    return self._xml_path.as_posix()

  @property
  def action_size(self) -> int:
    return self._mj_model.nu

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
