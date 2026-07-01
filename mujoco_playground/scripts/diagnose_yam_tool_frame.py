"""Compare YAM MuJoCo grasp frames with cuRobo's YAM ee_link frame.

Simulator/planner diagnostic only. This does not connect to or command hardware.
Run it from the TiPToP pixi environment so cutamp/curobo are importable.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys
import tempfile

import numpy as np

_MUJOCO_VENV_SITE = Path(__file__).resolve().parents[1] / ".venv" / "lib" / "python3.12" / "site-packages"
if _MUJOCO_VENV_SITE.exists():
    sys.path.append(str(_MUJOCO_VENV_SITE))

import mujoco

from save_tiptop_h5_from_yam import DEFAULT_Q_CAPTURE, _make_scene_xml, _parse_vec
from yam_assets import YAM_DIR, require_yam_file


DEFAULT_CAMERA_POS = np.array([0.65, -0.30, 0.42], dtype=np.float64)
DEFAULT_CAMERA_TARGET = np.array([0.45, 0.0, 0.025], dtype=np.float64)
DEFAULT_CUBE_POS = np.array([0.45, 0.0, 0.025], dtype=np.float64)


def _rz(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _pose_from_site(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    site_id = model.site(site_name).id
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site_id]
    return pose


def _pose_from_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = model.body(body_name).id
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose


def _load_model(camera_pos: np.ndarray, camera_target: np.ndarray, cube_pos: np.ndarray, fovy: float) -> mujoco.MjModel:
    require_yam_file("yam.xml")
    scene_xml = _make_scene_xml(
        camera_pos=camera_pos,
        camera_target=camera_target,
        cube_pos=cube_pos,
        fovy=fovy,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=YAM_DIR, delete=False) as tmp:
        tmp.write(scene_xml)
        tmp_xml = Path(tmp.name)

    try:
        return mujoco.MjModel.from_xml_path(str(tmp_xml))
    finally:
        tmp_xml.unlink(missing_ok=True)


def _set_state(model: mujoco.MjModel, data: mujoco.MjData, q: np.ndarray, cube_pos: np.ndarray) -> None:
    home_id = model.key("home").id
    data.qpos[:] = model.key_qpos[home_id]
    data.ctrl[:] = model.key_ctrl[home_id]
    data.qpos[:6] = q
    data.ctrl[:6] = q

    cube_jid = model.joint("tiptop_cube_freejoint").id
    cube_qadr = model.jnt_qposadr[cube_jid]
    data.qpos[cube_qadr : cube_qadr + 3] = cube_pos
    data.qpos[cube_qadr + 3 : cube_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    if model.nq >= 8:
        data.qpos[6] = 0.03
        data.qpos[7] = -0.03
    if model.nu >= 7:
        data.ctrl[6] = 0.03

    mujoco.mj_forward(model, data)


@lru_cache(maxsize=1)
def _curobo_context():
    import torch
    from curobo.types.base import TensorDeviceType
    from cutamp.robots import load_yam_container

    tensor_args = TensorDeviceType()
    container = load_yam_container(tensor_args)
    return torch, tensor_args, container


def _curobo_ee_pose(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    torch, tensor_args, container = _curobo_context()
    q_t = tensor_args.to_device(q).view(1, -1)
    with torch.no_grad():
        ee_pose = container.kin_model.get_state(q_t).ee_pose.get_numpy_matrix()[0]
    tool_from_ee = container.tool_from_ee.detach().cpu().numpy()
    return np.asarray(ee_pose, dtype=np.float64), np.asarray(tool_from_ee, dtype=np.float64)


def _rotation_error(a: np.ndarray, b: np.ndarray) -> float:
    r = a[:3, :3].T @ b[:3, :3]
    cos_angle = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_angle))


def _translation_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3]))


def _clip_to_joint_ranges(model: mujoco.MjModel, q: np.ndarray) -> np.ndarray:
    clipped = np.asarray(q, dtype=np.float64).copy()
    for i in range(6):
        joint_id = model.joint(f"joint{i + 1}").id
        lo, hi = model.jnt_range[joint_id]
        clipped[i] = np.clip(clipped[i], lo + 1e-4, hi - 1e-4)
    return clipped


def _sample_qs(model: mujoco.MjModel, q: np.ndarray) -> list[np.ndarray]:
    home_id = model.key("home").id
    home_q = np.asarray(model.key_qpos[home_id][:6], dtype=np.float64)
    deltas = [
        np.zeros(6, dtype=np.float64),
        np.array([0.25, -0.20, 0.15, 0.35, -0.25, 0.20], dtype=np.float64),
        np.array([-0.20, 0.10, -0.18, -0.30, 0.22, -0.15], dtype=np.float64),
    ]
    samples = [np.asarray(q, dtype=np.float64), home_q]
    samples.extend(np.asarray(q, dtype=np.float64) + delta for delta in deltas[1:])
    return [_clip_to_joint_ranges(model, sample) for sample in samples]


def _summarize_relative_transform_consistency(
    label: str,
    rels: list[np.ndarray],
) -> None:
    reference = rels[0]
    pos_errors = np.array([_translation_error(reference, rel) for rel in rels], dtype=np.float64)
    rot_errors = np.array([_rotation_error(reference, rel) for rel in rels], dtype=np.float64)
    translations = np.stack([rel[:3, 3] for rel in rels], axis=0)
    print(
        f"  {label:12s} "
        f"max_pos_delta={pos_errors.max():.9f} m "
        f"max_rot_delta={rot_errors.max():.9f} rad "
        f"translation_mean={translations.mean(axis=0).tolist()} "
        f"translation_std={translations.std(axis=0).tolist()}"
    )


def _relative_consistency_score(rels: list[np.ndarray]) -> tuple[float, float]:
    reference = rels[0]
    pos_errors = [_translation_error(reference, rel) for rel in rels]
    rot_errors = [_rotation_error(reference, rel) for rel in rels]
    return max(pos_errors), max(rot_errors)


def _search_curobo_q_signs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_samples: list[np.ndarray],
    cube_pos: np.ndarray,
    frame_getter,
) -> list[tuple[float, float, np.ndarray]]:
    results: list[tuple[float, float, np.ndarray]] = []
    for signs_tuple in product((-1.0, 1.0), repeat=6):
        signs = np.array(signs_tuple, dtype=np.float64)
        rels = []
        for q_sample in q_samples:
            _set_state(model, data, q_sample, cube_pos)
            sample_world_from_ee, _ = _curobo_ee_pose(q_sample * signs)
            rels.append(np.linalg.inv(frame_getter()) @ sample_world_from_ee)
        max_pos_delta, max_rot_delta = _relative_consistency_score(rels)
        results.append((max_pos_delta, max_rot_delta, signs))
    return sorted(results, key=lambda row: (row[1], row[0]))


def _print_pose(label: str, pose: np.ndarray) -> None:
    print(f"\n{label}")
    print(np.array2string(pose, precision=6, suppress_small=True))
    print(f"translation: {pose[:3, 3].tolist()}")


def _candidate_transforms(current_tool_from_ee: np.ndarray) -> dict[str, np.ndarray]:
    mujoco_site = np.eye(4, dtype=np.float64)
    mujoco_site[:3, :3] = _rz(-np.pi / 2.0)
    mujoco_site[:3, 3] = np.array([0.0, 0.0, 0.1347], dtype=np.float64)
    return {
        "current": current_tool_from_ee,
        "current-inverse": np.linalg.inv(current_tool_from_ee),
        "mujoco-site": mujoco_site,
        "mujoco-site-inverse": np.linalg.inv(mujoco_site),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--q",
        type=lambda s: _parse_vec(s, 6, "q"),
        default=DEFAULT_Q_CAPTURE,
        help="Six comma-separated YAM arm joint positions.",
    )
    parser.add_argument("--fovy", type=float, default=24.0)
    parser.add_argument(
        "--camera-pos",
        type=lambda s: _parse_vec(s, 3, "camera-pos"),
        default=DEFAULT_CAMERA_POS,
    )
    parser.add_argument(
        "--camera-target",
        type=lambda s: _parse_vec(s, 3, "camera-target"),
        default=DEFAULT_CAMERA_TARGET,
    )
    parser.add_argument(
        "--cube-pos",
        type=lambda s: _parse_vec(s, 3, "cube-pos"),
        default=DEFAULT_CUBE_POS,
    )
    parser.add_argument(
        "--curobo-q-signs",
        type=lambda s: _parse_vec(s, 6, "curobo-q-signs"),
        default=np.ones(6, dtype=np.float64),
        help="Signs applied to the MuJoCo q vector before evaluating cuRobo FK.",
    )
    args = parser.parse_args()

    model = _load_model(args.camera_pos, args.camera_target, args.cube_pos, args.fovy)
    data = mujoco.MjData(model)
    _set_state(model, data, args.q, args.cube_pos)

    world_from_link6 = _pose_from_body(model, data, "link_6")
    world_from_tcp = _pose_from_site(model, data, "tcp_site")
    world_from_grasp = _pose_from_site(model, data, "grasp_site")
    world_from_ee, current_tool_from_ee = _curobo_ee_pose(args.q * args.curobo_q_signs)

    _print_pose("MuJoCo world_from_link_6", world_from_link6)
    _print_pose("MuJoCo world_from_tcp_site", world_from_tcp)
    _print_pose("MuJoCo world_from_grasp_site", world_from_grasp)
    _print_pose("cuRobo world_from_ee_link", world_from_ee)
    _print_pose("Current cuTAMP RobotContainer.tool_from_ee", current_tool_from_ee)

    required_for_cutamp_code = np.linalg.inv(world_from_grasp) @ world_from_ee
    _print_pose(
        "Required tool_from_ee if cuTAMP uses world_from_ee = world_from_grasp @ tool_from_ee",
        required_for_cutamp_code,
    )

    frame_getters = {
        "link_6": lambda: _pose_from_body(model, data, "link_6"),
        "tcp_site": lambda: _pose_from_site(model, data, "tcp_site"),
        "grasp_site": lambda: _pose_from_site(model, data, "grasp_site"),
    }
    relative_by_frame: dict[str, list[np.ndarray]] = {name: [] for name in frame_getters}
    q_samples = _sample_qs(model, args.q)
    print("\nConstant-frame check across several joint configurations:")
    for sample_idx, q_sample in enumerate(q_samples):
        _set_state(model, data, q_sample, args.cube_pos)
        sample_world_from_ee, _ = _curobo_ee_pose(q_sample * args.curobo_q_signs)
        print(f"  sample {sample_idx}: q={q_sample.tolist()}")
        for frame_name, frame_getter in frame_getters.items():
            world_from_frame = frame_getter()
            relative_by_frame[frame_name].append(np.linalg.inv(world_from_frame) @ sample_world_from_ee)

    print("\nFrame_from_curobo_ee consistency; near zero deltas mean the two FK models agree:")
    for frame_name, rels in relative_by_frame.items():
        _summarize_relative_transform_consistency(frame_name, rels)

    print("\nMean relative transforms if cuTAMP uses world_from_ee = world_from_TOOL @ TOOL_from_ee:")
    for frame_name, rels in relative_by_frame.items():
        mean_rel = rels[0].copy()
        mean_rel[:3, 3] = np.stack([rel[:3, 3] for rel in rels], axis=0).mean(axis=0)
        _print_pose(f"{frame_name}_from_curobo_ee", mean_rel)

    best_signs = _search_curobo_q_signs(
        model,
        data,
        q_samples,
        args.cube_pos,
        frame_getters["grasp_site"],
    )
    print("\nBest cuRobo q sign mappings using MuJoCo grasp_site as the tool frame:")
    for max_pos_delta, max_rot_delta, signs in best_signs[:5]:
        print(
            f"  signs={signs.tolist()} "
            f"max_pos_delta={max_pos_delta:.9f} m "
            f"max_rot_delta={max_rot_delta:.9f} rad"
        )

    print("\nCandidate error if desired tool frame is MuJoCo grasp_site:")
    for name, candidate in _candidate_transforms(current_tool_from_ee).items():
        predicted_world_from_ee = world_from_grasp @ candidate
        print(
            f"  {name:20s} "
            f"ee_pos_err={_translation_error(predicted_world_from_ee, world_from_ee):.6f} m "
            f"ee_rot_err={_rotation_error(predicted_world_from_ee, world_from_ee):.6f} rad"
        )


if __name__ == "__main__":
    main()
