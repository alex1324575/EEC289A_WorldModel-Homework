"""Small helpers around the official MuJoCo Python package."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


def set_default_gl() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")


def model_from_xml(xml_path: str | Path) -> mujoco.MjModel:
    set_default_gl()
    return mujoco.MjModel.from_xml_path(str(xml_path))


def reset_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    return data


def body_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))


def joint_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))


def qpos_addr(model: mujoco.MjModel, joint: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, joint)])


def qvel_addr(model: mujoco.MjModel, joint: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, joint)])


def quat_to_euler_like(q: np.ndarray) -> np.ndarray:
    """Return a compact continuous orientation feature from a quaternion."""
    q = np.asarray(q, dtype=np.float32)
    return q / max(float(np.linalg.norm(q)), 1e-6)
