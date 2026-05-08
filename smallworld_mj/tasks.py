"""Frozen SmallWorld-MJ task specs.

The XML scenes and task specs are staff-owned benchmark code. Students train
world models on the generated state/action/parameter datasets; they should not
author MuJoCo scenes as part of the required assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .mujoco_utils import body_id, qpos_addr, qvel_addr, quat_to_euler_like


Array = np.ndarray
ROOT = Path(__file__).resolve().parent
XML_DIR = ROOT / "assets" / "xml"


ResetFn = Callable[[mujoco.MjModel, mujoco.MjData, np.random.Generator, Array], None]
ParamFn = Callable[[mujoco.MjModel, Array], None]
ActionFn = Callable[[mujoco.MjModel, mujoco.MjData, Array], None]
StateFn = Callable[[mujoco.MjModel, mujoco.MjData], Array]
PositionFn = Callable[[Array], Array]


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    name: str
    xml: str
    state_dim: int
    action_dim: int
    param_dim: int
    param_names: tuple[str, ...]
    id_ranges: tuple[tuple[float, float], ...]
    ood_ranges: tuple[tuple[float, float], ...]
    reset_fn: ResetFn
    param_fn: ParamFn
    action_fn: ActionFn
    state_fn: StateFn
    position_fn: PositionFn

    @property
    def xml_path(self) -> Path:
        return XML_DIR / self.xml

    def sample_params(self, rng: np.random.Generator, *, ood: bool = False) -> Array:
        ranges = self.ood_ranges if ood else self.id_ranges
        return np.asarray([rng.uniform(lo, hi) for lo, hi in ranges], dtype=np.float32)

    def position(self, state: Array) -> Array:
        return np.asarray(self.position_fn(np.asarray(state, dtype=np.float32)), dtype=np.float32)


def _set_gravity(model: mujoco.MjModel, params: Array) -> None:
    model.opt.gravity[:] = np.array([0.0, 0.0, -float(params[0])], dtype=np.float64)


def _set_none(_model: mujoco.MjModel, _params: Array) -> None:
    return None


def _set_hinge_damping(model: mujoco.MjModel, params: Array) -> None:
    if model.nv:
        model.dof_damping[:] = float(params[-1])


def _set_gravity_damping(model: mujoco.MjModel, params: Array) -> None:
    _set_gravity(model, params)
    if model.nv:
        model.dof_damping[:] = float(params[-1])


def _set_friction(model: mujoco.MjModel, params: Array) -> None:
    friction = float(params[-1])
    if model.ngeom:
        model.geom_friction[:, 0] = friction


def _free_q(model: mujoco.MjModel, joint: str) -> tuple[int, int]:
    return qpos_addr(model, joint), qvel_addr(model, joint)


def _reset_free_ball(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    params: Array,
    *,
    projectile: bool,
    box: bool = False,
) -> None:
    qpos, qvel = _free_q(model, "ball_free")
    data.qpos[qpos : qpos + 3] = np.array(
        [
            rng.uniform(-0.5, 0.5) if box else rng.uniform(-1.0, 0.2),
            rng.uniform(-0.5, 0.5) if box else rng.uniform(-0.2, 0.2),
            0.16 if box else rng.uniform(1.2, 3.2),
        ]
    )
    data.qpos[qpos + 3 : qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    if box:
        data.qvel[qvel : qvel + 3] = np.array([rng.uniform(0.6, 1.4), rng.uniform(-1.2, 1.2), 0.0])
    elif projectile:
        data.qvel[qvel : qvel + 3] = np.array([rng.uniform(1.0, 2.4), rng.uniform(-0.2, 0.2), rng.uniform(1.0, 2.2)])
    else:
        data.qvel[qvel : qvel + 3] = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2), rng.uniform(-0.1, 0.4)])
    data.qvel[qvel + 3 : qvel + 6] = 0.0


def _reset_free_fall(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, params: Array) -> None:
    _reset_free_ball(model, data, rng, params, projectile=False)


def _reset_projectile(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, params: Array) -> None:
    _reset_free_ball(model, data, rng, params, projectile=True)


def _reset_bouncing(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, params: Array) -> None:
    _reset_free_ball(model, data, rng, params, projectile=False, box=True)


def _reset_inclined(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, params: Array) -> None:
    qpos, qvel = _free_q(model, "ball_free")
    data.qpos[qpos : qpos + 3] = np.array([rng.uniform(-1.6, -1.0), 0.0, rng.uniform(0.9, 1.2)])
    data.qpos[qpos + 3 : qpos + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[qvel : qvel + 3] = np.array([rng.uniform(0.0, 0.4), 0.0, rng.uniform(-0.1, 0.1)])
    data.qvel[qvel + 3 : qvel + 6] = 0.0


def _reset_hinge(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, _params: Array) -> None:
    qpos, qvel = qpos_addr(model, "hinge"), qvel_addr(model, "hinge")
    data.qpos[qpos] = rng.uniform(-1.3, 1.3)
    data.qvel[qvel] = rng.uniform(-1.2, 1.2)


def _reset_circular(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, _params: Array) -> None:
    qpos, qvel = qpos_addr(model, "hinge"), qvel_addr(model, "hinge")
    data.qpos[qpos] = rng.uniform(-np.pi, np.pi)
    data.qvel[qvel] = rng.uniform(0.8, 1.8)


def _reset_rolling(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, params: Array) -> None:
    sx_q, sx_v = qpos_addr(model, "slide_x"), qvel_addr(model, "slide_x")
    roll_q, roll_v = qpos_addr(model, "roll"), qvel_addr(model, "roll")
    radius = float(params[0])
    omega = rng.uniform(1.0, 2.2)
    data.qpos[sx_q] = rng.uniform(-0.4, 0.4)
    data.qpos[roll_q] = rng.uniform(-np.pi, np.pi)
    data.qvel[sx_v] = radius * omega
    data.qvel[roll_v] = omega


def _reset_elastic(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, _params: Array) -> None:
    q1, v1 = _free_q(model, "ball1_free")
    q2, v2 = _free_q(model, "ball2_free")
    data.qpos[q1 : q1 + 3] = np.array([rng.uniform(-1.4, -0.8), 0.0, 0.13])
    data.qpos[q2 : q2 + 3] = np.array([rng.uniform(0.8, 1.4), 0.0, 0.13])
    data.qpos[q1 + 3 : q1 + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[q2 + 3 : q2 + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[v1 : v1 + 3] = np.array([rng.uniform(0.8, 1.5), 0.0, 0.0])
    data.qvel[v2 : v2 + 3] = np.array([rng.uniform(-1.5, -0.8), 0.0, 0.0])
    data.qvel[v1 + 3 : v1 + 6] = 0.0
    data.qvel[v2 + 3 : v2 + 6] = 0.0


def _reset_rolling_hinge(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, _params: Array) -> None:
    _reset_hinge(model, data, rng, _params)


def _reset_spin(model: mujoco.MjModel, data: mujoco.MjData, rng: np.random.Generator, _params: Array) -> None:
    qpos, qvel = _free_q(model, "top_free")
    data.qpos[qpos : qpos + 3] = np.array([0.0, 0.0, 0.45])
    data.qpos[qpos + 3 : qpos + 7] = np.array([0.998, rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), 0.0])
    data.qvel[qvel : qvel + 3] = 0.0
    data.qvel[qvel + 3 : qvel + 6] = np.array([rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(5.0, 8.0)])


def _force_body(body: str, scale: float) -> ActionFn:
    def apply(model: mujoco.MjModel, data: mujoco.MjData, action: Array) -> None:
        data.xfrc_applied[:] = 0.0
        force = np.zeros(3)
        n = min(3, len(action))
        force[:n] = np.asarray(action[:n], dtype=np.float64) * scale
        data.xfrc_applied[body_id(model, body), :3] = force

    return apply


def _force_two_balls(scale: float) -> ActionFn:
    def apply(model: mujoco.MjModel, data: mujoco.MjData, action: Array) -> None:
        data.xfrc_applied[:] = 0.0
        data.xfrc_applied[body_id(model, "ball1"), 0] = float(action[0]) * scale
        data.xfrc_applied[body_id(model, "ball2"), 0] = float(action[1]) * scale

    return apply


def _ctrl(scale: float) -> ActionFn:
    def apply(_model: mujoco.MjModel, data: mujoco.MjData, action: Array) -> None:
        if data.ctrl.size:
            data.ctrl[:] = np.asarray(action[: data.ctrl.size], dtype=np.float64) * scale

    return apply


def _free_state(joint: str) -> StateFn:
    def state(model: mujoco.MjModel, data: mujoco.MjData) -> Array:
        qpos, qvel = _free_q(model, joint)
        return np.concatenate([data.qpos[qpos : qpos + 3], data.qvel[qvel : qvel + 3]]).astype(np.float32)

    return state


def _two_ball_state(model: mujoco.MjModel, data: mujoco.MjData) -> Array:
    q1, v1 = _free_q(model, "ball1_free")
    q2, v2 = _free_q(model, "ball2_free")
    return np.concatenate(
        [data.qpos[q1 : q1 + 3], data.qvel[v1 : v1 + 3], data.qpos[q2 : q2 + 3], data.qvel[v2 : v2 + 3]]
    ).astype(np.float32)


def _hinge_state(model: mujoco.MjModel, data: mujoco.MjData) -> Array:
    qpos, qvel = qpos_addr(model, "hinge"), qvel_addr(model, "hinge")
    angle = float(data.qpos[qpos])
    return np.asarray([np.sin(angle), np.cos(angle), data.qvel[qvel]], dtype=np.float32)


def _rolling_state(model: mujoco.MjModel, data: mujoco.MjData) -> Array:
    sx_q, sx_v = qpos_addr(model, "slide_x"), qvel_addr(model, "slide_x")
    roll_q, roll_v = qpos_addr(model, "roll"), qvel_addr(model, "roll")
    angle = float(data.qpos[roll_q])
    return np.asarray([data.qpos[sx_q], data.qvel[sx_v], np.sin(angle), np.cos(angle), data.qvel[roll_v]], dtype=np.float32)


def _spin_state(model: mujoco.MjModel, data: mujoco.MjData) -> Array:
    qpos, qvel = _free_q(model, "top_free")
    quat = quat_to_euler_like(data.qpos[qpos + 3 : qpos + 7])
    return np.concatenate([quat, data.qvel[qvel + 3 : qvel + 6]]).astype(np.float32)


def _pos_xyz(state: Array) -> Array:
    return np.asarray(state[:3], dtype=np.float32)


def _pos_hinge(state: Array) -> Array:
    return np.asarray([state[0], 0.0, -state[1]], dtype=np.float32)


def _pos_circle(state: Array) -> Array:
    return np.asarray([state[1], state[0], 0.0], dtype=np.float32)


def _pos_rolling(state: Array) -> Array:
    return np.asarray([state[0], 0.0, 0.0], dtype=np.float32)


def _pos_elastic(state: Array) -> Array:
    return np.asarray([state[0], state[6], 0.0], dtype=np.float32)


def _pos_spin(state: Array) -> Array:
    return np.asarray([state[1], state[2], state[3]], dtype=np.float32)


_SPECS = [
    TaskSpec(0, "simple_pendulum", "simple_pendulum.xml", 3, 1, 3, ("gravity", "length", "damping"), ((8.5, 10.5), (0.9, 1.1), (0.0, 0.04)), ((11.0, 13.0), (1.2, 1.5), (0.05, 0.1)), _reset_hinge, _set_gravity_damping, _ctrl(1.5), _hinge_state, _pos_hinge),
    TaskSpec(1, "projectile", "projectile.xml", 6, 3, 1, ("gravity",), ((8.5, 10.5),), ((11.0, 13.0),), _reset_projectile, _set_gravity, _force_body("ball", 1.0), _free_state("ball_free"), _pos_xyz),
    TaskSpec(2, "circular_motion", "circular_motion.xml", 3, 1, 2, ("radius", "damping"), ((0.9, 1.1), (0.0, 0.02)), ((1.2, 1.5), (0.03, 0.08)), _reset_circular, _set_hinge_damping, _ctrl(1.2), _hinge_state, _pos_circle),
    TaskSpec(3, "bouncing_ball", "bouncing_ball.xml", 6, 3, 2, ("restitution", "box_size"), ((0.9, 1.0), (1.15, 1.25)), ((0.65, 0.85), (1.35, 1.55)), _reset_bouncing, _set_none, _force_body("ball", 0.7), _free_state("ball_free"), _pos_xyz),
    TaskSpec(4, "rolling", "rolling.xml", 5, 2, 2, ("radius", "friction"), ((0.22, 0.28), (0.8, 1.2)), ((0.32, 0.42), (0.4, 0.7)), _reset_rolling, _set_friction, _ctrl(0.8), _rolling_state, _pos_rolling),
    TaskSpec(5, "free_fall", "free_fall.xml", 6, 3, 2, ("gravity", "restitution"), ((8.5, 10.5), (0.9, 1.0)), ((11.0, 13.0), (0.65, 0.85)), _reset_free_fall, _set_gravity, _force_body("ball", 0.4), _free_state("ball_free"), _pos_xyz),
    TaskSpec(6, "inclined_plane", "inclined_plane.xml", 6, 3, 3, ("gravity", "slope", "friction"), ((8.5, 10.5), (0.35, 0.5), (0.2, 0.5)), ((11.0, 13.0), (0.55, 0.75), (0.6, 1.0)), _reset_inclined, _set_gravity, _force_body("ball", 0.5), _free_state("ball_free"), _pos_xyz),
    TaskSpec(7, "elastic_collision", "elastic_collision.xml", 12, 2, 1, ("restitution",), ((0.9, 1.0),), ((0.65, 0.85),), _reset_elastic, _set_none, _force_two_balls(0.6), _two_ball_state, _pos_elastic),
    TaskSpec(8, "rotation", "rotation.xml", 3, 1, 1, ("damping",), ((0.0, 0.02),), ((0.04, 0.08),), _reset_rolling_hinge, _set_hinge_damping, _ctrl(1.0), _hinge_state, _pos_circle),
    TaskSpec(9, "spin", "spin.xml", 7, 3, 2, ("damping", "tilt"), ((0.01, 0.04), (0.02, 0.08)), ((0.08, 0.14), (0.1, 0.2)), _reset_spin, _set_none, _force_body("top", 0.4), _spin_state, _pos_spin),
]

TASKS = {spec.name: spec for spec in _SPECS}
TASKPACKS = {
    "smallworld_all": [spec.name for spec in _SPECS],
    "pendulum_only": ["simple_pendulum"],
    "smallworld_core": ["simple_pendulum", "projectile", "circular_motion", "bouncing_ball", "rolling"],
}


def list_tasks() -> list[str]:
    return [spec.name for spec in _SPECS]


def get_task(name: str) -> TaskSpec:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(list_tasks())}")
    return TASKS[name]


def taskpack(name: str) -> list[str]:
    if name in TASKPACKS:
        return list(TASKPACKS[name])
    if name == "all":
        return list_tasks()
    return [get_task(name).name]


def max_dims(tasks: list[str] | None = None) -> tuple[int, int, int]:
    specs = [get_task(t) for t in (tasks or list_tasks())]
    return max(s.state_dim for s in specs), max(s.action_dim for s in specs), max(s.param_dim for s in specs)
