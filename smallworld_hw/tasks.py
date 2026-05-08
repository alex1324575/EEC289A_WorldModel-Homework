"""Analytic SmallWorld-Lite physics tasks.

The original SmallWorld paper uses MuJoCo scenes to isolate physical laws. This
course version keeps the same evaluation idea but implements lightweight,
deterministic state-space systems so the benchmark is easy to run in Colab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


def _clip_action(action: Array, action_dim: int) -> Array:
    return np.clip(np.asarray(action, dtype=np.float32).reshape(action_dim), -1.0, 1.0)


def _angle_features(angle: float) -> tuple[float, float]:
    return float(np.sin(angle)), float(np.cos(angle))


@dataclass(frozen=True)
class SmallWorldTask:
    name: str
    state_dim: int
    action_dim: int
    param_dim: int
    dt: float
    reset_fn: Callable[[np.random.Generator, bool], tuple[Array, Array]]
    step_fn: Callable[[Array, Array, Array, float], Array]
    position_fn: Callable[[Array], Array]
    energy_fn: Callable[[Array, Array], float] | None = None
    constraint_fn: Callable[[Array, Array], float] | None = None

    def reset(self, rng: np.random.Generator, *, ood: bool = False) -> tuple[Array, Array]:
        state, params = self.reset_fn(rng, ood)
        return state.astype(np.float32), params.astype(np.float32)

    def step(self, state: Array, action: Array, params: Array) -> Array:
        action = _clip_action(action, self.action_dim)
        next_state = self.step_fn(
            np.asarray(state, dtype=np.float32).reshape(self.state_dim),
            action,
            np.asarray(params, dtype=np.float32).reshape(self.param_dim),
            self.dt,
        )
        return np.asarray(next_state, dtype=np.float32).reshape(self.state_dim)

    def position(self, state: Array) -> Array:
        return np.asarray(self.position_fn(np.asarray(state, dtype=np.float32)), dtype=np.float32)

    def energy(self, state: Array, params: Array) -> float:
        if self.energy_fn is None:
            return float("nan")
        return float(self.energy_fn(np.asarray(state, dtype=np.float32), np.asarray(params, dtype=np.float32)))

    def constraint(self, state: Array, params: Array) -> float:
        if self.constraint_fn is None:
            return 0.0
        return float(self.constraint_fn(np.asarray(state, dtype=np.float32), np.asarray(params, dtype=np.float32)))


def _reset_free_fall(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    g = rng.uniform(12.0, 15.0) if ood else rng.uniform(8.5, 10.5)
    restitution = rng.uniform(0.65, 0.85) if ood else rng.uniform(0.85, 0.98)
    state = np.array([rng.uniform(-1.0, 1.0), rng.uniform(2.0, 4.0), rng.uniform(-0.4, 0.4), rng.uniform(-0.2, 0.5)])
    return state, np.array([g, restitution])


def _step_free_fall(state: Array, action: Array, params: Array, dt: float) -> Array:
    x, y, vx, vy = state
    g, restitution = params
    vx = vx + 0.5 * action[0] * dt
    vy = vy + (0.5 * action[1] - g) * dt
    x = x + vx * dt
    y = y + vy * dt
    if y < 0.0:
        y = -y
        vy = abs(vy) * restitution
    return np.array([x, y, vx, vy])


def _energy_gravity_2d(state: Array, params: Array) -> float:
    g = float(params[0])
    return 0.5 * float(state[2] ** 2 + state[3] ** 2) + g * max(float(state[1]), 0.0)


def _reset_projectile(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    g = rng.uniform(12.0, 14.0) if ood else rng.uniform(8.5, 10.5)
    state = np.array([rng.uniform(-1.0, 0.5), rng.uniform(0.4, 1.4), rng.uniform(1.0, 2.2), rng.uniform(1.0, 2.5)])
    return state, np.array([g])


def _step_projectile(state: Array, action: Array, params: Array, dt: float) -> Array:
    x, y, vx, vy = state
    g = params[0]
    vx = vx + 0.2 * action[0] * dt
    vy = vy + (0.2 * action[1] - g) * dt
    x = x + vx * dt
    y = y + vy * dt
    if y < -1.0:
        y = -1.0
        vy = 0.0
    return np.array([x, y, vx, vy])


def _reset_circular(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    radius = rng.uniform(1.4, 1.8) if ood else rng.uniform(0.8, 1.2)
    angle = rng.uniform(-np.pi, np.pi)
    speed = rng.uniform(1.0, 1.6)
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    vx, vy = -speed * np.sin(angle), speed * np.cos(angle)
    return np.array([x, y, vx, vy]), np.array([radius])


def _step_circular(state: Array, action: Array, params: Array, dt: float) -> Array:
    radius = max(float(params[0]), 1e-3)
    angle = np.arctan2(state[1], state[0])
    tangent = np.array([-np.sin(angle), np.cos(angle)])
    speed = float(np.dot(state[2:4], tangent))
    speed = speed + 0.6 * float(action[0]) * dt
    angle = angle + speed / radius * dt
    x, y = radius * np.cos(angle), radius * np.sin(angle)
    vx, vy = -speed * np.sin(angle), speed * np.cos(angle)
    return np.array([x, y, vx, vy])


def _constraint_circular(state: Array, params: Array) -> float:
    return abs(float(np.linalg.norm(state[:2]) - params[0]))


def _reset_inclined(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    angle = rng.uniform(0.6, 0.9) if ood else rng.uniform(0.25, 0.55)
    friction = rng.uniform(0.04, 0.08) if ood else rng.uniform(0.0, 0.04)
    state = np.array([rng.uniform(0.0, 0.5), rng.uniform(-0.2, 0.2)])
    return state, np.array([angle, friction, 9.81])


def _step_inclined(state: Array, action: Array, params: Array, dt: float) -> Array:
    s, v = state
    angle, friction, g = params
    friction_acc = friction * g * np.cos(angle) * np.tanh(6.0 * v)
    acc = g * np.sin(angle) - friction_acc + 0.8 * action[0]
    v = v + acc * dt
    s = s + v * dt
    return np.array([s, v])


def _reset_pendulum(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    g = rng.uniform(12.0, 14.0) if ood else rng.uniform(8.5, 10.5)
    length = rng.uniform(0.7, 0.9) if ood else rng.uniform(0.95, 1.2)
    damping = rng.uniform(0.04, 0.08) if ood else rng.uniform(0.01, 0.04)
    theta = rng.uniform(-1.2, 1.2)
    omega = rng.uniform(-0.8, 0.8)
    s, c = _angle_features(theta)
    return np.array([s, c, omega]), np.array([g, length, damping])


def _step_pendulum(state: Array, action: Array, params: Array, dt: float) -> Array:
    theta = np.arctan2(state[0], state[1])
    omega = float(state[2])
    g, length, damping = params
    acc = -(g / length) * np.sin(theta) - damping * omega + 1.4 * action[0]
    omega = omega + acc * dt
    theta = theta + omega * dt
    s, c = _angle_features(theta)
    return np.array([s, c, omega])


def _energy_pendulum(state: Array, params: Array) -> float:
    theta = np.arctan2(state[0], state[1])
    omega = float(state[2])
    g, length, _damping = params
    return 0.5 * (length * omega) ** 2 + g * length * (1.0 - np.cos(theta))


def _reset_rolling(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    radius = rng.uniform(0.45, 0.65) if ood else rng.uniform(0.25, 0.45)
    damping = rng.uniform(0.03, 0.06) if ood else rng.uniform(0.0, 0.025)
    omega = rng.uniform(1.0, 2.2)
    return np.array([0.0, radius * omega, rng.uniform(-np.pi, np.pi), omega]), np.array([radius, damping])


def _step_rolling(state: Array, action: Array, params: Array, dt: float) -> Array:
    x, _v, angle, omega = state
    radius, damping = params
    omega = omega + (1.0 * action[0] - damping * omega) * dt
    v = radius * omega
    x = x + v * dt
    angle = angle + omega * dt
    return np.array([x, v, angle, omega])


def _energy_rolling(state: Array, _params: Array) -> float:
    v = float(state[1])
    omega = float(state[3])
    return 0.5 * v * v + 0.25 * omega * omega


def _constraint_rolling(state: Array, params: Array) -> float:
    return abs(float(state[1] - params[0] * state[3]))


def _reset_rotation(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    damping = rng.uniform(0.02, 0.04) if ood else rng.uniform(0.0, 0.015)
    phi = rng.uniform(-np.pi, np.pi)
    omega = rng.uniform(-2.0, 2.0)
    s, c = _angle_features(phi)
    return np.array([s, c, omega]), np.array([damping])


def _step_rotation(state: Array, action: Array, params: Array, dt: float) -> Array:
    phi = np.arctan2(state[0], state[1])
    omega = float(state[2])
    damping = float(params[0])
    omega = omega + (1.2 * action[0] - damping * omega) * dt
    phi = phi + omega * dt
    s, c = _angle_features(phi)
    return np.array([s, c, omega])


def _energy_rotation(state: Array, _params: Array) -> float:
    return 0.5 * float(state[2] ** 2)


def _reset_spin(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    damping = rng.uniform(0.08, 0.12) if ood else rng.uniform(0.03, 0.07)
    instability = rng.uniform(0.35, 0.55) if ood else rng.uniform(0.15, 0.3)
    yaw = rng.uniform(-np.pi, np.pi)
    yaw_rate = rng.uniform(5.0, 8.0)
    tilt = rng.uniform(0.02, 0.08)
    tilt_rate = rng.uniform(-0.03, 0.03)
    s, c = _angle_features(yaw)
    return np.array([s, c, yaw_rate, tilt, tilt_rate]), np.array([damping, instability])


def _step_spin(state: Array, action: Array, params: Array, dt: float) -> Array:
    yaw = np.arctan2(state[0], state[1])
    yaw_rate = float(state[2])
    tilt = float(state[3])
    tilt_rate = float(state[4])
    damping, instability = params
    yaw_rate = yaw_rate + (0.8 * action[0] - damping * yaw_rate) * dt
    tilt_acc = instability * tilt + 0.012 * max(8.0 - yaw_rate, 0.0) - 0.15 * tilt_rate
    tilt_rate = tilt_rate + tilt_acc * dt
    tilt = np.clip(tilt + tilt_rate * dt, 0.0, 1.4)
    yaw = yaw + yaw_rate * dt
    s, c = _angle_features(yaw)
    return np.array([s, c, yaw_rate, tilt, tilt_rate])


def _reset_elastic(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    restitution = rng.uniform(0.65, 0.85) if ood else rng.uniform(0.92, 1.0)
    x1 = rng.uniform(-1.2, -0.5)
    x2 = rng.uniform(0.5, 1.2)
    v1 = rng.uniform(0.6, 1.3)
    v2 = rng.uniform(-1.1, -0.4)
    return np.array([x1, v1, x2, v2]), np.array([restitution])


def _step_elastic(state: Array, action: Array, params: Array, dt: float) -> Array:
    x1, v1, x2, v2 = state
    restitution = float(params[0])
    v1 = v1 + 0.25 * action[0] * dt
    v2 = v2 + 0.25 * action[1] * dt
    x1 = x1 + v1 * dt
    x2 = x2 + v2 * dt
    if x1 >= x2 and v1 > v2:
        center = 0.5 * (x1 + x2)
        x1 = center - 1e-3
        x2 = center + 1e-3
        new_v1 = 0.5 * ((1.0 - restitution) * v1 + (1.0 + restitution) * v2)
        new_v2 = 0.5 * ((1.0 + restitution) * v1 + (1.0 - restitution) * v2)
        v1, v2 = new_v1, new_v2
    return np.array([x1, v1, x2, v2])


def _energy_elastic(state: Array, _params: Array) -> float:
    return 0.5 * float(state[1] ** 2 + state[3] ** 2)


def _position_elastic(state: Array) -> Array:
    return np.array([state[0], state[2]])


def _reset_bouncing(rng: np.random.Generator, ood: bool) -> tuple[Array, Array]:
    restitution = rng.uniform(0.65, 0.85) if ood else rng.uniform(0.9, 1.0)
    box_size = rng.uniform(1.5, 1.8) if ood else rng.uniform(1.0, 1.3)
    state = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.6, 1.4), rng.uniform(-1.2, 1.2)])
    return state, np.array([restitution, box_size])


def _step_bouncing(state: Array, action: Array, params: Array, dt: float) -> Array:
    x, y, vx, vy = state
    restitution, box_size = params
    vx = vx + 0.4 * action[0] * dt
    vy = vy + 0.4 * action[1] * dt
    x = x + vx * dt
    y = y + vy * dt
    half = float(box_size)
    if x > half:
        x = 2 * half - x
        vx = -abs(vx) * restitution
    elif x < -half:
        x = -2 * half - x
        vx = abs(vx) * restitution
    if y > half:
        y = 2 * half - y
        vy = -abs(vy) * restitution
    elif y < -half:
        y = -2 * half - y
        vy = abs(vy) * restitution
    return np.array([x, y, vx, vy])


def _energy_bouncing(state: Array, _params: Array) -> float:
    return 0.5 * float(state[2] ** 2 + state[3] ** 2)


def _constraint_bouncing(state: Array, params: Array) -> float:
    half = float(params[1])
    return float(max(abs(state[0]) - half, abs(state[1]) - half, 0.0))


def _position_xy(state: Array) -> Array:
    return np.asarray(state[:2], dtype=np.float32)


def _position_line(state: Array) -> Array:
    return np.array([state[0], 0.0], dtype=np.float32)


def _position_angle(state: Array) -> Array:
    if state.shape[0] >= 3:
        angle = np.arctan2(state[0], state[1])
        return np.array([np.sin(angle), np.cos(angle)], dtype=np.float32)
    return np.array([state[0], 0.0], dtype=np.float32)


TASKS: dict[str, SmallWorldTask] = {
    "free_fall": SmallWorldTask("free_fall", 4, 2, 2, 0.04, _reset_free_fall, _step_free_fall, _position_xy, _energy_gravity_2d),
    "projectile": SmallWorldTask("projectile", 4, 2, 1, 0.04, _reset_projectile, _step_projectile, _position_xy, _energy_gravity_2d),
    "circular_motion": SmallWorldTask("circular_motion", 4, 1, 1, 0.04, _reset_circular, _step_circular, _position_xy, None, _constraint_circular),
    "inclined_plane": SmallWorldTask("inclined_plane", 2, 1, 3, 0.04, _reset_inclined, _step_inclined, _position_line),
    "simple_pendulum": SmallWorldTask("simple_pendulum", 3, 1, 3, 0.04, _reset_pendulum, _step_pendulum, _position_angle, _energy_pendulum),
    "rolling": SmallWorldTask("rolling", 4, 1, 2, 0.04, _reset_rolling, _step_rolling, _position_line, _energy_rolling, _constraint_rolling),
    "rotation": SmallWorldTask("rotation", 3, 1, 1, 0.04, _reset_rotation, _step_rotation, _position_angle, _energy_rotation),
    "spin": SmallWorldTask("spin", 5, 1, 2, 0.04, _reset_spin, _step_spin, _position_angle),
    "elastic_collision": SmallWorldTask("elastic_collision", 4, 2, 1, 0.04, _reset_elastic, _step_elastic, _position_elastic, _energy_elastic),
    "bouncing_ball": SmallWorldTask("bouncing_ball", 4, 2, 2, 0.04, _reset_bouncing, _step_bouncing, _position_xy, _energy_bouncing, _constraint_bouncing),
}


def list_tasks() -> list[str]:
    return sorted(TASKS)


def get_task(name: str) -> SmallWorldTask:
    if name not in TASKS:
        raise KeyError(f"Unknown SmallWorld task '{name}'. Available: {', '.join(list_tasks())}")
    return TASKS[name]
