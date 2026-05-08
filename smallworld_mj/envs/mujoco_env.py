"""Frozen MuJoCo environment wrapper for SmallWorld-MJ tasks."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from smallworld_mj.mujoco_utils import model_from_xml, reset_data
from smallworld_mj.tasks import TaskSpec


@dataclass
class StepResult:
    state: np.ndarray
    action: np.ndarray
    task_params: np.ndarray


class SmallWorldMJEnv:
    """A minimal MuJoCo wrapper with no reward or policy concepts."""

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.model = model_from_xml(spec.xml_path)
        self.data = reset_data(self.model)
        self.params = np.zeros(spec.param_dim, dtype=np.float32)

    def reset(self, rng: np.random.Generator, *, ood: bool = False, params: np.ndarray | None = None) -> np.ndarray:
        self.data = reset_data(self.model)
        self.params = np.asarray(params if params is not None else self.spec.sample_params(rng, ood=ood), dtype=np.float32)
        self.spec.param_fn(self.model, self.params)
        self.spec.reset_fn(self.model, self.data, rng, self.params)
        mujoco.mj_forward(self.model, self.data)
        return self.state()

    def state(self) -> np.ndarray:
        return self.spec.state_fn(self.model, self.data).astype(np.float32)

    def step(self, action: np.ndarray) -> StepResult:
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(self.spec.action_dim), -1.0, 1.0)
        self.spec.action_fn(self.model, self.data, action)
        mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        return StepResult(state=self.state(), action=action, task_params=self.params.copy())

    def render_rgb(self, width: int = 320, height: int = 240) -> np.ndarray:
        with mujoco.Renderer(self.model, height=height, width=width) as renderer:
            renderer.update_scene(self.data, camera="track")
            return renderer.render()
