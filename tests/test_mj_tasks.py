from __future__ import annotations

import numpy as np

from smallworld_mj.envs import SmallWorldMJEnv
from smallworld_mj.tasks import get_task, list_tasks, max_dims


def test_all_mujoco_xml_compile_and_step():
    assert len(list_tasks()) == 10
    for name in list_tasks():
        spec = get_task(name)
        env = SmallWorldMJEnv(spec)
        state = env.reset(np.random.default_rng(0))
        assert state.shape == (spec.state_dim,)
        result = env.step(np.zeros(spec.action_dim, dtype=np.float32))
        assert result.state.shape == (spec.state_dim,)


def test_max_dims_cover_all_tasks():
    s, a, p = max_dims()
    for name in list_tasks():
        spec = get_task(name)
        assert spec.state_dim <= s
        assert spec.action_dim <= a
        assert spec.param_dim <= p


def test_reset_determinism_for_same_seed():
    spec = get_task("projectile")
    env_a = SmallWorldMJEnv(spec)
    env_b = SmallWorldMJEnv(spec)
    state_a = env_a.reset(np.random.default_rng(123))
    state_b = env_b.reset(np.random.default_rng(123))
    np.testing.assert_allclose(state_a, state_b)
    np.testing.assert_allclose(env_a.params, env_b.params)
