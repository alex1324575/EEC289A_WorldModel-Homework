import numpy as np

from smallworld_hw.dataset import generate_split
from smallworld_hw.tasks import get_task, list_tasks


def test_all_smallworld_tasks_are_deterministic_and_shaped():
    for name in list_tasks():
        task = get_task(name)
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        state1, params1 = task.reset(rng1)
        state2, params2 = task.reset(rng2)
        action = np.linspace(-0.3, 0.4, task.action_dim, dtype=np.float32)
        next1 = task.step(state1, action, params1)
        next2 = task.step(state2, action, params2)
        assert state1.shape == (task.state_dim,)
        assert params1.shape == (task.param_dim,)
        assert next1.shape == (task.state_dim,)
        np.testing.assert_allclose(state1, state2)
        np.testing.assert_allclose(params1, params2)
        np.testing.assert_allclose(next1, next2)


def test_smallworld_dataset_schema():
    task = get_task("simple_pendulum")
    data = generate_split(task, split="train", episodes=3, steps=12, seed=5)
    assert data["states"].shape == (3, 13, task.state_dim)
    assert data["state"].shape == (3, 12, task.state_dim)
    assert data["next_state"].shape == (3, 12, task.state_dim)
    assert data["action"].shape == (3, 12, task.action_dim)
    assert data["task_params"].shape == (3, task.param_dim)
    np.testing.assert_allclose(data["state"][:, 1:], data["next_state"][:, :-1])
