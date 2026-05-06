import numpy as np

from world_model_hw.replay import EpisodeReplay


def test_replay_samples_sequences():
    replay = EpisodeReplay(capacity_steps=100, obs_dim=3, action_dim=1, seed=0)
    replay.start_episode(np.zeros(3, dtype=np.float32))
    for idx in range(10):
        replay.add(
            np.ones(1, dtype=np.float32) * idx,
            reward=float(idx),
            done=idx == 9,
            next_obs=np.ones(3, dtype=np.float32) * (idx + 1),
        )
    batch = replay.sample(batch_size=4, batch_length=5)
    assert batch["obs"].shape == (4, 6, 3)
    assert batch["actions"].shape == (4, 5, 1)
    assert batch["rewards"].shape == (4, 5)
    assert batch["dones"].shape == (4, 5)

