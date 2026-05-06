import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
pytestmark = pytest.mark.skipif(torch_spec is None, reason="torch is not installed")

if torch_spec is not None:
    import torch

    from world_model_hw.agent import MiniDreamerAgent, lambda_return
    from world_model_hw.models import WorldModel


def tiny_config():
    return {
        "world_model": {
            "embed_dim": 8,
            "deter_dim": 8,
            "stoch_dim": 4,
            "hidden_dim": 16,
            "free_nats": 1.0,
            "kl_scale": 1.0,
            "rep_kl_scale": 0.1,
            "obs_loss_scale": 1.0,
            "reward_loss_scale": 1.0,
            "continue_loss_scale": 0.1,
            "learning_rate": 0.001,
            "grad_clip": 10.0,
        },
        "actor_critic": {
            "hidden_dim": 16,
            "imag_horizon": 3,
            "discount": 0.99,
            "lambda": 0.95,
            "actor_learning_rate": 0.001,
            "critic_learning_rate": 0.001,
            "entropy_scale": 0.0,
            "min_std": 0.1,
            "target_update_interval": 10,
            "target_update_tau": 0.1,
            "grad_clip": 10.0,
        },
    }


def test_lambda_return_shape():
    rewards = torch.ones(3, 2)
    values = torch.zeros(3, 2)
    discounts = torch.ones(3, 2) * 0.9
    bootstrap = torch.zeros(2)
    returns = lambda_return(rewards, values, discounts, bootstrap, 0.95)
    assert returns.shape == (3, 2)
    assert torch.all(returns > 0)


def test_world_model_forward_loss_shapes():
    cfg = tiny_config()
    wm = WorldModel(obs_dim=3, action_dim=1, cfg=cfg["world_model"])
    batch = {
        "obs": torch.randn(2, 6, 3),
        "actions": torch.randn(2, 5, 1).clamp(-1, 1),
        "rewards": torch.randn(2, 5),
        "dones": torch.zeros(2, 5),
    }
    loss, metrics, rssm_out = wm.loss(batch, cfg["world_model"])
    assert loss.ndim == 0
    assert rssm_out["post_feat"].shape[:2] == (2, 5)
    assert "wm/loss" in metrics


def test_agent_update_roundtrip():
    cfg = tiny_config()
    agent = MiniDreamerAgent(obs_dim=3, action_dim=1, config=cfg, device="cpu")
    batch = {
        "obs": torch.randn(2, 6, 3).numpy(),
        "actions": torch.randn(2, 5, 1).clamp(-1, 1).numpy(),
        "rewards": torch.randn(2, 5).numpy(),
        "dones": torch.zeros(2, 5).numpy(),
    }
    wm_metrics = agent.train_world_model(batch)
    ac_metrics = agent.train_actor_critic(batch)
    assert "wm/loss" in wm_metrics
    assert "ac/actor_loss" in ac_metrics

