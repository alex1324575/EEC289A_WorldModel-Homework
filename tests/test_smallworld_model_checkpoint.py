import torch

from smallworld_hw.checkpointing import load_smallworld_checkpoint, save_smallworld_checkpoint
from smallworld_hw.models import SmallWorldRSSM
from smallworld_hw.tasks import get_task


def test_smallworld_model_forward_and_checkpoint_roundtrip(tmp_path):
    task = get_task("simple_pendulum")
    cfg = {
        "world_model": {
            "embed_dim": 8,
            "deter_dim": 8,
            "stoch_dim": 4,
            "hidden_dim": 16,
            "free_nats": 1.0,
            "kl_scale": 1.0,
            "rep_kl_scale": 0.1,
            "state_loss_scale": 1.0,
            "prior_state_loss_scale": 0.25,
            "learning_rate": 1e-3,
            "grad_clip": 10.0,
        },
        "evaluation": {"warmup_steps": 2, "open_loop_horizon": 3, "batch_size": 2},
    }
    model = SmallWorldRSSM(task.state_dim, task.action_dim, cfg["world_model"])
    batch = {
        "states": torch.zeros(2, 6, task.state_dim),
        "actions": torch.zeros(2, 5, task.action_dim),
    }
    loss, metrics = model.loss(batch, cfg["world_model"])
    assert loss.ndim == 0
    assert "sw/loss" in metrics

    save_smallworld_checkpoint(
        tmp_path / "ckpt",
        model=model,
        optimizer=None,
        config=cfg,
        task_name=task.name,
        metrics={"open_loop_15_rmse": 1.0},
        update=3,
    )
    loaded, loaded_cfg, payload = load_smallworld_checkpoint(tmp_path / "ckpt", "cpu")
    assert loaded_cfg == cfg
    assert payload["task_name"] == task.name
    assert sum(p.numel() for p in loaded.parameters()) == sum(p.numel() for p in model.parameters())
