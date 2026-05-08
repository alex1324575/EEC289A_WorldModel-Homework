"""Training loop for dynamics-only SmallWorld-MJ world models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from smallworld_mj.checkpoint import build_model, save_checkpoint
from smallworld_mj.config import save_json, set_seed
from smallworld_mj.data import Normalizer, load_split
from smallworld_mj.solution.losses import compute_loss as solution_loss
from smallworld_mj.student.losses import compute_loss as student_loss
from smallworld_mj.tasks import get_task, taskpack


def _tensor_batch(data: dict[str, np.ndarray], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    keys_float = ["states", "actions", "task_params", "state_mask", "action_mask", "param_mask"]
    batch = {key: torch.as_tensor(data[key][indices], dtype=torch.float32, device=device) for key in keys_float}
    batch["task_id"] = torch.as_tensor(data["task_id"][indices], dtype=torch.long, device=device)
    return batch


def _filter_tasks(data: dict[str, np.ndarray], tasks: list[str]) -> dict[str, np.ndarray]:
    allowed = {get_task(name).task_id for name in tasks}
    keep = np.asarray([int(t) in allowed for t in data["task_id"]], dtype=bool)
    return {key: value[keep] if hasattr(value, "__len__") and len(value) == len(keep) else value for key, value in data.items()}


def evaluate_loss(model: torch.nn.Module, data: dict[str, np.ndarray], cfg: dict[str, Any], device: torch.device, model_name: str) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        indices = np.arange(min(len(data["task_id"]), int(cfg["training"].get("eval_batch_size", 128))))
        batch = _tensor_batch(data, indices, device)
        loss_fn = student_loss if model_name == "student" else solution_loss
        _, metrics = loss_fn(model, batch, cfg)
    model.train()
    return {f"val/{k}": v for k, v in metrics.items()}


def train_model(
    cfg: dict[str, Any],
    *,
    model_name: str,
    taskpack_name: str,
    dataset_dir: str | Path,
    output_dir: str | Path,
    local_smoke: bool = False,
) -> dict[str, Any]:
    set_seed(int(cfg.get("seed", 0)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(dataset_dir, "train")
    val = load_split(dataset_dir, "val")
    tasks = taskpack(taskpack_name)
    train = _filter_tasks(train, tasks)
    val = _filter_tasks(val, tasks)
    normalizer = Normalizer.from_train(train)
    dims = (train["states"].shape[-1], train["actions"].shape[-1], train["task_params"].shape[-1])
    num_tasks = max(int(train["task_id"].max()) + 1, 10)
    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get("force_cpu", False) else "cpu")
    model = build_model(model_name, dims, num_tasks, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("learning_rate", 8e-4)))
    updates = int(cfg["training"]["updates_smoke"] if local_smoke else cfg["training"]["updates"])
    batch_size = int(cfg["training"]["batch_size_smoke"] if local_smoke else cfg["training"]["batch_size"])
    eval_every = int(cfg["training"].get("eval_every", 100 if not local_smoke else 5))
    rng = np.random.default_rng(int(cfg.get("seed", 0)) + 99)
    loss_fn = student_loss if model_name == "student" else solution_loss
    best_val = float("inf")
    best_metrics: dict[str, float] = {}
    print(f"[train] model={model_name} taskpack={taskpack_name} device={device} updates={updates}")
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(train["task_id"]), size=batch_size)
        batch = _tensor_batch(train, indices, device)
        loss, metrics = loss_fn(model, batch, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("grad_clip", 100.0)))
        opt.step()
        if update == 1 or update % eval_every == 0 or update == updates:
            val_metrics = evaluate_loss(model, val, cfg, device, model_name)
            metric_line = " ".join([f"{k}={v:.4f}" for k, v in {**metrics, **val_metrics}.items() if isinstance(v, float)])
            print(f"[train] update={update} {metric_line}")
            val_loss = val_metrics["val/loss/total"]
            if val_loss < best_val:
                best_val = val_loss
                best_metrics = {**metrics, **val_metrics}
                save_checkpoint(
                    output_dir / "best_checkpoint",
                    model=model,
                    config=cfg,
                    model_name=model_name,
                    dims=dims,
                    num_tasks=num_tasks,
                    normalizer=normalizer.to_dict(),
                    step=update,
                    metrics=best_metrics,
                )
    save_json(output_dir / "train_summary.json", {"model": model_name, "tasks": tasks, "updates": updates, "best_val_loss": best_val, "metrics": best_metrics})
    return {"output_dir": str(output_dir), "best_val_loss": best_val, "updates": updates, "device": str(device)}
