#!/usr/bin/env python3
"""Generate SmallWorld-Lite state/action datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from world_model_hw.config import load_json, save_json, set_runtime_env

from smallworld_hw.dataset import dataset_summary, generate_task_dataset, task_names_from_arg


DEFAULT_CONFIG = Path("configs/smallworld_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", type=str, default="all", help="Task name or 'all'.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_runtime_env()
    config = load_json(args.config)
    output_dir = args.output_dir or Path(config["dataset"]["default_output_dir"])
    tasks = task_names_from_arg(args.task)
    written = {}
    summaries = {}
    for task_name in tasks:
        written[task_name] = generate_task_dataset(output_dir, task_name, config, local_smoke=args.local_smoke)
        summaries[task_name] = dataset_summary(output_dir, task_name)
    summary = {
        "benchmark_name": config["benchmark_name"],
        "local_smoke": bool(args.local_smoke),
        "output_dir": str(output_dir),
        "tasks": tasks,
        "written": written,
        "summaries": summaries,
    }
    save_json(output_dir / "dataset_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
