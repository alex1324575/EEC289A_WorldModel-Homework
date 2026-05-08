from __future__ import annotations

import argparse
import json

from smallworld_mj.config import load_config
from smallworld_mj.train import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument("--taskpack", default=None)
    parser.add_argument("--model", choices=["baseline_mlp", "student", "solution"], required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    taskpack_name = args.taskpack or args.task
    if taskpack_name is None:
        raise SystemExit("Provide --task or --taskpack.")
    result = train_model(
        cfg,
        model_name=args.model,
        taskpack_name=taskpack_name,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        local_smoke=args.local_smoke,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
