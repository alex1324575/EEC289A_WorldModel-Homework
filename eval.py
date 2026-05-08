from __future__ import annotations

import argparse
import json

from smallworld_mj.eval import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=90)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint_dir,
        args.dataset_dir,
        args.split,
        args.output_dir,
        warmup=args.warmup,
        horizon=args.horizon,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
