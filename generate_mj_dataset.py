from __future__ import annotations

import argparse
import json

from smallworld_mj.config import load_config
from smallworld_mj.data.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smallworld_mj.yaml")
    parser.add_argument("--taskpack", default="smallworld_all")
    parser.add_argument("--profile", default="colab", choices=["smoke", "colab", "full"])
    parser.add_argument("--output-dir", default="artifacts/mj_data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = generate_dataset(cfg, taskpack_name=args.taskpack, profile=args.profile, output_dir=args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
