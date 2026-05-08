from __future__ import annotations

import argparse
import json

import mujoco

from smallworld_mj.config import load_config
from smallworld_mj.mujoco_utils import model_from_xml
from smallworld_mj.tasks import get_task, taskpack


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/smallworld_mj.yaml")
    parser.add_argument("--taskpack", default="smallworld_all")
    args = parser.parse_args()
    load_config(args.config)
    rows = []
    for name in taskpack(args.taskpack):
        spec = get_task(name)
        model = model_from_xml(spec.xml_path)
        rows.append(
            {
                "task": name,
                "xml": str(spec.xml_path),
                "state_dim": spec.state_dim,
                "action_dim": spec.action_dim,
                "param_dim": spec.param_dim,
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
            }
        )
    print(json.dumps({"mujoco": mujoco.__version__, "tasks": rows}, indent=2))


if __name__ == "__main__":
    main()
