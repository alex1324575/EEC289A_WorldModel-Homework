"""SmallWorld-MJ course benchmark.

The package provides frozen MuJoCo task generation plus dynamics-only world
model training utilities. There is intentionally no reward, no policy, and no
actor-critic loop in the active homework surface.
"""

from .tasks import TASKS, TASKPACKS, TaskSpec, get_task, list_tasks, taskpack

__all__ = ["TASKS", "TASKPACKS", "TaskSpec", "get_task", "list_tasks", "taskpack"]
