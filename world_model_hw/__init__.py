"""Teaching-first MiniDreamer homework package.

The package initializer intentionally avoids importing PyTorch modules. This
keeps lightweight utilities such as `public_eval.py` usable even before the
Colab dependency installation cell has run.
"""

__all__ = [
    "agent",
    "checkpointing",
    "config",
    "envs",
    "models",
    "replay",
    "visualization",
]
