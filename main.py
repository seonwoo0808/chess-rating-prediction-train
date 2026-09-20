"""Run the training entry point from the project root.

Use ``uv run train`` for the installed command or ``python main.py`` from an
environment where this project has been installed with ``uv sync``.
"""
from train.main import build_parser, main, run_training

__all__ = ["build_parser", "main", "run_training"]


if __name__ == "__main__":
    raise SystemExit(main())
