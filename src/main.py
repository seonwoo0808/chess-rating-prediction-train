"""Compatibility module for ``PYTHONPATH=src python -m main``."""
from train.main import build_parser, main, run_training

__all__ = ["build_parser", "main", "run_training"]


if __name__ == "__main__":
    raise SystemExit(main())
