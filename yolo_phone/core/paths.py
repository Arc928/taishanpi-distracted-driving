"""Project root and dataset lookup by directory name."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # yolo_phone/


def artifact(name):
    """Locate a dataset by its directory name via config/datasets.yaml (active or archived)."""
    from .registry import by_dirname  # registry imports BASE from here
    return by_dirname(name)
