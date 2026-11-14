"""Project root and archived-artifact lookup."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # yolo_phone/


def artifact(name):
    """Locate a dataset directory that may have moved to archive/data/."""
    active = BASE / name
    archived = BASE / 'archive' / 'data' / name
    return active if active.exists() else archived if archived.exists() else active
