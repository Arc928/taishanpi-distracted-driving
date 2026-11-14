"""Locate active and archived artifacts after the workspace cleanup."""
from pathlib import Path

BASE = Path(__file__).resolve().parent


def artifact(name):
    active = BASE / name
    archived = BASE / 'archive' / 'data' / name
    return active if active.exists() else archived if archived.exists() else active
