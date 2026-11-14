"""Raw-data locations (config/sources.yaml), overridable per machine by environment variable.

Manifests and symlinks keep the absolute paths recorded when they were built. `localize`
maps such a recorded path onto the current root, so scripts keep working after the raw data
moves; stored records are never rewritten.
"""
import os
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import BASE

SOURCES = BASE / 'config' / 'sources.yaml'


@lru_cache(maxsize=None)
def load(path=SOURCES):
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['sources']
    for name, entry in data.items():
        if not {'recorded', 'root'} <= set(entry):
            raise ValueError(f'{name}: 须包含 recorded 与 root')
        if not Path(entry['recorded']).is_absolute():
            raise ValueError(f'{name}: recorded 须为绝对路径')
    return data


def root(name):
    """Current location of a raw dataset: environment variable, else config root."""
    entry = load()[name]
    value = os.environ.get(entry['env']) if entry.get('env') else None
    return Path(value or entry['root']).expanduser()


def localize(path):
    """Map a path recorded under a source's original prefix onto that source's current root."""
    text = str(path)
    for name, entry in load().items():
        prefix = entry['recorded'].rstrip('/')
        if text == prefix or text.startswith(prefix + '/'):
            return str(root(name)) + text[len(prefix):]
    return text
