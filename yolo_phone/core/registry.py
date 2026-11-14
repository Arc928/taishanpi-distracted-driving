"""Dataset registry (config/datasets.yaml): scripts refer to datasets by id, never by directory.

Moving a dataset directory then means editing one `path:` line. Unknown ids and
unregistered directory names raise instead of silently falling back.
"""
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import BASE

REGISTRY = BASE / 'config' / 'datasets.yaml'
STATUSES = {'active', 'protected', 'historical', 'archived'}


@lru_cache(maxsize=None)
def load(path=REGISTRY):
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['datasets']
    for key, entry in data.items():
        if entry.get('status') not in STATUSES:
            raise ValueError(f'{key}: status 须为 {sorted(STATUSES)}')
        for parent in entry.get('parents', []):
            if parent not in data:
                raise ValueError(f'{key}: 上游 {parent} 未登记')
    names = [Path(e['path']).name for e in data.values()]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        raise ValueError(f'目录名重复，无法按名称定位: {sorted(duplicated)}')
    return data


def entry(dataset_id):
    data = load()
    if dataset_id not in data:
        raise KeyError(f'数据集未登记: {dataset_id}（见 config/datasets.yaml）')
    return data[dataset_id]


def path(dataset_id):
    return BASE / entry(dataset_id)['path']


def by_dirname(name):
    """Registered path for a directory name (used by older scripts that refer to datasets by folder)."""
    for dataset_id, item in load().items():
        if Path(item['path']).name == name:
            return BASE / item['path']
    raise KeyError(f'目录 {name} 未在 config/datasets.yaml 登记')
