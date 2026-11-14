"""Single source of truth for the currently active dataset, weights and threshold.

Values live in config/current.yaml; this module only loads and validates them.
"""
import hashlib
from functools import lru_cache
from pathlib import Path

import yaml

from .paths import BASE

CONFIG = BASE / 'config' / 'current.yaml'
REQUIRED = ('imgsz', 'nms_iou', 'dataset', 'weights', 'confidence')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=None)
def load(path=CONFIG):
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ValueError(f'{path}: 缺少配置项 {missing}')
    if data['imgsz'] != 640:
        raise ValueError(f'{path}: 训练与推理尺寸固定为 640')
    if not 0.1 <= float(data['confidence']) <= 0.95:
        raise ValueError(f'{path}: confidence 须在 0.1 到 0.95 之间')
    if not {'path', 'sha256'} <= set(data['weights']):
        raise ValueError(f'{path}: weights 须包含 path 与 sha256')
    return data


def dataset_path():
    """data.yaml of the current training set, located through config/datasets.yaml."""
    from . import registry
    dataset_id = load()['dataset']
    if registry.entry(dataset_id)['kind'] != 'training_set':
        raise ValueError(f'当前数据集 {dataset_id} 不是 training_set')
    return registry.path(dataset_id) / 'data.yaml'


def weights_path(verify=True):
    """Return the active weights; with verify, refuse a file whose hash differs from the config."""
    entry = load()['weights']
    path = BASE / entry['path']
    if not path.is_file():
        raise FileNotFoundError(f'当前权重不存在: {path}')
    if verify and sha256(path) != entry['sha256']:
        raise ValueError(f'当前权重哈希与 config/current.yaml 不一致: {path}')
    return path


def confidence():
    return float(load()['confidence'])


def imgsz():
    return int(load()['imgsz'])


def nms_iou():
    return float(load()['nms_iou'])
