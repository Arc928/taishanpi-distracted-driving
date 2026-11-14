#!/usr/bin/env python3
"""Run one training experiment from its plan file; the plan is the only source of hyperparameters.

    ./.venv/bin/python yolo_phone/train.py --plan yolo_phone/experiments/<实验>/plan.yaml [--dry-run]

The plan names the run, dataset, start weights (with SHA-256) and a `train:` block that is
passed verbatim to Ultralytics. imgsz comes from config/current.yaml (fixed 640); data,
project, name, imgsz, model and resume may not appear in `train:`. Outputs go next to the plan:
resolved_plan.json (absolute paths, hashes, git commit), status.json and result.json.
Resuming an interrupted run needs its own verified script (see experiments/official_resume*).
"""
import argparse
import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

from core import config as current_config
from core.io import digest, write_json
from core.labels import NAMES
from core.paths import BASE

REQUIRED = ('name', 'data', 'model', 'model_sha256', 'train')
RESERVED = {'data', 'project', 'name', 'imgsz', 'model', 'resume'}


def now():
    return datetime.now().astimezone().isoformat()


def git_state():
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=BASE, capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=BASE,
                                    capture_output=True, text=True, check=True).stdout.strip())
        return {'commit': commit, 'dirty': dirty}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}


def resolve(path):
    """Plan paths are relative to yolo_phone/."""
    path = Path(path)
    return (path if path.is_absolute() else BASE / path).resolve()


def load_plan(path):
    """Validate a plan and return (plan, resolved) without touching the run directory."""
    plan = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    missing = [k for k in REQUIRED if k not in plan]
    if missing:
        raise ValueError(f'计划缺少字段 {missing}')
    name = str(plan['name'])
    if not name or '<' in name or '/' in name:
        raise ValueError('name 须为 runs/ 下的新运行名（不能是模板占位符）')
    reserved = RESERVED & set(plan['train'])
    if reserved:
        raise ValueError(f'train 中不能出现 {sorted(reserved)}；它们由计划顶层字段或 config 决定')
    data, model = resolve(plan['data']), resolve(plan['model'])
    if not data.is_file():
        raise FileNotFoundError(f'训练集配置不存在: {data}')
    names = yaml.safe_load(data.read_text(encoding='utf-8')).get('names')
    if names != NAMES:
        raise ValueError(f'{data}: 类别映射 {names} 与固定 id {NAMES} 不符')
    if not model.is_file():
        raise FileNotFoundError(f'起点权重不存在: {model}')
    if digest(model) != plan['model_sha256']:
        raise ValueError(f'起点权重哈希与计划不符: {model}')
    manifest = data.parent / 'manifest.csv'
    run_dir = BASE / 'runs' / name
    resolved = {
        'plan': str(Path(path).resolve()), 'run_dir': str(run_dir), 'data': str(data),
        'dataset_manifest_sha256': digest(manifest) if manifest.is_file() else None,
        'model': str(model), 'model_sha256': plan['model_sha256'],
        'train_kwargs': {**plan['train'], 'data': str(data), 'project': str(run_dir.parent),
                         'name': run_dir.name, 'imgsz': current_config.imgsz()},
        'git': git_state(),
    }
    return plan, resolved


def best_epoch(run_dir):
    """Ultralytics 8.4 saves best.pt at the max of mAP50-95 (DetMetrics.fitness weights [0,0,0,1])."""
    rows = list(csv.DictReader((run_dir / 'results.csv').open()))
    best = max(rows, key=lambda r: float(r['metrics/mAP50-95(B)']))
    return len(rows), best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true', help='只校验计划并打印最终参数，不训练')
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    here = plan_path.parent
    plan, resolved = load_plan(plan_path)
    run_dir = Path(resolved['run_dir'])
    if run_dir.exists():
        raise SystemExit(f'运行目录已存在，拒绝覆盖: {run_dir}')
    if (here / 'status.json').exists():
        raise SystemExit(f'{here} 已有 status.json；一个计划只对应一次训练，请新建实验目录')
    if args.dry_run:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return
    if resolved['git']['dirty']:
        print('警告：yolo_phone/ 有未提交改动，resolved_plan.json 记录的提交不能完全复现本次代码', flush=True)
    write_json(here / 'resolved_plan.json', resolved)
    status = {'status': 'running', 'started_at': now(), 'pid': os.getpid(), 'run_dir': str(run_dir),
              'test_used': False, 'service_model_changed': False}
    write_json(here / 'status.json', status)
    try:
        from ultralytics import YOLO
        YOLO(resolved['model']).train(**resolved['train_kwargs'])
        epochs, best = best_epoch(run_dir)
        weights = run_dir / 'weights' / 'best.pt'
        write_json(here / 'result.json', {
            'completed_epochs': epochs, 'best_epoch': int(best['epoch']),
            'best_rule': 'max metrics/mAP50-95(B) on val, same as Ultralytics best.pt',
            'best_validation': best, 'best_weights': str(weights), 'best_weights_sha256': digest(weights),
            'test_used': False})
        status.update(status='complete', finished_at=now(), completed_epochs=epochs, best_epoch=int(best['epoch']))
    except BaseException as error:
        status.update(status='failed', finished_at=now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(here / 'status.json', status)


if __name__ == '__main__':
    main()
