#!/usr/bin/env python3
"""Rewrite project symlinks after raw data (or the project) moved. Preview by default.

    # 原始数据换位置后：先改 config/sources.yaml 的 root（或设环境变量），再预览、执行
    ./.venv/bin/python yolo_phone/relink_sources.py --source mdad
    ./.venv/bin/python yolo_phone/relink_sources.py --source mdad --apply

    # 把指向项目内部的绝对链接改为相对链接（项目整体搬家前）
    ./.venv/bin/python yolo_phone/relink_sources.py --internal-relative [--apply]

Only symlinks whose new target exists are rewritten; anything else is reported and the
whole run is refused. --apply writes a record of every change to archive/relink-<time>.json.
"""
import argparse
import os
from collections import Counter
from datetime import datetime
from pathlib import Path

from core.io import write_json
from core.paths import BASE
from core.sources import load as load_sources, root as source_root

PROJECT = BASE.parent  # 含 yolo_phone/ 与 .venv 的项目根
SKIP = {'.venv', '.git', '__pycache__'}


def symlinks(top=PROJECT):
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = [d for d in dirnames if d not in SKIP]
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                yield Path(path), os.readlink(path)


def plan_source(old_prefix, new_root, top=PROJECT):
    old_prefix = old_prefix.rstrip('/')
    changes = []
    for link, target in symlinks(top):
        if target == old_prefix or target.startswith(old_prefix + '/'):
            changes.append((link, target, str(new_root) + target[len(old_prefix):]))
    return changes


def plan_internal(top=PROJECT):
    top = Path(top).resolve()
    changes = []
    for link, target in symlinks(top):
        if os.path.isabs(target) and Path(target).is_relative_to(top):
            changes.append((link, target, os.path.relpath(target, link.parent)))
    return changes


def apply(changes):
    for link, _, new in changes:
        temporary = link.with_name(link.name + '.relink-tmp')
        os.symlink(new, temporary)
        os.replace(temporary, link)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source', choices=sorted(load_sources()), help='按 config/sources.yaml 改写该原始数据的链接')
    mode.add_argument('--internal-relative', action='store_true', help='项目内绝对链接改为相对链接')
    parser.add_argument('--from', dest='old', help='链接中现有的旧前缀；默认为 sources.yaml 的 recorded')
    parser.add_argument('--apply', action='store_true', help='实际改写；默认只预览')
    args = parser.parse_args()
    if args.source:
        old = args.old or load_sources()[args.source]['recorded']
        new = source_root(args.source).resolve()
        if Path(old).resolve() == new:
            print(f'{args.source}: 链接前缀 {old} 已指向当前位置，无需改写')
            return
        changes = plan_source(old, new)
    else:
        changes = plan_internal()
    missing = [c for c in changes if not (c[0].parent / c[2]).exists()]
    by_dir = Counter(str(c[0].relative_to(PROJECT)).split('/')[1] if len(c[0].relative_to(PROJECT).parts) > 1 else '.'
                     for c in changes)
    print(f'待改写 {len(changes)} 个链接，新目标缺失 {len(missing)} 个；按 yolo_phone 子目录: {dict(by_dir)}')
    for link, old_target, new_target in changes[:3]:
        print(f'  {link.relative_to(PROJECT)}\n    {old_target}\n -> {new_target}')
    if missing:
        for link, _, new_target in missing[:10]:
            print(f'  缺失: {link.relative_to(PROJECT)} -> {new_target}')
        raise SystemExit('有链接的新目标不存在，未做任何改写；请先确认原始数据已完整放到新位置')
    if not args.apply or not changes:
        print('预览完成，未改动（加 --apply 执行）' if changes else '没有需要改写的链接')
        return
    apply(changes)
    record = BASE / 'archive' / f'relink-{datetime.now():%Y%m%d-%H%M%S}.json'
    write_json(record, {'mode': args.source or 'internal-relative', 'count': len(changes),
                        'changes': [{'link': str(l.relative_to(PROJECT)), 'old': o, 'new': n} for l, o, n in changes]})
    print(f'已改写 {len(changes)} 个链接，记录在 {record.relative_to(PROJECT)}')


if __name__ == '__main__':
    main()
