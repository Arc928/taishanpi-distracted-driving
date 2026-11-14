"""Driver splits, temporal frame sampling and leakage checks."""
import json
import random
from collections import defaultdict

from .io import digest

# State Farm 独立 eval 司机，任何图片都不能进入训练（见 AGENTS.md 第 4 节）
EVAL_DRIVERS = {'p016', 'p021', 'p026'}


def split_subjects(subjects, seed=42):
    """Fixed 35/5/10 split of the 50 3MDAD subjects."""
    subjects = sorted(subjects, key=lambda x: int(x[1:]))
    if len(subjects) != 50:
        raise ValueError('Expected exactly 50 3MDAD subjects')
    random.Random(seed).shuffle(subjects)
    return {s: 'train' if n < 35 else 'val' if n < 40 else 'test'
            for n, s in enumerate(subjects)}


def select_frames(sequence, count):
    """Use midpoint quantiles to spread samples over the action sequence."""
    sequence = sorted(sequence, key=lambda r: int(r['frame']))
    if len(sequence) < count:
        raise ValueError('Sequence is too short for the requested sample count')
    return [sequence[min(len(sequence)-1, int((i+.5)*len(sequence)/count))]
            for i in range(count)]


def check_disjoint(rows):
    """Fail if a subject or an identical image appears in more than one split."""
    by_subject, by_hash = defaultdict(set), defaultdict(set)
    for row in rows:
        by_subject[(row['source'], row['subject'])].add(row['split'])
        by_hash[row['image_sha256']].add(row['split'])
        if row['source'] == 'statefarm' and row['split'] == 'train' and row['subject'] in EVAL_DRIVERS:
            raise ValueError('State Farm evaluation-driver leakage')
    if any(len(v) > 1 for v in by_subject.values()):
        raise ValueError('Subject appears in multiple splits')
    if any(len(v) > 1 for v in by_hash.values()):
        raise ValueError('Identical image appears in multiple splits')


def snapshot(root):
    """SHA-256 of every val image and label under a State Farm dataset root."""
    return {str(p.relative_to(root)): digest(p)
            for folder in ('images/val', 'labels/val')
            for p in sorted((root / folder).glob('*'))}


def check_snapshot(root):
    expected = json.loads((root / 'eval_files_sha256.json').read_text())
    if snapshot(root) != expected:
        raise ValueError('新 eval 文件与快照不符')
