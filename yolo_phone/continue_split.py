#!/usr/bin/env python3
"""Reuse reviewed labels and screen new proposals against an explicit driver split."""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

# csv_rows/digest/snapshot/check_snapshot/validate_label 已移入 core，此处导入以兼容旧引用
from core.io import csv_rows, digest  # noqa: F401
from core.labels import json_to_yolo, validate_label
from core.paths import artifact
from core.splits import check_snapshot, snapshot  # noqa: F401

BASE = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE / 'dataset_23train_3eval_1of8'


def materialize(root, out, records, seed_dir=None):
    if out.exists():
        raise ValueError(f'拒绝覆盖: {out}')
    check_snapshot(root)
    manifest = {Path(r['image']).name: r for r in csv_rows(root / 'manifest.csv')}
    names = [r['name'] for r in records]
    if len(set(names)) != len(names) or not names:
        raise ValueError('选择清单为空或有重复')
    for record in records:
        row = manifest[record['name']]
        if row['split'] != 'train':
            raise ValueError(f"eval 泄漏: {record['name']}")
        validate_label(record['label_text'], row)
        if not (root / row['image']).is_file():
            raise ValueError('图片缺失')
    if seed_dir:
        seed_names = {Path(r['image']).name for r in csv_rows(seed_dir / 'manifest.csv')}
        if not seed_names <= set(names):
            raise ValueError('拟合检查种子不在 train 中')
    for sub in ('images/train', 'labels/train'):
        (out / sub).mkdir(parents=True)
    selected = []
    for record in records:
        row = manifest[record['name']]
        (out / row['image']).symlink_to((root / row['image']).resolve())
        label = out / row['label']
        label.write_text(record['label_text'])
        selected.append({**row, 'dataset_image': row['image'],
                         'origin': record['origin'], 'label_source': record['label_source'],
                         'label_sha256': digest(label), 'status': 'label_available',
                         'annotation_classes': ','.join(sorted({
                             {'0':'texting','1':'calling'}[line.split()[0]]
                             for line in record['label_text'].splitlines()})),
                         'annotation_box_count': len(record['label_text'].splitlines())})
    with (out / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    val = str(seed_dir / 'images/train') if seed_dir else 'images/train'
    (out / 'data.yaml').write_text(
        '# val overlaps train: fitting diagnostics only; independent eval is separate.\n'
        f'path: {out}\ntrain: images/train\nval: {val}\n'
        'names:\n  0: texting\n  1: calling\n')
    (out / 'classes.txt').write_text('texting\ncalling\n')
    report = {'count': len(records), 'by_class': dict(Counter(r['class_name'] for r in selected)),
              'by_annotation_class': dict(Counter(r['annotation_classes'] or 'background' for r in selected)),
              'by_origin': dict(Counter(r['origin'] for r in selected)),
              'train_subjects': sorted({r['subject'] for r in selected}),
              'eval_subjects': sorted({r['subject'] for r in manifest.values() if r['split']=='val'}),
              'dataset': str(root), 'eval_files_unchanged': True,
              'val_is_training_fit_check': True}
    check_snapshot(root)
    (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(str(out), json.dumps(report, ensure_ascii=False), flush=True)


def migrate(args):
    root = args.dataset.resolve()
    old = artifact('selftrain_round2')
    reviewed = artifact('seed_reviewed')
    old_review = artifact('round2_review')
    receipt = json.loads((reviewed / 'human_review_receipt.json').read_text())
    for filename, sha in receipt['json_sha256'].items():
        if digest(reviewed / 'xlabel' / filename) != sha:
            raise ValueError(f'种子 JSON 与确认记录不符: {filename}')
    records, seeds = [], []
    for r in csv_rows(old / 'selection.csv'):
        name = Path(r['image']).name
        label = old / 'labels/train' / Path(name).with_suffix('.txt')
        text = label.read_text()
        origin = r['origin']
        if origin == 'round1:human_seed':
            jpath = reviewed / 'xlabel' / Path(name).with_suffix('.json')
            if text != json_to_yolo(jpath):
                raise ValueError(f'种子导出与 JSON 不符: {name}')
        elif origin.startswith('round2:'):
            jpath = old_review / 'xlabel' / Path(name).with_suffix('.json')
            if text != json_to_yolo(jpath):
                raise ValueError(f'第二轮导出与当前 JSON 不符: {name}')
        record = {'name': name, 'origin': 'reused:' + origin,
                  'label_text': text, 'label_source': str(label)}
        records.append(record)
        # Accepted prefill remains pseudo label; do not promote it to human truth.
        if origin == 'round1:human_seed' or (origin.startswith('round2:') and 'accepted_prefill' not in origin):
            seeds.append(record)
    materialize(root, args.seed_output.resolve(), seeds)
    materialize(root, args.output.resolve(), records, args.seed_output.resolve())


def screened_box(proposals, expected, threshold):
    visible = [b for b in proposals if b['confidence'] >= 0.5]
    if len(visible) != 1:
        return None
    box = visible[0]
    if box['class_id'] != expected or box['confidence'] < threshold:
        return None
    return box


def expand(args):
    root, base, proposals = args.dataset.resolve(), args.base.resolve(), args.proposals.resolve()
    provenance = json.loads((proposals / 'model_provenance.json').read_text())
    if Path(provenance['dataset']).resolve() != root:
        raise ValueError('伪标签来源不是当前数据集')
    manifest = {r['image']:r for r in csv_rows(root / 'manifest.csv')}
    records = {}
    for row in csv_rows(base / 'manifest.csv'):
        name = Path(row['image']).name
        label = base / row['label']
        if digest(label) != row['label_sha256']:
            raise ValueError('基础标注已改动')
        records[row['image']] = {'name':name,'origin':row['origin'],
                                'label_text':label.read_text(),'label_source':str(label)}
    raw = json.loads((proposals / 'raw_predictions.json').read_text())
    excluded, backgrounds = [], []
    seen = set()
    for r in raw:
        image = r['image']
        if image in seen:
            raise ValueError('重复推理记录')
        seen.add(image)
        row = manifest[image]
        if row['split'] != 'train' or row['class_name'] != r['source_behavior']:
            raise ValueError('伪标签源类别或划分不符')
        if image in records:
            continue
        if row['class_name'] == 'background':
            if not any(b['confidence'] >= 0.5 for b in r['proposals']):
                backgrounds.append(r)
            continue
        box = screened_box(r['proposals'], int(row['class_id']), args.threshold)
        if box is None:
            excluded.append({'image': image, 'reason': 'needs_manual_review'})
            continue
        x1,y1,x2,y2 = box['xyxy']
        w,h = r['image_width'], r['image_height']
        text = f"{int(row['class_id'])} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
        records[image] = {'name':Path(image).name,'origin':f'new_pseudo_ge_{args.threshold}',
                          'label_text':text,'label_source':str(proposals/'raw_predictions.json')}
    positives = sum(bool(r['label_text']) for r in records.values())
    negative_count = len(records)-positives
    need = max(0,positives//2-negative_count)
    for r in random.Random(42).sample(backgrounds,min(need,len(backgrounds))):
        records[r['image']] = {'name':Path(r['image']).name,'origin':'new_source_background_no_detection',
                               'label_text':'','label_source':str(proposals/'raw_predictions.json')}
    out = args.output.resolve()
    materialize(root,out,list(records.values()),args.seed_output.resolve())
    (out/'screening.json').write_text(json.dumps({
        'confidence_threshold':args.threshold,'unique_box_threshold':0.5,
        'base':str(base),'proposals':str(proposals),'model_provenance':provenance,
        'excluded_positive_count':len(excluded),'excluded_positives':excluded,
        'policy':'Positive misses never become background; human review required for excluded positives.',
    },ensure_ascii=False,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('migrate','expand'))
    parser.add_argument('--dataset',type=Path,default=DEFAULT_DATASET)
    parser.add_argument('--seed-output',type=Path,default=BASE/'seed_23train_reviewed')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base',type=Path)
    parser.add_argument('--proposals',type=Path)
    parser.add_argument('--threshold',type=float,default=0.8)
    args = parser.parse_args()
    if not 0.8 <= args.threshold <= 1:
        parser.error('伪标签门槛必须 >= 0.8')
    if args.mode=='expand' and (args.base is None or args.proposals is None):
        parser.error('expand 需要 --base 与 --proposals')
    if args.mode=='migrate':
        migrate(args)
    else:
        expand(args)


if __name__=='__main__':
    main()
