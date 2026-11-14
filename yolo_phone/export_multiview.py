#!/usr/bin/env python3
"""Export explicitly reviewed multi-view annotations; reject pending/empty positives."""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from core.io import csv_rows, digest, write_csv, write_json
from core.labels import json_to_yolo, validate_label
from core.paths import BASE
from core.splits import check_disjoint, check_snapshot


def review_label(row, decision, path):
    data = json.loads(path.read_text())
    if (data.get('imageWidth'),data.get('imageHeight')) != (640,480):
        raise ValueError(f'Wrong annotation dimensions: {row["id"]}')
    if data.get('imagePath') != Path(row['image']).name:
        raise ValueError(f'Wrong annotation image: {row["id"]}')
    text = json_to_yolo(path)
    if decision == 'positive' and not text:
        raise ValueError(f'Positive has no boxes: {row["id"]}')
    if decision == 'background' and (text or row['source_behavior'] != 'background_candidate'):
        raise ValueError(f'Background must be a reviewed empty source background: {row["id"]}')
    if decision not in ('positive','background'):
        raise ValueError(f'Unsupported review decision: {decision}')
    validate_label(text, {'image':row['image'],'class_name':'background' if decision=='background' else 'positive'})
    return text


def export(dataset, out, confirmation):
    if not confirmation.strip():
        raise ValueError('Explicit human-review completion confirmation is required')
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    if (dataset/'BUILDING.json').exists():
        raise ValueError('Dataset build is incomplete')
    frozen = json.loads((dataset/'selection_hashes.json').read_text())
    for path, sha in frozen.items():
        if digest(dataset/path) != sha:
            raise ValueError(f'Frozen selection changed: {path}')
    check_snapshot(BASE/'dataset_23train_3eval_1of8')
    rows = csv_rows(dataset/'manifest.csv')
    decisions = csv_rows(dataset/'review_status.csv')
    by_id = {r['id']:r for r in decisions}
    new = [r for r in rows if r['source']=='3mdad']
    if len(by_id) != len(decisions) or set(by_id) != {r['id'] for r in new}:
        raise ValueError('Review decisions differ from selected image inventory')
    for row in new:
        decision = by_id[row['id']]
        if any(decision[k] != row[k] for k in ('split','review_batch','source_behavior')):
            raise ValueError('Review metadata changed')
        if decision['decision'] not in ('pending','positive','background','exclude'):
            raise ValueError('Unknown review decision')
        if decision['decision']=='exclude' and not decision['note'].strip():
            raise ValueError('Exclusions need a reason')
    if any(by_id[r['id']]['decision']=='pending' for r in new if r['split']=='val'):
        raise ValueError('Validation review is incomplete; no training dataset exported')
    test_ready = all(by_id[r['id']]['decision']!='pending' for r in new if r['split']=='test')
    prepared, excluded = [], []
    for row in rows:
        if row['source']=='statefarm':
            if digest(dataset/row['label']) != row['label_sha256']:
                raise ValueError('Inherited label modified')
            text = (dataset/row['label']).read_text()
            origin = row['origin']
            json_sha = ''
        else:
            decision = by_id[row['id']]
            if decision['decision'] in ('pending','exclude') or (row['split']=='test' and not test_ready):
                excluded.append({'id':row['id'],'split':row['split'],'decision':decision['decision'],'note':decision['note']})
                continue
            path = dataset/row['annotation_json']
            text = review_label(row,decision['decision'],path)
            json_sha = digest(path)
            origin = 'human_review:'+decision['decision']
        if digest(dataset/row['image']) != row['image_sha256']:
            raise ValueError('Selected image modified')
        prepared.append({**row,'text':text,'origin':origin,'review_json_sha256':json_sha})
    for split in ('train','val') + (('test',) if test_ready else ()):
        new_split = [r for r in prepared if r['source']=='3mdad' and r['split']==split]
        classes = {line.split()[0] for r in new_split for line in r['text'].splitlines()}
        if classes != {'0','1'} or not any(not r['text'] for r in new_split):
            raise ValueError(f'{split} needs reviewed texting, calling and background samples')
    check_disjoint(prepared+csv_rows(dataset/'eval_statefarm/manifest.csv'))
    out.mkdir(parents=True)
    exported = []
    for row in prepared:
        image = f'images/{row["split"]}/{row["id"]}.jpg'
        label = f'labels/{row["split"]}/{row["id"]}.txt'
        for path in (out/image,out/label):
            path.parent.mkdir(parents=True,exist_ok=True)
        (out/image).symlink_to((dataset/row['image']).resolve())
        (out/label).write_text(row['text'])
        if row['source']=='3mdad':
            snapshot = out/'reviewed_json'/f'{row["id"]}.json'
            snapshot.parent.mkdir(exist_ok=True)
            shutil.copyfile(dataset/row['annotation_json'],snapshot)
        exported.append({**{k:v for k,v in row.items() if k!='text'},'image':image,'label':label,
                         'label_sha256':digest(out/label),'annotation_classes':','.join(sorted({
                             {'0':'texting','1':'calling'}[line.split()[0]] for line in row['text'].splitlines()})),
                         'annotation_status':'inherited_valid' if row['source']=='statefarm' else 'human_reviewed'})
    write_csv(out/'manifest.csv',exported)
    shutil.copyfile(dataset/'subject_splits.json',out/'subject_splits.json')
    write_json(out/'human_review_receipt.json', {'confirmation':confirmation,'source_dataset':str(dataset),
               'selection_hashes':frozen,'review_status_sha256':digest(dataset/'review_status.csv'),
               'counts':dict(Counter(r['split'] for r in exported)), 'test_included':test_ready,
               'omitted':excluded,'eval_files_unchanged':True})
    check_snapshot(BASE/'dataset_23train_3eval_1of8')
    # Write usable training entry point last, after every verification succeeds.
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\n'+
                              ('test: images/test\n' if test_ready else '')+
                              'names:\n  0: texting\n  1: calling\n')
    print(f'Exported {len(exported)} verified samples to {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--confirmation',required=True)
    args = parser.parse_args()
    export(args.dataset.resolve(),args.output.resolve(),args.confirmation)


if __name__=='__main__':
    main()
