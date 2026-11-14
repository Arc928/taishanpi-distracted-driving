#!/usr/bin/env python3
"""Merge a reviewed targeted 3MDAD round into a copy of the current training set.

Decisions come from the reviewed boxes: any box -> positive; an empty source phone
action -> excluded (phone not visible), never background; an empty hard negative ->
background. exclusions.csv drops ambiguous images. Base labels are copied unchanged.
"""
import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from annotation_io import json_to_yolo
from continue_split import check_snapshot, csv_rows, digest
from prepare_multiview import BASE, check_disjoint, write_csv, write_json

DEFAULT_BASE = BASE/'selftrain_multiview_gemini_reviewed_20261010'
TEST_MANIFEST = BASE/'gemini38flash_3mdad_v1/review_dataset/manifest.csv'


def decide(row, text, excluded):
    if row['id'] in excluded:
        return 'exclude'
    if text:
        return 'positive'
    return 'background' if row['kind']=='hard_negative' else 'exclude'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--targeted',type=Path,required=True)
    parser.add_argument('--confirmation',required=True,help='Human statement that the targeted review is complete')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    base, targeted, out = args.base.resolve(), args.targeted.resolve(), args.output.resolve()
    if not args.confirmation.strip():
        raise ValueError('Explicit human-review completion confirmation is required')
    if out.exists():
        raise ValueError('Output exists; refusing overwrite')
    if (targeted/'BUILDING.json').exists():
        raise ValueError('Targeted selection is incomplete')
    excluded = {}
    for e in csv_rows(targeted/'exclusions.csv'):
        if not e['reason'].strip():
            raise ValueError(f'Exclusion needs a reason: {e["id"]}')
        excluded[e['id']] = e['reason']
    initial = json.loads((targeted/'review_initial_hashes.json').read_text())
    base_rows = csv_rows(base/'manifest.csv')
    for r in base_rows:
        if digest(base/r['label'])!=r['label_sha256']:
            raise ValueError(f'Base label modified: {r["id"]}')
    new, receipt = [], []
    for r in csv_rows(targeted/'manifest.csv'):
        p = targeted/r['annotation_json']
        d = json.loads(p.read_text())
        if (d.get('imageWidth'),d.get('imageHeight'))!=(640,480) or d.get('imagePath')!=Path(r['image']).name:
            raise ValueError(f'Annotation does not match image: {r["id"]}')
        if digest(targeted/r['image'])!=r['image_sha256']:
            raise ValueError(f'Image checksum mismatch: {r["id"]}')
        text = json_to_yolo(p)
        decision = decide(r,text,excluded)
        sha = digest(p)
        receipt.append({'id':r['id'],'split':r['split'],'kind':r['kind'],'review_batch':r['review_batch'],'decision':decision,
                        'json_sha256':sha,'changed_from_initial':sha!=initial[r['id']],
                        'saved_at':datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(),
                        'exclude_reason':excluded.get(r['id'],'')})
        if decision!='exclude':
            new.append({**r,'text':text,'decision':decision})
    if set(excluded)-{x['id'] for x in receipt}:
        raise ValueError('Exclusion ids are not in the targeted manifest')
    test_rows = [r for r in csv_rows(TEST_MANIFEST) if r['split']=='test']
    check_disjoint(base_rows+[{**r,'source':'3mdad'} for r in new]+test_rows)
    out.mkdir(parents=True)
    rows = []
    for r in base_rows:
        for rel in (r['image'],r['label']):
            (out/rel).parent.mkdir(parents=True,exist_ok=True)
        (out/r['image']).symlink_to((base/r['image']).resolve())
        shutil.copyfile(base/r['label'],out/r['label'])
        rows.append({**r,'round':'base'})
    for r in new:
        image, label = f'images/{r["split"]}/{r["id"]}.jpg', f'labels/{r["split"]}/{r["id"]}.txt'
        if (out/image).exists():
            raise ValueError(f'Duplicate id: {r["id"]}')
        (out/image).symlink_to(r['source_image'])
        (out/label).write_text(r['text'])
        snapshot = out/'reviewed_json'/f'{r["id"]}.json'
        snapshot.parent.mkdir(exist_ok=True)
        shutil.copyfile(targeted/r['annotation_json'],snapshot)
        classes = ','.join(sorted({{'0':'texting','1':'calling'}[line.split()[0]] for line in r['text'].splitlines()}))
        rows.append({'id':r['id'],'source':'3mdad','subject':r['subject'],'view':r['view'],'split':r['split'],
                     'source_action':r['source_action'],'source_behavior':r['source_behavior'],'source_image':r['source_image'],
                     'image_sha256':r['image_sha256'],'image':image,'label':label,'label_sha256':digest(out/label),
                     'annotation_json':'','annotation_status':'human_reviewed',
                     'origin':f'targeted_round2:{r["kind"]}:{r["decision"]}','original_label_source':'',
                     'annotation_classes':classes,'review_batch':r['review_batch'],'sample_rank':'','round':'targeted_round2'})
    write_csv(out/'manifest.csv',rows)
    shutil.copyfile(base/'subject_splits.json',out/'subject_splits.json')
    counts = Counter((r['split'],r['round'],r['annotation_classes'] or 'background') for r in rows)
    write_json(out/'human_review_receipt.json',{
        'confirmation':args.confirmation,'captured_at':datetime.now().astimezone().isoformat(),
        'base_dataset':str(base),'targeted_dataset':str(targeted),
        'decisions':dict(Counter(f'{x["split"]}/{x["kind"]}/{x["decision"]}' for x in receipt)),
        'changed_from_initial':dict(Counter(f'{x["review_batch"]}/{x["changed_from_initial"]}' for x in receipt)),
        'files':receipt})
    write_json(out/'summary.json',{'counts':{'/'.join(k):v for k,v in sorted(counts.items())},
               'totals':dict(Counter(r['split'] for r in rows)),
               'rule':'box->positive; empty source phone action->excluded; empty hard negative->background',
               'test_included':False,'leakage_checks':'subject and SHA-256 disjoint incl. 3MDAD test'})
    check_snapshot(BASE/'dataset_23train_3eval_1of8')
    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: texting\n  1: calling\n')
    print(json.dumps(json.loads((out/'summary.json').read_text()),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
