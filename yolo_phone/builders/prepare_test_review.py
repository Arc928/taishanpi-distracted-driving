#!/usr/bin/env python3
"""Freeze the pre-review state of the 3MDAD test annotations.

Only records hashes of the current (unreviewed) JSON files so later evaluation
can show which ones a human changed. It accepts no boxes and edits nothing.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # builders/ 下的脚本需能导入 yolo_phone/core

from core.io import csv_rows, digest, write_json
from core.registry import path as dataset

REVIEW = dataset('mdad_review_v1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output exists; refusing overwrite')
    rows = [r for r in csv_rows(REVIEW/'manifest.csv') if r['source']=='3mdad' and r['split']=='test']
    status = {r['id']:r for r in csv_rows(REVIEW/'review_status.csv')}
    if len(rows)!=440 or any(status[r['id']]['decision']!='pending' for r in rows):
        raise ValueError('Expected 440 pending 3MDAD test images')
    files = {}
    for r in rows:
        p = REVIEW/r['annotation_json']
        if digest(REVIEW/r['image'])!=r['image_sha256']:
            raise ValueError(f'Test image checksum mismatch: {r["id"]}')
        d = json.loads(p.read_text())
        files[r['id']] = {'annotation_json':r['annotation_json'],'json_sha256':digest(p),
                          'json_mtime':datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(),
                          'checked':d.get('checked'),'prefill_boxes':len(d.get('shapes',[])),
                          'review_batch':r['review_batch'],'view':r['view'],'source_behavior':r['source_behavior']}
    out.mkdir(parents=True)
    write_json(out/'baseline.json',{
        'captured_at':datetime.now().astimezone().isoformat(),'review_dir':str(REVIEW),
        'note':'Pre-review state. Prefilled boxes are Gemini/Grounding DINO candidates, not ground truth.',
        'counts':{f'{k[0]}/{k[1]}/prefill={k[2]}':v for k,v in sorted(Counter(
            (f['review_batch'],f['source_behavior'],f['prefill_boxes']>0) for f in files.values()).items())},
        'files':files})
    (out/'exclusions.csv').write_text('id,reason\n',encoding='utf-8-sig')
    print(out)


if __name__=='__main__':
    main()
