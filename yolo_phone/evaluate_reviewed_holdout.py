#!/usr/bin/env python3
"""Evaluate fixed weights on human-reviewed 3MDAD holdout boxes.

Reviewed boxes are the ground truth. A source phone action left without a box is
reported as "phone not visible" and kept out of both mAP and hit rates; it is
never treated as background. --labels-only stops before any model is run so the
reviewer can audit decisions without seeing model results.
"""
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

from core.io import csv_rows, digest, write_csv, write_json
from core.labels import NAMES, json_to_yolo
from core.metrics import VIEWS, categorize, ground_truth, image_status, iou, summarize  # noqa: F401
from core.paths import BASE
from core.registry import path as dataset
from core.splits import check_disjoint

REVIEW = dataset('mdad_review_v1')
TRAIN = dataset('train_mixed_v1')
DEFAULT_WEIGHTS = [
    'official_resume100_e79='+str(BASE/'runs/multiview_official_resume100_20261011_yolo11n/weights/best.pt'),
    'mixed_round27='+str(BASE/'runs/multiview_gemini_reviewed_20261010_yolo11n/weights/best.pt')]


def contact_sheet(rows, path, title):
    cols, w, h = 4, 320, 240
    canvas = Image.new('RGB',(cols*w,24+((len(rows)+cols-1)//cols)*(h+16)),'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((6,6),title,fill='black')
    for n,r in enumerate(rows):
        x,y = (n%cols)*w,24+(n//cols)*(h+16)
        canvas.paste(Image.open(r['path']).convert('RGB').resize((w,h)),(x,y))
        draw.text((x+4,y+h+2),f"{r['id']} {r['view']} {r['source_action']}",fill='black')
    canvas.save(path,quality=90)


def load_reviewed(split, baseline, excluded):
    rows = []
    for r in csv_rows(REVIEW/'manifest.csv'):
        if r['source']!='3mdad' or r['split']!=split:
            continue
        p = REVIEW/r['annotation_json']
        d = json.loads(p.read_text())
        if (d.get('imageWidth'),d.get('imageHeight'))!=(640,480) or d.get('imagePath')!=Path(r['image']).name:
            raise ValueError(f'Annotation does not match image: {r["id"]}')
        text = json_to_yolo(p)
        gt = ground_truth(text)
        category = categorize(r,gt,excluded)
        classes = sorted({NAMES[c] for c,_ in gt})
        base = baseline.get(r['id'],{})
        rows.append({**r,'path':str((REVIEW/r['image']).resolve()),'json_path':str(p),'json_sha256':digest(p),
                     'json_mtime':datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(),
                     'changed_from_baseline':digest(p)!=base['json_sha256'] if base else '',
                     'label_text':text,'gt':gt,'category':category,'boxes_reviewed':len(gt),
                     'gt_class':classes[0] if len(classes)==1 else 'mixed' if classes else '',
                     'exclude_reason':excluded.get(r['id'],'')})
    if not rows:
        raise ValueError(f'No 3MDAD rows for split {split}')
    if set(excluded)-{r['id'] for r in rows}:
        raise ValueError('Exclusion ids are not in this split')
    check_disjoint(csv_rows(TRAIN/'manifest.csv')+rows)
    for r in rows:
        if digest(Path(r['path']))!=r['image_sha256']:
            raise ValueError(f'Image checksum mismatch: {r["id"]}')
    return rows


def write_yolo(out, rows):
    root = out/'yolo'
    (root/'images').mkdir(parents=True)
    (root/'labels').mkdir()
    scored = [r for r in rows if r['category'] in ('positive','background')]
    for r in scored:
        (root/'images'/f'{r["id"]}.jpg').symlink_to(r['path'])
        (root/'labels'/f'{r["id"]}.txt').write_text(r['label_text'])
    yamls = {}
    for view in VIEWS:
        listing = root/f'{view}.txt'
        listing.write_text(''.join(f'{root/"images"/r["id"]}.jpg\n' for r in scored if view=='all' or r['view']==view))
        yamls[view] = root/f'{view}.yaml'
        yamls[view].write_text(f'path: {root}\ntrain: {listing}\nval: {listing}\nnames:\n  0: texting\n  1: calling\n')
    return yamls


def detection_metrics(model, yamls, project, tag):
    result = {}
    for view,yaml in yamls.items():
        m = model.val(data=str(yaml),imgsz=640,batch=16,conf=.001,iou=.7,device='mps',plots=False,
                      project=str(project),name=f'{tag}_{view}',exist_ok=False,verbose=False)
        box = m.box
        result[view] = {'precision':float(box.mp),'recall':float(box.mr),'mAP50':float(box.map50),
                        'mAP50-95':float(box.map),
                        'per_class':{NAMES[int(c)]:{'AP50':float(box.ap50[i]),'AP50-95':float(box.ap[i])}
                                     for i,c in enumerate(box.ap_class_index)}}
    return result


def image_level(model, rows):
    preds = {}
    for start in range(0,len(rows),16):
        batch = rows[start:start+16]
        for r,res in zip(batch,model.predict([r['path'] for r in batch],imgsz=640,conf=.1,iou=.5,device='mps',verbose=False)):
            preds[r['id']] = [{'class_id':int(c),'confidence':float(s),'xyxy':b} for b,s,c in
                              zip(res.boxes.xyxy.cpu().tolist(),res.boxes.conf.cpu().tolist(),res.boxes.cls.cpu().tolist())]
    return preds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',choices=('test','val'),default='test')
    parser.add_argument('--baseline',type=Path,default=dataset('mdad_test_review_baseline'))
    parser.add_argument('--weights',nargs='+',default=DEFAULT_WEIGHTS,help='tag=path; fixed before looking at results')
    parser.add_argument('--confirmation',default='',help='Human statement that review of this split is complete')
    parser.add_argument('--labels-only',action='store_true',help='Audit review decisions without running any model')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output exists; refusing overwrite')
    if not args.labels_only and not args.confirmation.strip():
        raise ValueError('Explicit human-review completion confirmation is required')
    baseline, excluded = {}, {}
    if args.split=='test':
        baseline = json.loads((args.baseline/'baseline.json').read_text())['files']
        for e in csv_rows(args.baseline/'exclusions.csv'):
            if not e['reason'].strip():
                raise ValueError(f'Exclusion needs a reason: {e["id"]}')
            excluded[e['id']] = e['reason']
    rows = load_reviewed(args.split,baseline,excluded)
    out.mkdir(parents=True)
    decisions = [{k:r[k] for k in ('id','subject','view','source_action','source_behavior','review_batch','category',
                                   'gt_class','boxes_reviewed','changed_from_baseline','json_mtime','exclude_reason')}
                 for r in rows]
    write_csv(out/'review_decisions.csv',decisions)
    counts = {'/'.join(map(str,k)):v for k,v in sorted(Counter((r['view'],r['source_behavior'],r['category']) for r in rows).items())}
    corrected = [r['id'] for r in rows if r['category']=='positive' and r['source_behavior']!=r['gt_class']]
    unchanged = sum(r['changed_from_baseline'] is False for r in rows)
    not_visible = [r for r in rows if r['category']=='not_visible']
    if not_visible:
        contact_sheet(not_visible,out/'not_visible_check.jpg','Source phone action, no reviewed box -> phone not visible?')
    receipt = {'split':args.split,'confirmation':args.confirmation,'captured_at':datetime.now().astimezone().isoformat(),
               'images':len(rows),'category_counts':counts,'class_differs_from_source_action':corrected,
               'json_unchanged_from_baseline':unchanged if baseline else 'no baseline for this split',
               'annotation_sha256':{r['id']:r['json_sha256'] for r in rows}}
    write_json(out/'review_receipt.json',receipt)
    if args.labels_only:
        print(json.dumps({k:v for k,v in receipt.items() if k!='annotation_sha256'},ensure_ascii=False,indent=2))
        return
    yamls = write_yolo(out,rows)
    results = {}
    for spec in args.weights:
        tag,path = spec.split('=',1)
        path = Path(path).resolve()
        weights_hash = digest(path)
        model = YOLO(str(path))
        if model.names!=NAMES:
            raise ValueError(f'Incorrect class mapping: {tag}')
        detection = detection_metrics(model,yamls,out/'ultralytics',tag)
        preds = image_level(model,rows)
        report = []
        for r in rows:
            shown = [p for p in preds[r['id']] if p['confidence']>=.5]
            report.append({'id':r['id'],'subject':r['subject'],'view':r['view'],'source_action':r['source_action'],
                           'category':r['category'],'gt_class':r['gt_class'],'boxes_reviewed':r['boxes_reviewed'],
                           'predictions':len(shown),'top_prediction':NAMES[max(shown,key=lambda p:p['confidence'])['class_id']] if shown else '',
                           'status':image_status(r['category'],r['gt'],shown) if r['category']!='exclude' else 'excluded'})
        write_csv(out/f'{tag}_per_image.csv',report)
        write_json(out/f'{tag}_raw_predictions.json',preds)
        if digest(path)!=weights_hash:
            raise ValueError(f'Weights changed during evaluation: {tag}')
        results[tag] = {'weights':str(path),'weights_sha256':weights_hash,'detection':detection,
                        'image_level_at_0.5':summarize([{**r,'status':s['status']} for r,s in zip(rows,report)])}
    if any(digest(Path(r['json_path']))!=r['json_sha256'] or digest(Path(r['path']))!=r['image_sha256'] for r in rows):
        raise ValueError('Annotations or images changed during evaluation')
    write_json(out/'summary.json',{
        'split':args.split,'imgsz':640,'detection':'Ultralytics val conf=0.001 iou=0.7 on reviewed positives+backgrounds',
        'image_level':'conf>=0.5, NMS iou=0.5; localized_hit needs every reviewed box matched by same class IoU>=0.5',
        'not_visible':'source phone action without reviewed box; excluded from mAP and hit rates, reported separately',
        'category_counts':counts,'models':results,
        'model_selection':'weights fixed before evaluation; no threshold tuning or model selection on this split'})
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
