#!/usr/bin/env python3
"""Select targeted 3MDAD review candidates for weak cases; nothing is auto-accepted.

Hard positives: extra time-spread frames of front AC6 (train + val) and side AC8 (train).
Hard negatives: train background-action frames where the fixed model fires at >=0.5.
Test subjects are never read. Output is review-only; empty JSON is not a label.
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from core.io import csv_rows, digest, write_csv, write_json
from core.labels import BACKGROUND, NAMES
from core.paths import BASE
from core.splits import check_disjoint, select_frames

INVENTORY = BASE/'multiview_statefarm_3mdad_v1/source_inventory_3mdad.csv'
REVIEW = BASE/'gemini38flash_3mdad_v1/review_dataset'
TRAIN = BASE/'selftrain_multiview_gemini_reviewed_20261010'
WEIGHTS = BASE/'runs/multiview_official_resume100_20261011_yolo11n/weights/best.pt'
POSITIVE_PLAN = [('train','front','AC6',8), ('train','side','AC8',3), ('val','front','AC6',8)]
BATCH = 280


def pick_negatives(scores, per_group, cap, min_gap):
    """scores: {(subject,view,action): [(conf, frame, row)]}; spread picks over groups, then by confidence."""
    picks = []
    for rank in range(per_group):
        tier = []
        for group,items in scores.items():
            chosen = [p for p in picks if p[0]==group]
            for conf,frame,row in sorted(items,key=lambda x:(-x[0],x[1])):
                if len(chosen)==rank and all(abs(frame-c[2])>=min_gap for c in chosen):
                    tier.append((group,conf,frame,row))
                    break
        picks += sorted(tier,key=lambda x:-x[1])[:max(0,cap-len(picks))]
    return picks


def proposals(model, rows, conf):
    out = {}
    for start in range(0,len(rows),32):
        batch = rows[start:start+32]
        for r,res in zip(batch,model.predict([r['source_image'] for r in batch],imgsz=640,conf=conf,iou=.5,device='mps',verbose=False)):
            if tuple(res.orig_shape)!=(480,640):
                raise ValueError(f'Unexpected source dimensions: {r["source_image"]}')
            out[r['source_image']] = [{'class_id':int(c),'confidence':float(s),'xyxy':b} for b,s,c in
                                      zip(res.boxes.xyxy.cpu().tolist(),res.boxes.conf.cpu().tolist(),res.boxes.cls.cpu().tolist())]
        if start%3200==0:
            print(f'{datetime.now():%H:%M:%S} predict {min(start+32,len(rows))}/{len(rows)}',flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=WEIGHTS)
    parser.add_argument('--negative-cap',type=int,default=300)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise ValueError('Output exists; refusing overwrite')
    from ultralytics import YOLO
    model = YOLO(str(args.weights.resolve()))
    if model.names!=NAMES:
        raise ValueError('Model class mapping differs from project mapping')
    weights_hash = digest(args.weights)
    taken = {r['image_sha256'] for r in csv_rows(REVIEW/'manifest.csv')}
    pool = [r for r in csv_rows(INVENTORY) if r['split'] in ('train','val') and r['image_sha256'] not in taken]
    by_seq = defaultdict(list)
    for r in pool:
        by_seq[(r['split'],r['view'],r['source_action'],r['subject'])].append(r)
    selected = []
    for split,view,action,count in POSITIVE_PLAN:
        for (s,v,a,subject),seq in sorted(by_seq.items()):
            if (s,v,a)==(split,view,action):
                selected += [{**r,'kind':'hard_positive','note':''} for r in select_frames(seq,count)]
    mining = [r for r in pool if r['split']=='train' and int(r['source_action'][2:]) in BACKGROUND]
    out.mkdir(parents=True)
    write_json(out/'BUILDING.json',{'status':'mining_in_progress','started':datetime.now().astimezone().isoformat()})
    mined = proposals(model,mining,.25)
    scores = defaultdict(list)
    for r in mining:
        top = max((p['confidence'] for p in mined[r['source_image']]),default=0)
        if top>=.5:
            scores[(r['subject'],r['view'],r['source_action'])].append((top,int(r['frame']),r))
    for group,conf,frame,r in pick_negatives(scores,2,args.negative_cap,15):
        best = max(mined[r['source_image']],key=lambda p:p['confidence'])
        selected.append({**r,'kind':'hard_negative',
                         'note':f'e79 fired {NAMES[best["class_id"]]} {best["confidence"]:.2f} at {[round(v) for v in best["xyxy"]]}'})
    prefill = proposals(model,[r for r in selected if r['kind']=='hard_positive' and r['split']=='train'],.25)
    rows = []
    groups = [('train','hard_positive','hardpos'),('train','hard_negative','hardneg'),('val','hard_positive','val')]
    for split,kind,prefix in groups:
        chosen = sorted((r for r in selected if r['split']==split and r['kind']==kind),
                        key=lambda r:(r['view'],r['source_action'],int(r['subject'][1:]),int(r['frame'])))
        for n,r in enumerate(chosen):
            batch = f'{prefix}_{n//BATCH+1:02d}'
            stem = f"md_{Path(r['source_image']).stem}"
            folder = out/'review'/split/batch
            folder.mkdir(parents=True,exist_ok=True)
            (folder/f'{stem}.jpg').symlink_to(r['source_image'])
            if digest(folder/f'{stem}.jpg')!=r['image_sha256']:
                raise ValueError(f'Source checksum mismatch: {stem}')
            shapes = []
            for p in prefill.get(r['source_image'],[]):
                x1,y1,x2,y2 = p['xyxy']
                shapes.append({'label':NAMES[p['class_id']],'shape_type':'rectangle','points':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                               'group_id':None,'flags':{},'score':p['confidence'],
                               'description':'MODEL PROPOSAL (conf>=0.25) — verify or delete'})
            description = {'hard_positive':'Source phone action. Box only if the phone is visible; empty = not visible, excluded.',
                           'hard_negative':'Hard negative: the model fired here ('+r['note']+'). Box only if the driver really uses a phone.'}[kind]
            write_json(folder/f'{stem}.json',{'version':'4.1.0','flags':{},'checked':False,'shapes':shapes,
                       'imagePath':f'{stem}.jpg','imageData':None,'imageHeight':480,'imageWidth':640,'description':description})
            rows.append({'id':stem,'source':'3mdad','subject':r['subject'],'view':r['view'],'split':split,
                         'source_action':r['source_action'],'frame':r['frame'],'source_behavior':r['source_behavior'],
                         'source_image':r['source_image'],'image_sha256':r['image_sha256'],
                         'image':str((folder/f'{stem}.jpg').relative_to(out)),'annotation_json':str((folder/f'{stem}.json').relative_to(out)),
                         'kind':kind,'review_batch':batch,'prefill_boxes':len(shapes),'note':r['note']})
    old = csv_rows(TRAIN/'manifest.csv')+[r for r in csv_rows(REVIEW/'manifest.csv') if r['split']=='test']
    check_disjoint(old+rows)
    if any(r['split']=='test' for r in rows):
        raise ValueError('Test subject selected')
    write_csv(out/'manifest.csv',rows)
    write_csv(out/'review_status.csv',[{'id':r['id'],'split':r['split'],'review_batch':r['review_batch'],'kind':r['kind'],
               'source_behavior':r['source_behavior'],'decision':'pending','note':''} for r in rows])
    write_json(out/'review_initial_hashes.json',{r['id']:digest(out/r['annotation_json']) for r in rows})
    with (out/'mining_scores.jsonl').open('w') as f:
        for r in mining:
            f.write(json.dumps({'source_image':r['source_image'],'subject':r['subject'],'view':r['view'],
                                'source_action':r['source_action'],'proposals':mined[r['source_image']]})+'\n')
    summary = {'status':'awaiting_human_review_not_training_ready','created':datetime.now().astimezone().isoformat(),
               'weights':str(args.weights.resolve()),'weights_sha256':weights_hash,'imgsz':640,
               'positive_plan':POSITIVE_PLAN,'mined_train_background_frames':len(mining),
               'mined_frames_firing_at_0.5':sum(len(v) for v in scores.values()),
               'mined_firing_by_action':dict(Counter(k[2] for k,v in scores.items() for _ in v)),
               'negative_cap':args.negative_cap,'negatives_per_group_max':2,'negative_min_frame_gap':15,
               'counts':{'/'.join(k):v for k,v in sorted(Counter((r['split'],r['kind'],r['review_batch']) for r in rows).items())},
               'prefill':'train hard positives only, model proposals conf>=0.25; val and negatives not prefilled',
               'excluded_already_selected_frames':len(taken),'test_subjects_read':False,
               'leakage_checks':'subject and SHA-256 disjoint vs current train/val and 3MDAD test','auto_accepted':0}
    write_json(out/'summary.json',summary)
    if digest(args.weights)!=weights_hash:
        raise ValueError('Weights changed during selection')
    (out/'BUILDING.json').unlink()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
