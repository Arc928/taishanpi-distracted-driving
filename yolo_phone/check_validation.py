#!/usr/bin/env python3
"""Predict held-out drivers without writing to validation images or labels.

Ground truth here is image-level State Farm behavior labels. Diagnostics and
the top-score confusion matrix assess image behavior, NOT box localization.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image,ImageDraw
from ultralytics import YOLO

from core import config as current_config
from core.io import csv_rows
from core.labels import NAMES as CLASS_NAMES
from core.registry import path as dataset
from core.splits import snapshot as val_snapshot

BASE=Path(__file__).resolve().parent
ROOT=dataset('statefarm_base_1of8')
NAMES=(*CLASS_NAMES.values(),'background')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--weights',type=Path,default=None,help='默认为 config/current.yaml 的当前权重')
    parser.add_argument('--output',type=Path,default=BASE/'validation_new')
    parser.add_argument('--conf',type=float,default=0.5)
    parser.add_argument('--dataset',type=Path,default=ROOT)
    args=parser.parse_args()
    out=args.output.resolve()
    root=args.dataset.resolve()
    if out.exists():
        raise SystemExit('输出目录已存在，拒绝覆盖。请用 --output 指定新目录。')
    if not 0.1<=args.conf<=1:
        parser.error('--conf 必须在 0.1 到 1 之间')
    before=val_snapshot(root)
    rows=[r for r in csv_rows(root/'manifest.csv') if r['split']=='val']
    train_subjects={r['subject'] for r in csv_rows(root/'manifest.csv') if r['split']=='train'}
    assert not train_subjects & {r['subject'] for r in rows}
    weights=args.weights.resolve() if args.weights else current_config.weights_path(verify=True)
    model=YOLO(str(weights))
    assert model.names==CLASS_NAMES
    for sub in ('predictions','xlabel','review/sheets'):
        (out/sub).mkdir(parents=True)
    raw=[]
    report=[]
    for i,row in enumerate(rows,1):
        src=root/row['image']
        result=model.predict(str(src),imgsz=current_config.imgsz(),conf=0.1,iou=current_config.nms_iou(),device='mps',verbose=False)[0]
        height,width=result.orig_shape
        boxes=[{'class_id':int(c),'confidence':float(s),'xyxy':xy}
               for xy,s,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]
        visible=[b for b in boxes if b['confidence']>=args.conf]
        detected={b['class_id'] for b in visible}
        highest=max(visible,key=lambda b:b['confidence'],default=None)
        predicted=NAMES[highest['class_id']] if highest else 'background'
        if row['class_name']=='background':
            status='background_false_positive' if visible else 'background_no_detection'
        elif not visible:
            status='positive_missed'
        elif any(c!=int(row['class_id']) for c in detected):
            status='class_conflict'
        else:
            status='positive_candidate'
        raw.append({'image':row['image'],'subject':row['subject'],
                    'source_behavior':row['class_name'],'proposals':boxes})
        report.append({'image':row['image'],'subject':row['subject'],'source_class':row['source_class'],
                       'source_behavior':row['class_name'],'predicted_behavior':predicted,
                       'boxes':len(visible),'max_conf':highest['confidence'] if highest else 0,
                       'status':status,'detected_classes':','.join(map(str,sorted(detected)))})
        # Keep the selected confidence threshold identical in labels and previews.
        result.boxes=result.boxes[result.boxes.conf>=args.conf]
        result.save(filename=str(out/'predictions'/src.name))
        shapes=[{'label':NAMES[b['class_id']],'shape_type':'rectangle',
                 'points':[[b['xyxy'][0],b['xyxy'][1]],[b['xyxy'][2],b['xyxy'][1]],
                           [b['xyxy'][2],b['xyxy'][3]],[b['xyxy'][0],b['xyxy'][3]]],
                 'score':b['confidence'],'flags':{},'group_id':None} for b in visible]
        (out/'xlabel'/src.name).symlink_to(src.resolve())
        (out/'xlabel'/f'{src.stem}.json').write_text(json.dumps({
            'version':'4.1.0','flags':{},'checked':False,'shapes':shapes,
            'imagePath':src.name,'imageHeight':height,'imageWidth':width,'imageData':None,
        },indent=2))
        if i%100==0:
            print(f'{i}/{len(rows)}',flush=True)
    (out/'raw_predictions.json').write_text(json.dumps(raw))
    for filename,values in (
        ('report.csv',report),
        ('review_priority.csv',[r for r in report if r['status'] in ('positive_missed','class_conflict','background_false_positive')]),
    ):
        with (out/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(values)
    diagnostics=[]
    confusion={}
    for threshold in (0.25,0.5,0.75,0.8):
        matrix=[[0]*3 for _ in NAMES]
        for row in raw:
            visible=[b for b in row['proposals'] if b['confidence']>=threshold]
            best=max(visible,key=lambda b:b['confidence'],default=None)
            pred=best['class_id'] if best else 2
            matrix[NAMES.index(row['source_behavior'])][pred]+=1
        confusion[str(threshold)]=matrix
        for name in NAMES:
            group=[r for r in raw if r['source_behavior']==name]
            with_any=correct=wrong=0
            for row in group:
                classes={b['class_id'] for b in row['proposals'] if b['confidence']>=threshold}
                with_any+=bool(classes)
                if name!='background':
                    expected=NAMES.index(name)
                    correct+=expected in classes
                    wrong+=bool(classes-{expected})
            diagnostics.append({'threshold':threshold,'source_behavior':name,
                                'total':len(group),'with_any_box':with_any,
                                'with_expected_class':correct,'with_wrong_class':wrong})
    with (out/'threshold_diagnostics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(diagnostics[0]));writer.writeheader();writer.writerows(diagnostics)
    by_driver=[]
    for subject in sorted({r['subject'] for r in report}):
        for name in NAMES:
            group=[r for r in report if r['subject']==subject and r['source_behavior']==name]
            by_driver.append({'subject':subject,'source_behavior':name,'total':len(group),
                              'with_any_box':sum(r['boxes']>0 for r in group),
                              'top_class_correct':sum(r['predicted_behavior']==name for r in group)})
    with (out/'by_driver.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(by_driver[0]));writer.writeheader();writer.writerows(by_driver)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matrix=confusion[str(args.conf)] if str(args.conf) in confusion else None
    if matrix is not None:
        fig,ax=plt.subplots(figsize=(6,5));ax.imshow(matrix,cmap='Blues')
        ax.set_xticks(range(3),NAMES);ax.set_yticks(range(3),NAMES)
        ax.set_xlabel('Highest-score predicted image behavior');ax.set_ylabel('Source image behavior')
        ax.set_title(f'Held-out drivers, confidence >= {args.conf:.2f}\nImage labels only; not box accuracy')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,str(matrix[i][j]),ha='center',va='center',color='white' if matrix[i][j]>max(map(max,matrix))/2 else 'black')
        fig.tight_layout();fig.savefig(out/'confusion_matrix.png',dpi=160);plt.close(fig)
    rng=random.Random(42);selected=[]
    for behavior,status,num in (
        ('texting','positive_candidate',12),('calling','positive_candidate',12),
        ('background','background_false_positive',12),('texting','positive_missed',6),
        ('calling','positive_missed',6),('background','background_no_detection',6)):
        pool=[r for r in report if r['source_behavior']==behavior and r['status']==status]
        selected.extend(rng.sample(pool,min(num,len(pool))))
    with (out/'review/sample.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(report[0]));writer.writeheader();writer.writerows(selected)
    for start in range(0,len(selected),6):
        canvas=Image.new('RGB',(1280,1530),'white');d=ImageDraw.Draw(canvas)
        for i,row in enumerate(selected[start:start+6]):
            name=Path(row['image']).name;x=i%2*640;y=i//2*510
            d.text((x+5,y+5),f"{start+i:02d} {row['source_behavior']} {row['status']} {name}",fill='black')
            canvas.paste(Image.open(out/'predictions'/name).convert('RGB'),(x,y+30))
        canvas.save(out/'review/sheets'/f'{start//6:02d}.jpg',quality=94)
    preview_rows=[]
    for behavior,status in [('texting','positive_candidate'),('calling','positive_candidate'),
                            ('texting','positive_missed'),('background','background_false_positive')]:
        row=next((r for r in selected if r['source_behavior']==behavior and r['status']==status),None)
        if row:preview_rows.append(row)
    preview=Image.new('RGB',(1280,1020),'white');d=ImageDraw.Draw(preview)
    for i,row in enumerate(preview_rows):
        x=i%2*640;y=i//2*510;name=Path(row['image']).name
        d.text((x+5,y+5),f"{row['source_behavior']} {row['status']} {name}",fill='black')
        preview.paste(Image.open(out/'predictions'/name).convert('RGB'),(x,y+30))
    preview.save(out/'review/preview.jpg',quality=94)
    after=val_snapshot(root)
    assert before==after,'验证集被修改'
    summary={'weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
             'dataset':str(root),
             'image_count':len(rows),'subjects':sorted({r['subject'] for r in report}),
             'confidence':args.conf,'status_counts':dict(Counter(r['status'] for r in report)),
             'diagnostics':diagnostics,'image_level_confusion':confusion,'validation_files_unchanged':True,
             'limitations':'Image-level source class labels only. No positive ground-truth boxes; no detection mAP or localization recall.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'val_files_sha256.json').write_text(json.dumps(before,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
