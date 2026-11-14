"""Compare fixed weights on the new 212-image val: detection mAP (all/front/side) and @0.5 image-level hits."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from ultralytics import YOLO

from continue_split import csv_rows
from evaluate_reviewed_holdout import detection_metrics, ground_truth, image_level, image_status, summarize, write_yolo
from annotation_io import NAMES
from prepare_multiview import BASE, write_json

DATA = BASE/'selftrain_multiview_targeted_round2_20261021'
RUN = BASE/'runs/multiview_targeted_round2_20261021_yolo11n/weights'
WEIGHTS = {'e79':BASE/'runs/multiview_official_resume100_20261011_yolo11n/weights/best.pt',
           'e150_last':BASE/'runs/multiview_official_resume150_20261012_yolo11n/weights/last.pt',
           'new_best_e51':RUN/'best.pt','new_last_e80':RUN/'last.pt'}
OUT = Path(__file__).resolve().parent/'val_compare'


def main():
    if OUT.exists():
        raise SystemExit('Output exists; refusing overwrite')
    rows = []
    for r in csv_rows(DATA/'manifest.csv'):
        if r['split']!='val':
            continue
        text = (DATA/r['label']).read_text()
        gt = ground_truth(text)
        classes = sorted({NAMES[c] for c,_ in gt})
        rows.append({**r,'path':str((DATA/r['image']).resolve()),'label_text':text,'gt':gt,
                     'category':'positive' if gt else 'background',
                     'gt_class':classes[0] if len(classes)==1 else 'mixed' if classes else '',
                     'new_round':r['round']=='targeted_round2'})
    OUT.mkdir()
    yamls = write_yolo(OUT,rows)
    report = {'val_images':len(rows),'threshold':0.5,'weights':{k:str(v) for k,v in WEIGHTS.items()},'results':{}}
    for tag,w in WEIGHTS.items():
        model = YOLO(str(w))
        preds = image_level(model,rows)
        for r in rows:
            r['status'] = image_status(r['category'],r['gt'],[p for p in preds[r['id']] if p['confidence']>=.5])
        new12 = Counter(r['status'] for r in rows if r['new_round'])
        report['results'][tag] = {'detection':detection_metrics(model,yamls,OUT/'val_runs',tag),
                                  'image_level':summarize(rows),'new_front_ac6_12':dict(new12),
                                  'front_ac6_all':dict(Counter(r['status'] for r in rows
                                                               if r['view']=='front' and r['source_action']=='AC6' and r['category']=='positive')),
                                  'false_positive_actions':dict(Counter(r['source_action'] for r in rows if r['status']=='false_positive'))}
    write_json(OUT/'report.json',report)
    for tag,res in report['results'].items():
        d,i = res['detection'],res['image_level']['all']
        print(tag,{v:round(d[v]['mAP50'],3) for v in d},round(d['all']['mAP50-95'],3),
              'tex',f"{i['texting']['localized_hit']}/{i['texting']['total']}",'call',f"{i['calling']['localized_hit']}/{i['calling']['total']}",
              'fp',f"{i['background']['false_positive']}/{i['background']['total']}",'AC6front',res['front_ac6_all'],'fpact',res['false_positive_actions'])


if __name__=='__main__':
    main()
