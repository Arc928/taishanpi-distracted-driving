"""Pick one display threshold for the e51 weights on the new val, then report test once at that threshold.

Rule fixed before looking: thresholds 0.20..0.60 step 0.05; score = localized texting hits +
localized calling hits - background false positives on val; ties -> higher threshold.
Test is read only after the choice is written, from raw predictions (conf>=0.1) saved by the
earlier fixed-weight test run; nothing on test feeds back into the choice.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from ultralytics import YOLO

from core.io import csv_rows, digest, write_json
from evaluate_reviewed_holdout import ground_truth, image_level, image_status, load_reviewed, summarize
from core.labels import NAMES
from core.paths import BASE
from core.registry import path as dataset  # noqa: E402  数据集目录按 config/datasets.yaml 定位

DATA = dataset('train_targeted_round2')
WEIGHTS = BASE/'runs/multiview_targeted_round2_20261021_yolo11n/weights/best.pt'
TEST_RUN = BASE/'evals/test_reviewed_eval_targeted_round2_20261021'
TEST_REVIEW = dataset('mdad_test_review_baseline')
GRID = [round(.2+.05*i,2) for i in range(9)]
OUT = HERE/'threshold_select'


def scored(rows, preds, t):
    out = [{**r,'status':image_status(r['category'],r['gt'],[p for p in preds[r['id']] if p['confidence']>=t])} for r in rows]
    s = summarize(out)
    a = s['all']
    return s, a['texting']['localized_hit']+a['calling']['localized_hit']-a['background']['false_positive']


def main():
    if OUT.exists():
        raise SystemExit('Output exists; refusing overwrite')
    rows = []
    for r in csv_rows(DATA/'manifest.csv'):
        if r['split']=='val':
            gt = ground_truth((DATA/r['label']).read_text())
            classes = sorted({NAMES[c] for c,_ in gt})
            rows.append({**r,'path':str((DATA/r['image']).resolve()),'gt':gt,'category':'positive' if gt else 'background',
                         'gt_class':classes[0] if len(classes)==1 else 'mixed' if classes else ''})
    preds = image_level(YOLO(str(WEIGHTS)),rows)
    val = {t:scored(rows,preds,t) for t in GRID}
    chosen = max(GRID,key=lambda t:(val[t][1],t))
    OUT.mkdir()
    write_json(OUT/'val_choice.json',{'rule':__doc__.strip(),'weights':str(WEIGHTS),'weights_sha256':digest(WEIGHTS),
                                      'chosen_threshold':chosen,'val':{str(t):{'score':v[1],'summary':v[0]} for t,v in val.items()}})
    test_preds = json.loads((TEST_RUN/'targeted_round2_e51_raw_predictions.json').read_text())
    summary = json.loads((TEST_RUN/'summary.json').read_text())
    if summary['models']['targeted_round2_e51']['weights_sha256']!=digest(WEIGHTS):
        raise ValueError('Saved test predictions are from different weights')
    excluded = {e['id']:e['reason'] for e in csv_rows(TEST_REVIEW/'exclusions.csv')}
    test_rows = [r for r in load_reviewed('test',json.loads((TEST_REVIEW/'baseline.json').read_text())['files'],excluded)
                 if r['category']!='exclude']
    test = {str(t):scored(test_rows,test_preds,t)[0] for t in (0.5,chosen)}
    write_json(OUT/'test_at_chosen.json',{'chosen_on_val':chosen,'test':test,
               'note':'test reported once after choice; not used for selection'})
    for t in GRID:
        a = val[t][0]['all']
        print('val',t,'score',val[t][1],'tex',a['texting']['localized_hit'],'call',a['calling']['localized_hit'],'fp',a['background']['false_positive'],'nv' if False else '')
    print('chosen',chosen)
    for t,s in test.items():
        print('test',t,{v:(s[v]['texting']['localized_hit'],s[v]['calling']['localized_hit'],s[v]['background']['false_positive'],s[v]['not_visible']['fired']) for v in ('all','front','side')})


if __name__=='__main__':
    main()
