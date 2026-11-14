"""One-off, isolated low-learning-rate continuation with durable status/results."""
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0, str(BASE))


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    target = HERE/name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(target)


def records(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def worker(plan):
    from ultralytics import YOLO
    from continue_split import check_snapshot
    from prepare_multiview import check_disjoint

    run = Path(plan['run_dir'])
    if run.exists():
        raise ValueError('Output run exists; refusing overwrite')
    dataset = Path(plan['data']).parent
    assert sha(dataset/'manifest.csv') == plan['dataset_manifest_sha256']
    for name, expected in plan['source_weights_sha256'].items():
        assert sha(name) == expected, 'Source checkpoint changed'
    rows = records(dataset/'manifest.csv')
    assert len([r for r in rows if r['split']=='train']) == 2952
    assert len([r for r in rows if r['split']=='val']) == 200
    assert all(r['split'] in ('train','val') for r in rows)
    for row in rows:
        assert sha(dataset/row['image']) == row['image_sha256'], row['id']
        assert sha(dataset/row['label']) == row['label_sha256'], row['id']
    check_disjoint(rows)
    check_snapshot(BASE/'dataset_23train_3eval_1of8')
    model = YOLO(plan['source_last'])
    assert model.names == {0:'texting',1:'calling'}

    def verify(trainer):
        assert trainer.save_dir.resolve() == run.resolve()
        assert trainer.start_epoch == 0 and trainer.epochs == 50
        assert not trainer.resume
        assert type(trainer.optimizer).__name__ == 'AdamW'
        assert trainer.args.imgsz == 640 and trainer.args.batch == 16
        assert trainer.args.patience == 10 and trainer.args.warmup_epochs == 0
        assert trainer.args.mosaic == 0 and trainer.args.close_mosaic == 0
        assert all(abs(g['lr']-3e-5)<1e-12 for g in trainer.optimizer.param_groups)
        save('preflight.json', {'verified_at':now(),'dataset_hashes_verified':True,
             'source_weights_verified':True,'class_mapping_verified':True,
             'independent_eval_snapshot_verified':True,'new_optimizer':True,
             'optimizer':'AdamW','initial_lr':3e-5,'additional_epochs_max':50,
             'lineage_completed_epochs':100,'patience':10,'test_used':False,
             'run_dir':str(run)})
        print('Preflight passed: immutable source weights/data, new AdamW lr=3e-5, train/val only.',flush=True)

    def progress(trainer):
        # CSV has completed training epochs only; final validation does not add a row.
        if trainer.csv.exists():
            history = records(trainer.csv)
            if history:
                latest = history[-1]
                save('progress.json', {'updated_at':now(),'additional_epochs_completed':len(history),
                     'lineage_epochs_completed':100+len(history),'max_lineage_epochs':150,
                     'latest_metrics':latest,
                     'best_additional_metrics':max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))})

    model.add_callback('on_pretrain_routine_end',verify)
    model.add_callback('on_fit_epoch_end',progress)
    model.train(data=plan['data'],epochs=50,batch=16,imgsz=640,device='mps',workers=0,
                project=str(BASE/'runs'),name=run.name,exist_ok=False,
                optimizer='AdamW',lr0=3e-5,lrf=.1,momentum=.9,weight_decay=.0005,
                warmup_epochs=0,mosaic=0,close_mosaic=0,patience=10,seed=0,
                save=True,save_period=5,cache=False,resume=False,plots=True)
    check_snapshot(BASE/'dataset_23train_3eval_1of8')
    for name, expected in plan['source_weights_sha256'].items():
        assert sha(name)==expected
    for row in rows:
        assert sha(dataset/row['image'])==row['image_sha256']
        assert sha(dataset/row['label'])==row['label_sha256']


def supervise(plan):
    if (HERE/'status.json').exists():
        raise ValueError('Experiment already started; refusing duplicate launch')
    state = {'status':'starting','started_at':now(),'supervisor_pid':os.getpid(),
             'run_dir':plan['run_dir'],'source_last':plan['source_last'],
             'service_model_changed':False,'test_used':False}
    save('status.json',state)
    try:
        with (HERE/'training.log').open('xb') as log:
            process = subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),'--worker'],
                 cwd=plan['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',training_pid=process.pid)
            save('status.json',state)
            code = process.wait()
        if code:
            raise RuntimeError(f'Training exited with code {code}; see training.log')
        run = Path(plan['run_dir'])
        rows = records(run/'results.csv')
        best = max(rows,key=lambda r:float(r['metrics/mAP50-95(B)']))
        baseline = plan['baseline_best_validation']
        improved = float(best['metrics/mAP50-95(B)']) > float(baseline['metrics/mAP50-95(B)'])
        result = {'completed_at':now(),'additional_epochs':len(rows),'lineage_epochs':100+len(rows),
                  'early_stopped':len(rows)<50,'best_additional_epoch':int(best['epoch']),
                  'best_lineage_epoch':100+int(best['epoch']), 'best_validation':best,
                  'baseline_best_validation':baseline,'validation_map50_95_improved':improved,
                  'map50_95_delta':float(best['metrics/mAP50-95(B)'])-float(baseline['metrics/mAP50-95(B)']),
                  'new_best_weights':str(run/'weights/best.pt'),
                  'new_best_weights_sha256':sha(run/'weights/best.pt'),
                  'validation_preferred_weights':str(run/'weights/best.pt') if improved else plan['baseline_best'],
                  'selection_metric':'validation mAP50-95 only; test not used',
                  'original_weights_unchanged':True,'service_model_changed':False,
                  'limitation':'Same filtered 200-image val; 20 source-positive images excluded. New AdamW state; not a full-state resume.'}
        save('result.json',result)
        state.update(status='complete',finished_at=now(),additional_epochs=len(rows),
                     lineage_epochs=100+len(rows),validation_map50_95_improved=improved,
                     new_best_weights=str(run/'weights/best.pt'))
        save('status.json',state)
    except Exception as error:
        state.update(status='failed',finished_at=now(),error=str(error))
        save('status.json',state)
        raise


if __name__=='__main__':
    plan = json.loads((HERE/'plan.json').read_text())
    if '--worker' in sys.argv:
        worker(plan)
    else:
        supervise(plan)
