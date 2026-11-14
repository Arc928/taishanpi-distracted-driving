"""Resume the original optimizer state through epoch 150 without early stopping."""
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
sys.path.insert(0,str(BASE))


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    path = HERE/name
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def worker(plan):
    import torch
    from ultralytics import YOLO
    from core.splits import check_disjoint, check_snapshot

    run = Path(plan['run_dir'])
    checkpoint = Path(plan['resume_checkpoint'])
    dataset = Path(plan['data']).parent
    if run.exists():
        raise ValueError('Refusing to overwrite run')
    assert digest(checkpoint)==plan['resume_checkpoint_sha256']
    for path,sha in plan['protected_weight_hashes'].items():
        assert digest(path)==sha
    assert digest(dataset/'manifest.csv')==plan['dataset_manifest_sha256']
    manifest = rows(dataset/'manifest.csv')
    assert sum(r['split']=='train' for r in manifest)==2952
    assert sum(r['split']=='val' for r in manifest)==200
    assert all(r['split'] in ('train','val') for r in manifest)
    for row in manifest:
        assert digest(dataset/row['image'])==row['image_sha256']
        assert digest(dataset/row['label'])==row['label_sha256']
    check_disjoint(manifest)
    check_snapshot(BASE/'data/dataset_23train_3eval_1of8')
    original = torch.load(checkpoint,map_location='cpu',weights_only=False)

    def verify_restore(trainer):
        assert trainer.save_dir.resolve()==run.resolve()
        assert trainer.resume and trainer.start_epoch==96 and trainer.epochs==150
        assert trainer.args.patience==0 and math.isinf(trainer.stopper.patience)
        assert type(trainer.optimizer).__name__=='AdamW'
        for key in ('lr0','lrf','momentum','weight_decay','warmup_epochs','warmup_bias_lr',
                    'cos_lr','mosaic','close_mosaic','optimizer'):
            assert getattr(trainer.args,key)==original['train_args'][key], key
        restored = trainer.optimizer.state_dict()
        expected = original['optimizer']
        assert len(restored['state'])==len(expected['state'])==255
        compared_tensors=0
        for key, state in expected['state'].items():
            actual=restored['state'][key]
            for field,value in state.items():
                if torch.is_tensor(value):
                    assert torch.equal(value.cpu().float(),actual[field].cpu().float()), (key,field)
                    compared_tensors+=1
                else:
                    assert value==actual[field]
        for exp,got in zip(expected['param_groups'],restored['param_groups']):
            for key in ('lr','initial_lr','betas','weight_decay'):
                assert exp.get(key)==got.get(key),(key,exp.get(key),got.get(key))
        assert trainer.ema.updates==original['updates']==4571
        assert trainer.scaler.state_dict()==original['scaler']
        assert abs(trainer.best_fitness-.60045)<1e-8
        history=[r for r in rows(Path(plan['baseline_run_dir'])/'results.csv') if int(r['epoch'])<=96]
        assert len(history)==96 and not trainer.csv.exists()
        with trainer.csv.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(history[0]))
            writer.writeheader();writer.writerows(history)
        trainer.best.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(plan['baseline_best'],trainer.best)
        save('resume_verification.json',{'verified_at':now(),'restored':True,'start_epoch':97,
             'total_epochs':150,'optimizer':'AdamW','optimizer_state_entries':255,
             'optimizer_state_tensors_equal':compared_tensors,'optimizer_groups_equal':True,
             'ema_updates':trainer.ema.updates,'scaler_state_equal':True,
             'lr0':trainer.args.lr0,'lrf':trainer.args.lrf,'early_stopping_disabled':True,
             'copied_history_epochs':96,'inherited_best_epoch':79,'inherited_best_fitness':trainer.best_fitness,
             'dataset_hashes_verified':True,'test_used':False,
             'scheduler_note':'Original linear rule and LR hyperparameters retained; horizon extends to 150, changing future per-epoch LR.'})
        print('Verified full optimizer moments, parameter groups, EMA/scaler, original LR settings; early stopping disabled.',flush=True)

    def progress(trainer):
        if trainer.csv.exists():
            history=rows(trainer.csv)
            if history:
                save('progress.json',{'updated_at':now(),'completed_epochs':int(history[-1]['epoch']),
                     'new_run_epochs':len(history)-96,'target_epochs':150,'latest_metrics':history[-1],
                     'best_validation':max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))})

    model=YOLO(str(checkpoint))
    assert model.names=={0:'texting',1:'calling'}
    model.add_callback('on_pretrain_routine_end',verify_restore)
    model.add_callback('on_fit_epoch_end',progress)
    model.train(resume=True,save_dir=str(run),patience=0,device='mps')
    assert int(rows(run/'results.csv')[-1]['epoch'])==150,'Did not finish all requested epochs'
    check_snapshot(BASE/'data/dataset_23train_3eval_1of8')
    for path,sha in plan['protected_weight_hashes'].items():
        assert digest(path)==sha
    for row in manifest:
        assert digest(dataset/row['image'])==row['image_sha256']
        assert digest(dataset/row['label'])==row['label_sha256']


def supervise(plan):
    if (HERE/'status.json').exists():
        raise ValueError('Experiment already launched')
    state={'status':'starting','started_at':now(),'supervisor_pid':os.getpid(),
           'run_dir':plan['run_dir'],'target_epochs':150,'patience':0,
           'test_used':False,'service_model_changed':False}
    save('status.json',state)
    try:
        with (HERE/'training.log').open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(HERE/'run.py'),'--worker'],
                 cwd=plan['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            state.update(status='running',training_pid=proc.pid);save('status.json',state)
            returncode=proc.wait()
        if returncode:
            raise RuntimeError(f'Training exited with code {returncode}; see training.log')
        run=Path(plan['run_dir']);history=rows(run/'results.csv')
        best=max(history,key=lambda r:float(r['metrics/mAP50-95(B)']))
        previous=plan['baseline_best_validation']
        delta=float(best['metrics/mAP50-95(B)'])-float(previous['metrics/mAP50-95(B)'])
        result={'completed_at':now(),'completed_epochs':int(history[-1]['epoch']),
                'new_run_epochs':len(history)-96,'replayed_epochs':[97,98,99,100],
                'best_epoch':int(best['epoch']),'best_validation':best,'last_validation':history[-1],
                'baseline_best_validation':previous,'map50_95_delta':delta,
                'validation_map50_95_improved':delta>0,'early_stopping_disabled':True,
                'best_weights':str(run/'weights/best.pt'),'best_weights_sha256':digest(run/'weights/best.pt'),
                'last_weights':str(run/'weights/last.pt'),'original_weights_unchanged':True,
                'test_used':False,'service_model_changed':False,
                'limitation':'Same filtered 200-image val; 20 source positives excluded. Resumes epoch 96, original optimizer/LR settings, extended 150-epoch scheduler horizon.'}
        save('result.json',result)
        state.update(status='complete',finished_at=now(),completed_epochs=150,best_epoch=int(best['epoch']),
                     validation_map50_95_improved=delta>0,best_weights=str(run/'weights/best.pt'))
        save('status.json',state)
    except Exception as error:
        state.update(status='failed',finished_at=now(),error=str(error));save('status.json',state)
        raise


if __name__=='__main__':
    plan=json.loads((HERE/'plan.json').read_text())
    if '--worker' in sys.argv:
        worker(plan)
    else:
        supervise(plan)
