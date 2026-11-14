"""Resume the intact epoch-66 checkpoint to 100, preserving prior best and metrics."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    here = Path(__file__).resolve().parent
    plan = json.loads((here/'plan.json').read_text())
    run = Path(plan['run_dir'])
    baseline = Path(plan['baseline_run_dir'])
    checkpoint = Path(plan['resume_checkpoint'])
    if run.exists():
        raise SystemExit('Refusing to overwrite an existing run')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != plan['resume_checkpoint_sha256']:
        raise ValueError('Resume checkpoint changed')

    def verify_restore(trainer):
        assert trainer.save_dir.resolve() == run.resolve()
        assert trainer.start_epoch == 66 and trainer.epochs == 100
        assert trainer.resume and trainer.args.patience == 0
        assert math.isinf(trainer.stopper.patience)
        assert len(trainer.optimizer.state) > 0
        assert trainer.ema.updates == 3183
        assert trainer.args.mosaic == 0 and trainer.args.close_mosaic == 0
        rows = list(csv.DictReader((baseline/'results.csv').open()))
        history = [r for r in rows if int(r['epoch']) <= 66]
        assert len(history) == 66
        if trainer.csv.exists():
            raise ValueError('Unexpected existing metrics in new run')
        with trainer.csv.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        trainer.best.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline/'weights/best.pt',trainer.best)
        evidence = {'restored':True,'start_epoch':67,'total_epochs':100,
                    'optimizer':type(trainer.optimizer).__name__,
                    'restored_optimizer_parameter_states':len(trainer.optimizer.state),
                    'ema_updates':trainer.ema.updates,'early_stopping_disabled':True,
                    'inherited_best_epoch':42,'inherited_best_fitness':trainer.best_fitness,
                    'copied_history_epochs':66,'mosaic_remains_disabled':True}
        (here/'resume_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print('Resume state verified: optimizer, EMA, 66-epoch history and previous best retained.',flush=True)

    model = YOLO(str(checkpoint))
    model.add_callback('on_pretrain_routine_end',verify_restore)
    model.train(resume=True,save_dir=str(run),patience=0,device='mps')


if __name__ == '__main__':
    main()
