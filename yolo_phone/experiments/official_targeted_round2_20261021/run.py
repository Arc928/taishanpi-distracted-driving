"""One-off training supervisor; records completion and compares against e79/e150 on the same new val."""
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent


def save(name, data):
    target = HERE/name
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(target)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    plan = json.loads((HERE/'plan.json').read_text())
    status = {'status':'starting','started_at':now(),'supervisor_pid':os.getpid(),
              'run_dir':plan['run_dir'],'service_model_changed':False}
    save('status.json',status)
    try:
        with (HERE/'training.log').open('xb') as log:
            process = subprocess.Popen(plan['command'],cwd=plan['cwd'],stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT)
            status.update(status='running',training_pid=process.pid)
            save('status.json',status)
            returncode = process.wait()
        status.update(finished_at=now(),returncode=returncode)
        if returncode:
            status['status']='failed'
            save('status.json',status)
            return
        run = Path(plan['run_dir'])
        rows = list(csv.DictReader((run/'results.csv').open()))
        best = max(rows,key=lambda row: 0.1*float(row['metrics/mAP50(B)'])+0.9*float(row['metrics/mAP50-95(B)']))
        weights = run/'weights/best.pt'
        result = {'completed_epochs':len(rows),'best_validation':best,
                  'baselines_same_val':json.loads((HERE/plan['baseline_on_new_val']).read_text()),
                  'best_weights':str(weights),'best_weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                  'comparison_scope':'same 212-image val (200 old + 12 new front AC6); no held-out test used'}
        save('result.json',result)
        status.update(status='complete',completed_epochs=len(rows),best_epoch=int(best['epoch']),
                      best_weights=str(weights))
        save('status.json',status)
    except Exception as error:
        status.update(status='failed',finished_at=now(),error_type=type(error).__name__,error=str(error))
        save('status.json',status)
        raise


if __name__=='__main__':
    main()
