#!/usr/bin/env python3
"""Launch a persistent local demo and wait briefly for model readiness."""
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent


def ready(runtime):
    try:
        with urlopen(runtime['url']+'/health', timeout=2) as response:
            data=json.load(response)
        return data.get('ready') is True and data.get('pid') == runtime['pid']
    except Exception:
        return False


def main():
    path=HERE/'runtime.json'
    if path.exists():
        runtime=json.loads(path.read_text())
        if ready(runtime):
            print('已在运行: '+runtime['url'])
            return
    with (HERE/'server.log').open('ab') as log:
        process=subprocess.Popen([sys.executable,'-u',str(HERE/'server.py')],
                                 cwd=HERE.parents[1],stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(90):
        if process.poll() is not None:
            raise SystemExit('启动失败，请查看 local_demo/server.log')
        if path.exists():
            runtime=json.loads(path.read_text())
            if runtime['pid']==process.pid and ready(runtime):
                print('启动成功: '+runtime['url'])
                print(f'PID: {process.pid}; 推理设备: {runtime["device"]}')
                return
        time.sleep(0.5)
    raise SystemExit('启动仍在进行，请查看 local_demo/server.log')


if __name__=='__main__':
    main()
