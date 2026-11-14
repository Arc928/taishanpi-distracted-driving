#!/usr/bin/env python3
"""Stop only this demo server using its local runtime token."""
import json
from pathlib import Path
from urllib.request import Request, urlopen

path=Path(__file__).resolve().parent/'runtime.json'
if not path.exists():
    raise SystemExit('尚未启动测试服务。')
runtime=json.loads(path.read_text())
try:
    with urlopen(Request(runtime['url']+'/api/shutdown',data=b'',method='POST',
                         headers={'X-Demo-Token':runtime['shutdown_token']}),timeout=5) as response:
        print('测试服务已停止。')
except OSError:
    raise SystemExit('服务当前未运行。')
