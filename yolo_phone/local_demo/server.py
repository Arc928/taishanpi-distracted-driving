#!/usr/bin/env python3
"""Local-only YOLO phone detection demo; no additional web dependencies."""
import argparse
import hashlib
import json
import math
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
import torch
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_WEIGHTS = PROJECT / 'yolo_phone/runs/multiview_targeted_round2_20261021_yolo11n/weights/best.pt'


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, content, content_type='application/json; charset=utf-8'):
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def local_request(self):
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        host = self.headers.get('Host', '')
        origin = self.headers.get('Origin')
        return host in allowed and (origin is None or origin in {f'http://{h}' for h in allowed})

    def do_GET(self):
        if not self.local_request():
            return self.respond(403, {'error': '仅支持本机访问'})
        route = urlsplit(self.path).path
        if route in ('/', '/index.html'):
            return self.respond(200, (HERE/'index.html').read_bytes(), 'text/html; charset=utf-8')
        if route == '/health':
            return self.respond(200, {'ready': True, 'pid': os.getpid(), 'model': 'YOLO11n',
                                      'classes': self.server.model.names, 'default_confidence': 0.35})
        if route == '/favicon.ico':
            return self.respond(204, b'', 'image/x-icon')
        self.respond(404, {'error': '页面不存在'})

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, {'error': '仅支持本机访问'})
        url = urlsplit(self.path)
        if url.path == '/api/shutdown':
            if self.headers.get('X-Demo-Token') != self.server.shutdown_token:
                return self.respond(403, {'error': '无效的停止请求'})
            self.respond(200, {'stopped': True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        if url.path != '/api/predict':
            return self.respond(404, {'error': '接口不存在'})
        try:
            confidence = float(parse_qs(url.query).get('confidence', ['0.35'])[0])
            length = int(self.headers.get('Content-Length', '0'))
            if not math.isfinite(confidence) or not 0.1 <= confidence <= 0.95:
                return self.respond(400, {'error': '阈值须在 0.1 到 0.95 之间'})
            if not 0 < length <= 16 * 1024 * 1024:
                return self.respond(413, {'error': '图片过大，请缩小图片后重试'})
            payload = self.rfile.read(length)
            decode_started = time.perf_counter()
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                return self.respond(400, {'error': '无法读取图片，请选择有效的图片文件'})
            height, width = frame.shape[:2]
            if width * height > 20_000_000:
                return self.respond(413, {'error': '图片分辨率过高'})
            decode_ms = (time.perf_counter()-decode_started)*1000
            started = time.perf_counter()
            with self.server.inference_lock:
                result = self.server.model.predict(frame, imgsz=640, conf=confidence, iou=0.5,
                                                   device=self.server.device, verbose=False)[0]
                detections = [{'class_id': int(cls), 'label': self.server.model.names[int(cls)],
                               'confidence': float(score), 'xyxy': box}
                              for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(),
                                                         result.boxes.conf.cpu().tolist(),
                                                         result.boxes.cls.cpu().tolist())]
            self.respond(200, {'width': width, 'height': height,
                               'detections': detections,
                               'inference_ms': round(result.speed['inference'], 1),
                               'processing_ms': round((time.perf_counter()-started)*1000, 1),
                               'decode_ms': round(decode_ms, 1),
                               'preprocess_ms': round(result.speed['preprocess'], 1),
                               'postprocess_ms': round(result.speed['postprocess'], 1)})
        except (ValueError, UnidentifiedImageError, OSError) as error:
            self.respond(400, {'error': '无法读取图片或请求参数'})
        except Exception as error:
            print(f'Inference error: {error!r}', flush=True)
            self.respond(500, {'error': '检测失败，请重试'})

    def log_message(self, format, *args):
        # Avoid logging every webcam frame; retain failures and startup messages.
        if len(args) > 1 and str(args[1]) not in ('200', '204'):
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not weights.is_file():
        raise SystemExit(f'权重不存在: {weights}')
    model = YOLO(str(weights))
    if model.names != {0: 'texting', 1: 'calling'}:
        raise SystemExit('模型类别映射错误')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.predict(Image.new('RGB', (640, 480)), imgsz=640, device=device, verbose=False)
    try:
        server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    except OSError as error:
        if error.errno not in (48, 98):
            raise
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.model = model
    server.device = device
    server.inference_lock = threading.Lock()
    server.shutdown_token = secrets.token_urlsafe(24)
    runtime = {'pid': os.getpid(), 'url': f'http://127.0.0.1:{server.server_port}',
               'weights': str(weights), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
               'shutdown_token': server.shutdown_token, 'device': device}
    runtime_file = HERE/'runtime.json'
    temporary = HERE/'runtime.json.tmp'
    temporary.write_text(json.dumps(runtime, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(runtime_file)
    print(f'READY {runtime["url"]} | device={device} | weights={weights}', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
