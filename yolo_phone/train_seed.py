#!/usr/bin/env python3
"""Train a first YOLO11n seed model after labels have been reviewed."""
from pathlib import Path
import argparse
from ultralytics import YOLO

from core import config as current_config

BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    # 默认数据集与起点权重取自 config/current.yaml
    p.add_argument('--data', default=str(current_config.dataset_path()))
    p.add_argument('--name', default='phone_yolo11n')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--model', default=None, help='起点权重；默认为 config/current.yaml 的当前权重')
    args = p.parse_args()
    if not Path(args.data).is_file():
        raise SystemExit('找不到训练配置，请用 --data 指定已完成标注的训练集。')
    if (BASE / 'runs' / args.name).exists():
        raise SystemExit('运行目录已存在，拒绝覆盖；请用 --name 指定新运行名。')
    model = YOLO(args.model or str(current_config.weights_path(verify=True)))
    model.train(
        data=args.data, epochs=args.epochs, imgsz=current_config.imgsz(), batch=args.batch, device='mps',
        project=str(BASE / 'runs'), name=args.name, pretrained=True,
        patience=args.patience, workers=args.workers, cache=False,
    )


if __name__ == '__main__':
    main()
