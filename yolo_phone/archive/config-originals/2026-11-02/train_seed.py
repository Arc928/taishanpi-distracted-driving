#!/usr/bin/env python3
"""Train a first YOLO11n seed model after labels have been reviewed."""
from pathlib import Path
import argparse
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(BASE / 'selftrain_23train_round2/data.yaml'))
    p.add_argument('--name', default='phone_yolo11n')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--model', default=str(BASE / 'runs/selftrain_23train_round2_yolo11n/weights/best.pt'))
    args = p.parse_args()
    if not Path(args.data).is_file():
        raise SystemExit('找不到训练配置，请用 --data 指定已完成标注的训练集。')
    if (BASE / 'runs' / args.name).exists():
        raise SystemExit('运行目录已存在，拒绝覆盖；请用 --name 指定新运行名。')
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=640, batch=args.batch, device='mps',
        project=str(BASE / 'runs'), name=args.name, pretrained=True,
        patience=args.patience, workers=args.workers, cache=False,
    )


if __name__ == '__main__':
    main()
