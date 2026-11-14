"""Train a complete 70-epoch official-initialization experiment without early stopping."""
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    here = Path(__file__).resolve().parent
    plan = json.loads((here/'plan.json').read_text())
    base = Path(plan['cwd'])
    run = Path(plan['run_dir'])
    if run.exists():
        raise SystemExit('Refusing to overwrite an existing training run')
    model = YOLO(str(base/'yolo11n.pt'))
    model.train(
        data=str(base/'yolo_phone/selftrain_multiview_gemini_reviewed_20261010/data.yaml'),
        project=str(run.parent), name=run.name, epochs=70, patience=0,
        imgsz=640, batch=16, device='mps', workers=2, cache=False,
        pretrained=True, optimizer='AdamW', lr0=0.001667, momentum=0.9,
        warmup_bias_lr=0.0, seed=0, deterministic=True, close_mosaic=10,
        save_period=5,
    )


if __name__ == '__main__':
    main()
