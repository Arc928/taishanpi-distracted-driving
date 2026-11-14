#!/usr/bin/env python3
"""Prepare a YOLO dataset from State Farm distracted-driver classification data.

The source dataset has image-level labels only. Positive samples are queued for
manual box annotation; other classes get empty background label files. No
bounding boxes are invented. Image numbers are not treated as frame timestamps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
from pathlib import Path

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # builders/ 下的脚本需能导入 yolo_phone/core

from core.labels import NAMES
from core.sources import root as source_root


CLASS_MAP = {
    "c1": 0,  # texting with right hand
    "c3": 0,  # texting with left hand
    "c2": 1,  # talking on phone with right hand
    "c4": 1,  # talking on phone with left hand
}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Symlinks keep the generated dataset small and avoid duplicating 4 GB.
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def choose_val_subjects(subjects: list[str], fraction: float) -> set[str]:
    """Select a deterministic subject holdout, independent of row order."""
    if len(subjects) < 2:
        raise ValueError("至少需要两位司机才能划分 train/val")
    count = min(len(subjects) - 1, max(1, round(len(subjects) * fraction)))
    ranked = sorted(
        subjects,
        key=lambda s: hashlib.sha1(s.encode("utf-8")).hexdigest(),
    )
    return set(ranked[:count])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=source_root("statefarm"),
        help="State Farm 数据集根目录（包含 driver_imgs_list.csv 和 imgs/train），默认取 config/sources.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "dataset")
    parser.add_argument(
        "--sample-every", "--stride", dest="sample_every",
        type=int,
        default=8,
        help="每个司机、每个原始类别随机保留约 1/N；默认 N=8，并非时间抽帧",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--val-subjects", nargs="+", default=None,
        help="显式指定独立评估司机（空格分隔）；指定后不使用 --val-fraction",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制图片而不是建立软链接（软链接默认更快且节省空间）",
    )
    args = parser.parse_args()

    if args.sample_every < 1:
        parser.error("--sample-every 必须 >= 1")
    if args.val_subjects is None and not 0 < args.val_fraction < 1:
        parser.error("--val-fraction 必须在 0 与 1 之间")
    args.output = args.output.expanduser().resolve()
    if args.output.exists() or args.output.is_symlink():
        parser.error("输出目录已存在；为保留标注，请用 --output 指定新目录")
    source = args.source.expanduser().resolve()
    csv_path = source / "driver_imgs_list.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        src = source / "imgs" / "train" / row["classname"] / row["img"]
        if not src.is_file():
            raise FileNotFoundError(src)
    subjects = sorted({row["subject"] for row in rows})
    if args.val_subjects is None:
        val_subjects = choose_val_subjects(subjects, args.val_fraction)
    else:
        val_subjects = set(args.val_subjects)
        unknown = val_subjects - set(subjects)
        if unknown:
            parser.error(f"评估司机不在源数据中: {sorted(unknown)}")
        if len(val_subjects) != len(args.val_subjects):
            parser.error("--val-subjects 中存在重复司机")
        if len(val_subjects) == len(subjects):
            parser.error("至少需要保留一位训练司机")
    print(f"司机数: {len(subjects)}; val 司机: {sorted(val_subjects)}")

    # Filenames provide no verified temporal order. Use reproducible sampling
    # within every subject/category, preserving representation in each group.
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["subject"], row["classname"]), []).append(row)
    selected: list[dict[str, str]] = []
    rng = random.Random(args.seed)
    for group in sorted(grouped):
        group_rows = sorted(grouped[group], key=lambda row: row["img"])
        selected.extend(rng.sample(group_rows, math.ceil(len(group_rows) / args.sample_every)))
    selected.sort(key=lambda row: (row["subject"], row["classname"], row["img"]))

    for split in ("train", "val"):
        (args.output / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": {"texting": 0, "calling": 0, "background": 0},
              "val": {"texting": 0, "calling": 0, "background": 0}}
    manifest = []
    for row in selected:
        split = "val" if row["subject"] in val_subjects else "train"
        src = source / "imgs" / "train" / row["classname"] / row["img"]
        if not src.exists():
            raise FileNotFoundError(src)
        stem = f"{row['subject']}_{row['classname']}_{Path(row['img']).stem}"
        image_dst = args.output / "images" / split / f"{stem}.jpg"
        label_dst = args.output / "labels" / split / f"{stem}.txt"
        if args.copy:
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, image_dst)
        else:
            link_or_copy(src, image_dst)

        target = CLASS_MAP.get(row["classname"])
        if target is None:
            label_dst.write_text("", encoding="utf-8")
            counts[split]["background"] += 1
        else:
            # Leave positive labels absent until actual boxes have been drawn.
            counts[split][NAMES[target]] += 1
        manifest.append({
            "subject": row["subject"], "source_class": row["classname"],
            "source_image": str(src), "split": split,
            "image": str(image_dst.relative_to(args.output)),
            "label": str(label_dst.relative_to(args.output)),
            "class_id": "" if target is None else target,
            "class_name": "background" if target is None else NAMES[target],
            "status": "background" if target is None else "needs_annotation",
        })

    for name, entries in (
        ("manifest.csv", manifest),
        ("annotation_queue.csv", [row for row in manifest if row["status"] == "needs_annotation"]),
    ):
        with (args.output / name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(entries)

    yaml_path = args.output / "data.yaml"
    yaml_path.write_text(
        "# NOT READY FOR TRAINING: annotate every positive image in annotation_queue.csv first.\n"
        f"path: {args.output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: texting\n"
        "  1: calling\n",
        encoding="utf-8",
    )
    (args.output / "split_subjects.txt").write_text(
        "val: " + ", ".join(sorted(val_subjects)) + "\n"
        "train: " + ", ".join(s for s in subjects if s not in val_subjects) + "\n",
        encoding="utf-8",
    )
    (args.output / "summary.json").write_text(
        json.dumps({
            "source": str(source),
            "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "source_count": len(rows),
            "sample_every": args.sample_every,
            "seed": args.seed,
            "sampling": "ceil(count / sample_every) per subject and source class",
            "split_method": "explicit_subjects" if args.val_subjects is not None else "sha1_rank",
            "train_subjects": [s for s in subjects if s not in val_subjects],
            "eval_subjects": sorted(val_subjects),
            "eval_split_directory": "val",
            "selected_count": len(selected),
            "counts": counts,
            "ready_for_training": False,
            "positive_labels": "absent until box annotation; never confirmed background",
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"抽样图片: {len(selected)}（随机保留约 1/{args.sample_every}, seed={args.seed}）")
    print(f"train: {counts['train']}")
    print(f"val:   {counts['val']}")
    print(f"已生成: {yaml_path}")
    print("尚未画框，不能直接训练：否则缺少标签的正样本也会被当成背景。")


if __name__ == "__main__":
    main()
