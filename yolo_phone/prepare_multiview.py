#!/usr/bin/env python3
"""Build a review-gated State Farm + 3MDAD detection dataset, without training."""
import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from core.io import csv_rows, digest, write_csv, write_json
from core.labels import BACKGROUND, NAMES, POSITIVE, validate_label
from core.paths import BASE, artifact
from core.registry import path as dataset
from core.splits import EVAL_DRIVERS, check_disjoint, check_snapshot, select_frames, split_subjects


def inventory(source):
    rows = []
    for line in (source/'files_sha256.jsonl').open():
        entry = json.loads(line)
        if not entry['path'].startswith(('Day-RGB1/', 'Day-RGB2/')):
            continue
        view_dir, camera, subject, action_dir, name = Path(entry['path']).parts
        match = re.fullmatch(r'RGB([12])_SUB(\d+)ACT(\d+)F(\d+)\.jpg', name)
        if not match:
            raise ValueError(f'Unrecognized source name: {name}')
        view, person, action, frame = map(int, match.groups())
        if (camera, subject, action_dir, view_dir) != (f'RGB{view}', f'S{person}', f'AC{action}', f'Day-RGB{view}'):
            raise ValueError('Source path disagrees with filename')
        rows.append({'source': '3mdad', 'subject': subject, 'view': 'side' if view == 1 else 'front',
                     'source_action': f'AC{action}', 'frame': frame,
                     'source_image': str(source/entry['path']), 'image_sha256': entry['sha256'],
                     'source_behavior': POSITIVE.get(action, 'background_candidate' if action in BACKGROUND else 'ambiguous'),
                     'sampling_policy': 'eligible' if action in POSITIVE or action in BACKGROUND else 'exclude_gps_or_photography'})
    if len(rows) != 222034:
        raise ValueError('Downloaded RGB inventory is incomplete')
    return rows


def prefill(out, rows, weights):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if model.names != NAMES:
        raise ValueError('Model class mapping differs from project mapping')
    train = [r for r in rows if r['source'] == '3mdad' and r['split'] == 'train']
    proposals, snapshots = [], {}
    for start in range(0, len(train), 16):
        batch = train[start:start+16]
        results = model.predict([r['source_image'] for r in batch], imgsz=640,
                                conf=.1, iou=.5, device='mps', verbose=False)
        for row, result in zip(batch, results):
            path = out/row['annotation_json']
            data = json.loads(path.read_text())
            if tuple(result.orig_shape) != (480, 640):
                raise ValueError('Unexpected source dimensions')
            boxes, shapes = [], []
            for xyxy, confidence, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                cls = int(cls)
                boxes.append({'class_id': cls, 'confidence': confidence, 'xyxy': xyxy})
                if confidence >= .5:
                    x1,y1,x2,y2 = xyxy
                    shapes.append({'label': NAMES[cls], 'shape_type': 'rectangle',
                                   'points': [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                                   'group_id': None, 'flags': {}, 'score': confidence,
                                   'description': 'MODEL PROPOSAL ONLY — requires human review'})
            data['shapes'] = shapes
            write_json(path, data)
            snapshots[row['id']] = {'json_sha256': digest(path), 'proposal_count': len(shapes)}
            proposals.append({'id': row['id'], 'split': 'train', 'source_behavior': row['source_behavior'], 'proposals': boxes})
        if start % 160 == 0:
            print(f'Prefill {min(start+16,len(train))}/{len(train)}', flush=True)
    write_json(out/'train_proposals.json', proposals)
    write_json(out/'prefill_hashes.json', snapshots)
    write_json(out/'model_provenance.json', {'weights':str(weights), 'sha256':digest(weights),
               'imgsz':640, 'raw_threshold':.1, 'display_threshold':.5, 'split':'train',
               'auto_accepted_count':0, 'val_test_prefilled':False})


def build(args):
    out, source = args.output.resolve(), args.source.resolve()
    sf = dataset('statefarm_base_1of8')
    old = artifact('selftrain_23train_round2')
    if out.exists():
        raise ValueError(f'Refusing to overwrite {out}')
    check_snapshot(sf)
    full = inventory(source)
    assignments = split_subjects({r['subject'] for r in full}, args.seed)
    for row in full:
        row['split'] = assignments[row['subject']]
    check_disjoint(full)
    groups = defaultdict(list)
    for row in full:
        if row['sampling_policy'] == 'eligible':
            groups[(row['subject'],row['view'],row['source_action'])].append(row)
    sampled = []
    for key in sorted(groups):
        positive = groups[key][0]['source_behavior'] in ('texting','calling')
        for rank, row in enumerate(select_frames(groups[key], 3 if positive else 1)):
            sampled.append({**row, 'sample_rank': rank if positive else 3})
    # Verify source bytes for every selected image, not just downloaded metadata.
    for row in sampled:
        if digest(Path(row['source_image'])) != row['image_sha256']:
            raise ValueError('Selected image differs from download checksum')
    sf_base = {r['image']:r for r in csv_rows(sf/'manifest.csv')}
    old_rows = csv_rows(old/'manifest.csv')
    for r in old_rows:
        if sf_base[r['image']]['split'] != 'train' or r['subject'] in EVAL_DRIVERS:
            raise ValueError('Invalid inherited training image')
        if digest(old/r['label']) != r['label_sha256']:
            raise ValueError('Inherited label checksum mismatch')
        validate_label((old/r['label']).read_text(), r)
        if (old/r['image']).resolve() != (sf/r['image']).resolve():
            raise ValueError('Inherited image source mismatch')
    out.mkdir(parents=True)
    write_json(out/'BUILDING.json', {'status':'incomplete_until_summary_is_written'})
    write_json(out/'subject_splits.json', {'seed':args.seed, '3mdad':assignments,
               'statefarm_eval':sorted(EVAL_DRIVERS), 'policy':'split by subject, all views and frames stay together'})
    write_csv(out/'source_inventory_3mdad.csv', full)
    (out/'classes.txt').write_text('texting\ncalling\n')
    rows = []
    for r in old_rows:
        stem = 'sf_'+Path(r['image']).stem
        image = f'inherited/images/train/{stem}.jpg'
        label = f'inherited/labels/train/{stem}.txt'
        for path in (out/image,out/label):
            path.parent.mkdir(parents=True,exist_ok=True)
        (out/image).symlink_to((old/r['image']).resolve())
        shutil.copyfile(old/r['label'],out/label)
        rows.append({'id':stem, 'source':'statefarm', 'subject':r['subject'], 'view':'source_camera',
                     'split':'train', 'source_action':r['source_class'], 'source_behavior':r['class_name'],
                     'source_image':r['source_image'], 'image_sha256':digest(out/image), 'image':image,
                     'label':label,'label_sha256':digest(out/label),'annotation_json':'',
                     'annotation_status':'inherited_valid', 'origin':r['origin'],
                     'original_label_source':r['label_source'], 'annotation_classes':r['annotation_classes'],
                     'review_batch':'', 'sample_rank':''})
    for split in ('train','val','test'):
        selected = sorted((r for r in sampled if r['split']==split),
                          key=lambda r:(r['sample_rank'],int(r['subject'][1:]),r['view'],r['source_action']))
        for n, r in enumerate(selected):
            stem = 'md_'+Path(r['source_image']).stem
            batch = f'{split}_{n//280+1:02d}'
            folder = out/'review'/split/batch
            folder.mkdir(parents=True,exist_ok=True)
            image = str((folder/f'{stem}.jpg').relative_to(out))
            jpath = str((folder/f'{stem}.json').relative_to(out))
            (out/image).symlink_to(r['source_image'])
            write_json(out/jpath, {'version':'4.1.0','flags':{},'checked':False,'shapes':[],
                       'imagePath':f'{stem}.jpg','imageData':None,'imageHeight':480,'imageWidth':640,
                       'description':'Pending human review. Empty shapes are NOT a background label.'})
            rows.append({'id':stem,'source':'3mdad','subject':r['subject'],'view':r['view'],
                         'split':split,'source_action':r['source_action'],'source_behavior':r['source_behavior'],
                         'source_image':r['source_image'],'image_sha256':r['image_sha256'],'image':image,
                         'label':'','label_sha256':'','annotation_json':jpath,
                         'annotation_status':'pending_human_review','origin':'3mdad_weak_action_class_only',
                         'original_label_source':'','annotation_classes':'','review_batch':batch,'sample_rank':r['sample_rank']})
    statefarm_eval = []
    for r in sf_base.values():
        if r['split'] != 'val':
            continue
        image = f'eval_statefarm/images/{Path(r["image"]).name}'
        (out/image).parent.mkdir(parents=True,exist_ok=True)
        (out/image).symlink_to((sf/r['image']).resolve())
        statefarm_eval.append({'id':'sf_'+Path(r['image']).stem,'source':'statefarm','subject':r['subject'],
                              'split':'statefarm_eval','source_behavior':r['class_name'],'image':image,
                              'image_sha256':digest(out/image),'annotation_status':'original_image_level_eval_only'})
    check_disjoint(rows+statefarm_eval)
    write_csv(out/'manifest.csv',rows)
    write_csv(out/'eval_statefarm/manifest.csv',statefarm_eval)
    pending = [r for r in rows if r['source']=='3mdad']
    write_csv(out/'review_status.csv',[{'id':r['id'],'split':r['split'],'review_batch':r['review_batch'],
               'source_behavior':r['source_behavior'],'decision':'pending','note':''} for r in pending])
    # Freeze image/split selection; editable annotations remain separate.
    shutil.copyfile(sf/'eval_files_sha256.json',out/'statefarm_eval_original_snapshot.json')
    if args.prefill:
        prefill(out,rows,args.weights.resolve())
    write_json(out/'review_initial_hashes.json', {r['id']:digest(out/r['annotation_json']) for r in pending})
    write_json(out/'selection_hashes.json', {name:digest(out/name) for name in
               ('manifest.csv','subject_splits.json','source_inventory_3mdad.csv','eval_statefarm/manifest.csv')})
    check_snapshot(sf)
    summary = {'status':'awaiting_human_boxes_not_training_ready','seed':args.seed,'imgsz':640,
               'inherited_train':len(old_rows),'new_review':len(pending),
               'new_by_split':dict(Counter(r['split'] for r in pending)),
               'new_by_split_behavior':dict(Counter(r['split']+'/'+r['source_behavior'] for r in pending)),
               'new_by_view':dict(Counter(r['view'] for r in pending)),
               'inherited_annotation_classes':dict(Counter(r['annotation_classes'] or 'background' for r in rows if r['source']=='statefarm')),
               'original_statefarm_eval_count':len(statefarm_eval),'eval_files_unchanged':True,
               'subject_and_exact_hash_leakage_checks':'passed', 'prefill_train_only':args.prefill,
               'new_auto_accepted':0,'original_annotations':str(source/'Annotations 3MDAD')}
    (out/'README.md').write_text(README.format(out=out),encoding='utf-8')
    write_json(out/'summary.json',summary)
    (out/'BUILDING.json').unlink()
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


README = '''# State Farm + 3MDAD 多视角数据集 v1

这是已完成划分、抽样和预标注的待复核数据集，尚不能直接训练。

- 继承 State Farm 1534 张有效训练样本，保留原标签及其人工/伪标签来源，不将伪标签升级成人工真值。
- 3MDAD 50 人按 seed=42 固定为 35 train / 5 val / 10 test；每位司机所有视角、所有原始帧同组。
- 每个司机、每个视角，AC5/6/7/8 每个动作取 3 个等分区间的中点帧；10 类背景候选各取 1 帧，共 44 张/人、2200 张。
- AC4 操作 GPS、AC9 拍照暂不采样；完整 222034 张来源及划分在 source_inventory_3mdad.csv。
- 同步双视角不会跨 split；按司机和图片 SHA-256 检查泄漏。原 State Farm p016/p021/p026 完全隔离。
- 两类固定 0 texting / 1 calling；训练和推理尺寸 640。

## 标注

在 X-AnyLabeling 打开 `{out}/review/train/train_01`（第一批 280 张，覆盖 35 位司机、两个视角和四个手机动作）。
其他批次在 review/train/ 下。验证、测试分别在 review/val/、review/test/，按原图独立标注，未提供模型预填。

texting 框持机手和手机，不含扶方向盘的手；calling 框持机手、手机及附近脸部，不框整个人。只识别司机。
训练预填框仅供辅助，必须逐张核对。空 shapes、没有模型预测，均不表示已确认背景。
原始 .mat 的头/手框不是行为框，未自动转换。

JSON 保存后，在 review_status.csv 将相应 decision 改为：

- `positive`：存在正确的 texting/calling 人工框；允许人工纠正弱行为类别。
- `background`：人工确认无目标，且 JSON 无框；仅允许源背景候选。
- `exclude`：遮挡、动作过渡、不可判断或无框源正样本；note 写明原因。
- `pending`：尚未完成。

只改 JSON 不代表已审核。最终导出还需记录用户明确的复核完成确认。
源 AC5–AC8 没有框必须排除，不会导出为空背景标签。
val 用于新训练的模型选择；test 是最终独立测试，不能用于调阈值或反复挑模型。
原 State Farm 452 张在 eval_statefarm/，仅保持原图像级评估，不能当检测 mAP 真值。

## 导出

使用 `export_multiview.py --dataset <本目录> --output <新目录> --confirmation <明确的复核完成记录>`。
导出需至少有已复核的 3MDAD train 和完整已复核的 val；test 全部复核后才会写入 YOLO test 配置。
未审核图片不会出现在导出的训练目录，已有输出拒绝覆盖。

## 文件说明

- manifest.csv：3734 张选样清单，来源、划分、框状态与路径。
- inherited/：1534 张原训练图片软链接和标签副本。
- review/：2200 张新数据软链接及独立可编辑 JSON；没有未审核 YOLO 空标签。
- train_proposals.json、model_provenance.json：训练区模型预测与权重哈希（若启用预填）。
- selection_hashes.json：划分与选样冻结校验；review_initial_hashes.json：初始 JSON 校验。

3MDAD 原始数据位于 /Users/arc928/Downloads/3MDAD；许可 CC BY-NC-ND 4.0，教育/研究、非商业用途。
本次构建没有修改原数据、现有训练集、独立评估文件，也没有启动训练。
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/Users/arc928/Downloads/3MDAD'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--prefill',action='store_true')
    parser.add_argument('--weights',type=Path,default=BASE/'runs/selftrain_23train_round2_yolo11n/weights/best.pt')
    build(parser.parse_args())


if __name__ == '__main__':
    main()
