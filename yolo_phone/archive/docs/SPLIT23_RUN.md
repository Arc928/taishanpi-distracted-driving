# 23 train / 3 eval：沿用人工种子与严格伪标签策略

2026-10-09，用户确认 eval 为 p016、p021、p026，其余 23 位为 train，
并要求“按照以前的策略继续”。全部后续选样仅来自
`dataset_23train_3eval_1of8/manifest.csv` 的 train。

## 标注复用与种子训练

- 复用原 150 张人工种子，核对 `human_review_receipt.json` 中全部 JSON 哈希。
- 复用第二轮 40 张修正正样本及 60 张背景，合计新种子 250 张：texting 61、calling 79、背景 110。
- 33 张未改动的 accepted_prefill 保留伪标签身份，不纳入新人工种子。
- 核对种子及第二轮当前 JSON 与旧 YOLO 导出一致；旧自训练集 1399 张可追溯复用。
- 新目录：`seed_23train_reviewed/`、`selftrain_23train_reused/`。
- 从根目录 `yolo11n.pt` 重新训练种子 30 epoch，640、batch 8、MPS。
- 种子最佳权重：`runs/seed_23train_yolo11n/weights/best.pt`；30 轮耗时约 554 秒。
- 种子最佳拟合 mAP50≈0.995、mAP50-95≈0.721；val 与 train 重叠，非独立检测精度。

## 预标注与自训练

对剩余 2224 张 train 生成 `remaining_23train_seed/`。
0.5 下 731 张有框、1493 张空标签；空标签包含漏检正样本，不能整体用作背景。

新正伪标签要求：0.5 下唯一框、类别与源类别一致、该框置信度 >=0.8。
不接受其他 >=0.5 的同类或错类竞争框。此前标注独立保留，不用新预测覆盖人工框。

- 新增 texting 14、calling 1；全部 15 框已做助手视觉检查，仍是伪标签而非人工真值。
- 新增源类别背景 8 张，模型在 0.5 下无框，维持约 2:1 正负比例。
- 新训练集 `selftrain_23train_round1/`：1422 张，texting 480 / calling 468 / 背景 474。
- p039 使用 9 张正样本，p081 使用 6 张正样本；23 位 train 司机均有样本。
- 自训练从新种子最佳权重继续，20 epoch、640、batch 16、MPS，已完成，耗时约 1292 秒。
- 最佳权重：`runs/selftrain_23train_round1_yolo11n/weights/best.pt`。
- 自训练种子拟合 mAP50≈0.995、mAP50-95≈0.711；不代表独立检测指标。
- 新模型对剩余 2224 张 train 的预标注已保存为 `remaining_23train_selftrained/`：
  0.5 下 862 张有框、1362 张空标签。该目录仍须筛选，不能直接训练。

## 人工复核

`review_23train_round1/xlabel/` 共 140 张，预填来自种子模型，标注顺序优先低置信正样本：

| 分组 | 张数 |
|---|---:|
| p039、p081 的全部正样本（含已加入的 15 张伪标签） | 79 |
| 未选入训练的其他正样本 | 7 |
| 新背景误报 | 3 |
| 背景控制组 | 51 |

均来自新 train。先检查预填框、补漏框、删误报，再保存 JSON；不可把正样本空框当背景。
`prefill_hashes.json` 保存预填状态，未修改文件不自动视为人工确认。
用户明确完成本轮复核后再导出、合并及训练；不能把本目录直接拿去训练。

## 独立评估

独立 eval 仅为 p016、p021、p026，共 452 张，固定分辨率 640、阈值 0.5。
需从旧结果中只取这三位司机作比较，避免把更换评估名单误当模型提升：

| 指标 | 旧第一轮，同三司机 | 旧第二轮，同三司机 | 新划分本轮 |
|---|---:|---:|---:|
| texting 命中 | 94/96 | 96/96 | 96/96 |
| calling 命中 | 87/92 | 90/92 | 87/92 |
| 背景误报 | 5/264 | 4/264 | 2/264 |

新模型独立评估已完成，输出 `validation_23train_round1/`。
同图片集合的对比保存为其中的 `same_cohort_comparison.csv/json`。
与旧第二轮相比 texting 持平、calling 少命中 3 张、背景误报减少 2 张，不能称为整体提升。
评估仅为图像级命中，不评估框 IoU 或真实检测 mAP。

p039、p081 现在属于 train，以下只是训练司机诊断，不能用于独立泛化结论：

| 司机 | 行为 | 种子模型 @0.5 | 自训练模型 @0.5 |
|---|---|---:|---:|
| p039 | texting | 16/17 | 17/17 |
| p039 | calling | 4/17 | 6/17 |
| p081 | texting | 12/23 | 18/23 |
| p081 | calling | 15/22 | 17/22 |

p039 calling 仍是明显盲区；后续需人工补框，而不是继续降低伪标签门槛。

最终核验记录在 `selftrain_23train_round1/workflow_verification.json`：
1422 张标签合法、23 位 train 均覆盖、三位 eval 零泄漏，
旧 eval 1021 个文件与新 eval 716 个文件快照一致。
本记录对应阶段结束时，140 张人工复核尚未确认完成。
用户随后已回复“修完了”；合并、续训及评估结果见 `SPLIT23_ROUND2.md`。

## 复现

已有输出均拒绝覆盖，重跑请换新目录与运行名。

```bash
./.venv/bin/python yolo_phone/continue_split.py migrate --output yolo_phone/selftrain_23train_reused
./.venv/bin/python yolo_phone/train_seed.py --data yolo_phone/seed_23train_reviewed/data.yaml \
  --model yolo11n.pt --name seed_23train_yolo11n --epochs 30 --batch 8
./.venv/bin/python yolo_phone/pseudo_label_remaining.py \
  --dataset yolo_phone/dataset_23train_3eval_1of8 --seed-dataset yolo_phone/seed_23train_reviewed \
  --weights yolo_phone/runs/seed_23train_yolo11n/weights/best.pt --output yolo_phone/remaining_23train_seed
./.venv/bin/python yolo_phone/continue_split.py expand --base yolo_phone/selftrain_23train_reused \
  --proposals yolo_phone/remaining_23train_seed --output yolo_phone/selftrain_23train_round1
./.venv/bin/python yolo_phone/train_seed.py --data yolo_phone/selftrain_23train_round1/data.yaml \
  --model yolo_phone/runs/seed_23train_yolo11n/weights/best.pt \
  --name selftrain_23train_round1_yolo11n --epochs 20 --batch 16
./.venv/bin/python yolo_phone/check_validation.py --dataset yolo_phone/dataset_23train_3eval_1of8 \
  --weights yolo_phone/runs/selftrain_23train_round1_yolo11n/weights/best.pt \
  --output yolo_phone/validation_23train_round1
```
