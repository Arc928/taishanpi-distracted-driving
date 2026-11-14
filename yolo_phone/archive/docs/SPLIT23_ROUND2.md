# 新 23/3 划分：人工复核后的第二轮

2026-10-09 目录清理后，上一轮数据集已移至 `archive/data/selftrain_23train_round1/`，
更早阶段记录位于 `archive/docs/`。本文保留训练时的路径描述；实际映射见
`archive/cleanup-2026-10-09.json`。当前脚本已支持归档定位，最新训练集及模型路径不变。

用户在当前对话明确回复“修完了”，确认 `review_23train_round1/` 的本轮复核完成。
据此读取实际保存的 JSON，而不是依赖 GUI 的 `checked` 标志。
全部样本来自 `dataset_23train_3eval_1of8/manifest.csv` 的 train；
eval 仍为 p016、p021、p026，旧数据和原始 JSON 不修改。

## 本轮复核状态

| 实际保存状态 | 张数 | 处理 |
|---|---:|---|
| 修改并画框 | 27 | 使用用户框 |
| 未修改、有预填框 | 46 | 用户确认本轮完成后接受，保留来源记录 |
| 修改并删掉背景误报框 | 3 | 空标签背景 |
| 背景控制组、未修改空框 | 51 | 背景 |
| 源类别为正样本、仍无框 | 13 | 排除，绝不当背景 |

`p014_c2_img_86001.jpg` 原整图类别为 c2，但用户实际改为 texting。
图中驾驶员将手机举在脸前，助手视觉核对与用户标签一致；用人工类别 0，保留源 c2 记录。
全程类别顺序仍为 0 texting / 1 calling，未改变映射。

127 张有效复核结果中，15 张覆盖上一轮伪标签、112 张新增，
新集为 `selftrain_23train_round2/`，共 1534 张：
**按实际检测标签** texting 507 / calling 499 / 背景 528。
按源整图类别统计为 texting 506 / calling 500 / 背景 528；两者相差来自上述人工纠正。

`human_review_receipt.json` 记录用户确认、全部 140 个 JSON 的哈希、13 张排除清单与人工类别纠正。
`reviewed_json/` 保存全部 140 个 JSON 的导出快照，包括排除样本，供追溯。
原 `review_23train_round1/xlabel/` 保持用户保存状态。

## 训练与评估

从 `runs/selftrain_23train_round1_yolo11n/weights/best.pt` 继续训练，
640、batch 16、MPS，最多 20 epoch、patience 10。
输出 `runs/selftrain_23train_round2_yolo11n/`，实际完成 11 epoch，耗时约 751 秒。
默认 patience=10 在第 11 轮触发提前停止：种子拟合指标连续 10 轮未超过第 1 轮。
按既有选择规则保留第 1 轮的 `weights/best.pt`；未根据独立 eval 选择模型或调整参数。
最佳种子拟合 mAP50≈0.994、mAP50-95≈0.698，只用于拟合保持诊断。

训练配置的 val 仍为 250 张人工种子的拟合保持检查，与 train 重叠，不能视为独立 mAP。
独立评估仅使用三位留出司机 452 张图，阈值 0.5；结果已输出到
`validation_23train_round2/`，与上一轮及旧模型在完全相同图片集合上比较。

对 train 的 2224 张预标注已另存 `remaining_23train_round2/`：
0.5 下 888 张有框、1336 张空标签，不覆盖用户 JSON，不能直接作为训练集。
原始整图标签没有真实验证框，因此评估仍仅限图像级命中与背景误报。

## 结果

| 指标 @0.5，同三位独立 eval 司机 | 补标前 | 补标后 |
|---|---:|---:|
| texting 命中 | 96/96 | 93/96 |
| calling 命中 | 87/92 | 92/92 |
| 背景误报 | 2/264 | 2/264 |

calling 多命中 5 张，texting 少命中 3 张，新增漏检均为 p016 的 c3 图片：
`p016_c3_img_16866`、`p016_c3_img_17741`、`p016_c3_img_68237`。
`p026_c2_img_69691` 虽检出 calling，还同时有 texting 错类竞争框；92/92 不代表每个框都正确。
背景误报为两张 c8 摸脸/头部动作：`p016_c8_img_76845`、`p021_c8_img_75735`。

训练司机诊断（与训练集重叠，不能用作独立泛化指标）：

| 司机 | 指标 @0.5 | 补标前 | 补标后 |
|---|---|---:|---:|
| p039 | texting 命中 | 17/17 | 16/17 |
| p039 | calling 命中 | 6/17 | 13/17 |
| p039 | 背景误报 | 0/51 | 0/51 |
| p081 | texting 命中 | 18/23 | 19/23 |
| p081 | calling 命中 | 17/22 | 21/22 |
| p081 | 背景误报 | 1/62 | 4/62 |

本轮改善了 calling，仍存在 texting 回退及 p081 背景误报增加，不能称为全面提升。
后续若继续补标，优先从 train 中查找对应混淆姿态；独立 eval 仍不参与训练。
13 张空框正样本继续排除，名单见 `selftrain_23train_round2/human_review_receipt.json`。

最终核验在 `selftrain_23train_round2/workflow_verification.json`：
1534 张标签与来源合法、23 位 train 均覆盖、eval 零泄漏；
旧 eval 1021 个文件及新 eval 716 个文件哈希一致，全部 140 个用户 JSON 与导出快照一致。

## 复现

已有目录拒绝覆盖，重跑请换新输出目录或运行名。

```bash
./.venv/bin/python yolo_phone/merge_split_review.py \
  --output yolo_phone/selftrain_23train_round2 \
  --confirmation '用户在当前对话明确回复“修完了”，确认本轮人工复核完成。'
./.venv/bin/python yolo_phone/train_seed.py \
  --data yolo_phone/selftrain_23train_round2/data.yaml \
  --model yolo_phone/runs/selftrain_23train_round1_yolo11n/weights/best.pt \
  --name selftrain_23train_round2_yolo11n --epochs 20 --batch 16
./.venv/bin/python yolo_phone/check_validation.py \
  --dataset yolo_phone/dataset_23train_3eval_1of8 \
  --weights yolo_phone/runs/selftrain_23train_round2_yolo11n/weights/best.pt \
  --output yolo_phone/validation_23train_round2
./.venv/bin/python yolo_phone/compare_split_validation.py \
  --validation yolo_phone/validation_23train_round2 \
  --previous-validation yolo_phone/validation_23train_round1 --stage new_round2
```
