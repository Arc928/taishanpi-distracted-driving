# 人工修正种子后的训练记录

用户确认“种子修完了”后，150 张种子标注已从 `seed_corrected/xlabel` 导出到 `seed_reviewed`。其中 50 张 texting、50 张 calling、50 张背景。导出验证了类别、矩形框坐标及图片来源；用户保存的 JSON 保持不变。训练环境为本地 `.venv`，Ultralytics 8.4.172、PyTorch 2.14.1，设备 Apple M1 Pro/MPS。

## 第一轮：人工种子

- 训练 30 epoch、图像尺寸 640、batch 8。
- 模型：`runs/seed_reviewed_yolo11n/weights/best.pt`。
- 种子集拟合指标 mAP50=0.990、mAP50-95=0.720。训练和这里的验证图片相同，不能当作独立验证精度。
- 对剩余 2132 张 train 图片预标注，输出：`remaining_reviewed/xlabel`。

固定阈值 0.50 的整图行为标签诊断：

| 原始行为 | 图片数 | 旧模型有框图片 | 人工种子模型有框图片 |
|---|---:|---:|---:|
| texting | 417 | 79 | 392 |
| calling | 423 | 233 | 356 |
| 背景 | 1292 | 76 | 87 |

人工种子模型的 calling 有框图片中 355 张含对应类别，2 张出现非预期类别（其中一张同时有对应类别）。这些计数依据原始整图分类标签，不判断框的定位是否准确。

助手抽查了 54 张：24 张正样本框、12 张漏检正样本、12 张背景误报和 6 张未检出背景。24 张正样本候选框均可用，抽查的背景误报包括喝水、摸脸等动作。抽查清单位于 `remaining_reviewed/review/sample.csv`，保留示例图及预测置信度。

## 第二轮：自训练

扩充集 `selftrain_round1` 共 1002 张：

- 150 张人工种子原样保留；
- 568 张预测类别与原始分类一致、置信度 >= 0.80、没有高置信度错类竞争框的正样本伪标签；
- 12 张助手逐图核对的误报背景；
- 272 张原始分类为背景、模型在 0.50 下未检出的图片。

合计 texting 400 张、calling 268 张、背景 334 张。伪标签不是人工真值；全部来源可在 `selftrain_round1/selection.csv` 追溯。漏检正样本和错类图片未加入训练，也未转成背景。

从第一轮权重继续训练 20 epoch。这里的 `val` 仍为 150 张人工种子，用于检查原有标注的拟合保持情况，和训练集重叠。

20 轮自训练已完成，新模型已保存到 `runs/selftrain_round1_yolo11n/weights/best.pt`。种子集拟合保持指标 mAP50=0.995、mAP50-95=0.734，并非独立验证指标。

剩余 2132 张 train 图片已用该权重重新预标注，结果保存到 `remaining_selftrained/xlabel`（X-AnyLabeling JSON）和 `remaining_selftrained/labels/train`（YOLO）。对比在 `selftrain_round1/comparison.csv`。

| 原始行为 | 图片数 | 人工种子模型有框图片 | 自训练模型有框图片 |
|---|---:|---:|---:|
| texting | 417 | 392 | 405 |
| calling | 423 | 356 | 393 |
| 背景（误报候选） | 1292 | 87 | 38 |

以上阈值均为 0.50。calling 中 392 张含对应类别，1 张仅出现错误类别。合计 836 张有框。其余 1296 张为空标签，其中 42 张原始类别为正样本，不能当成已确认背景。

仅比较未加入第二轮训练的图片：texting 67 张中含对应类别的图片数从 42 增到 55；calling 205 张中从 137 增到 174（有任意框从 138 增到 175）；背景 1008 张中有框数从 75 降到 35。该排除集合来自相同司机，只是图片层面的排除诊断，不能替代跨司机独立验证。

最终预标注的重点检查清单为 `remaining_selftrained/review_priority.csv`，共 81 张：42 张漏检正样本、38 张背景误报候选和 1 张错类。原始整图类别可能不能穷尽同时发生的动作，背景有框也应逐张核对，不能仅凭原始类别直接删掉正确目标。

最终样例再次视觉检查：6 张 texting 和 6 张 calling 预测框的位置与动作相符；6 张背景有框样例仍显示杯子、水瓶、摸脸等混淆，以及需人工判断的持机情形。预览在 `remaining_selftrained/review/preview.jpg`，全部抽查候选图在 `remaining_selftrained/review/sheets`。

原始 `dataset/images/val`、`dataset/labels/val` 不参与上述过程，前后用 SHA-256 校验。该独立验证集尚未完成真实检测框标注，因此最终跨司机检测精度仍未验证。

## 重现

已有输出目录会拒绝覆盖，请保留标注并为新一轮指定新输出目录或运行名。

```bash
cd /Users/arc928/Documents/ChatGPT/泰山派
./.venv/bin/python yolo_phone/train_seed.py
./.venv/bin/python yolo_phone/pseudo_label_remaining.py
./.venv/bin/python yolo_phone/review_predictions.py
./.venv/bin/python yolo_phone/build_selftrain_dataset.py
./.venv/bin/python yolo_phone/train_seed.py \
  --data yolo_phone/selftrain_round1/data.yaml \
  --model yolo_phone/runs/seed_reviewed_yolo11n/weights/best.pt \
  --name selftrain_round1_yolo11n --epochs 20
./.venv/bin/python yolo_phone/pseudo_label_remaining.py \
  --weights yolo_phone/runs/selftrain_round1_yolo11n/weights/best.pt \
  --output yolo_phone/remaining_selftrained
./.venv/bin/python yolo_phone/review_predictions.py \
  --output yolo_phone/remaining_selftrained
./.venv/bin/python yolo_phone/compare_rounds.py
```
