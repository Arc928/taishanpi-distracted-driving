# 针对性补充第二轮训练（官方起点 80 轮）

用户 2026-10-21 审完针对性批次，13 张手机不明显的预填框按用户选择排除。
数据：`selftrain_multiview_targeted_round2_20261021/`。
- train 3449 张：原 2952 + 新增 300 困难背景、45 calling（侧面 AC8）、152 texting（正面 AC6）。
- val 212 张：原 200 + 12 张正面 AC6 texting。

配置与官方起点 70 轮实验相同，只改为 80 轮：
- 官方 yolo11n.pt 起点，640、batch 16、MPS、seed 0、workers 2；
- AdamW lr0 0.001667、momentum 0.9、warmup_bias_lr 0；
- mosaic 开启，最后 10 轮关闭（close_mosaic 10）；
- patience 0，不提前停止；每 5 轮保存检查点。

best.pt 按 Ultralytics fitness（0.1·mAP50 + 0.9·mAP50-95）在新 val 上选择。test 不用于选模。
旧权重在同一新 val 上的基线见 `baseline_new_val.json`（conf 0.001、iou 0.7）：
- e79：mAP50 0.894、mAP50-95 0.581；texting 0.824、calling 0.963。
- e150 last：mAP50 0.821、mAP50-95 0.517。

状态见 `status.json`，进度见 `training.log`，完成后自动写 `result.json`。不切换网站模型。

## 结果（22:41 完成 80 轮）

best.pt 为第 51 轮。`val_compare.py` 在新 val 上比较四个权重（@0.5 为定位命中）：

| 权重 | mAP50 | mAP50-95 | texting | calling | 背景误报 |
|---|---:|---:|---:|---:|---:|
| e79 | 0.894 | 0.581 | 43/59 | 45/54 | 8/99 |
| e150 last | 0.821 | 0.517 | 42/59 | 46/54 | 4/99 |
| 新 e51（best） | 0.913 | 0.573 | 49/59 | 42/54 | 1/99 |
| 新 e80（last） | 0.924 | 0.542 | 46/59 | 48/54 | 1/99 |

test 回归见 `../../test_reviewed_eval_targeted_round2_20261021/`，权重在跑 test 前已固定为 e51。

## 阈值（2026-10-22，只用 val 选）

`threshold_select.py`：在 0.20–0.60 间每 0.05 取一个阈值，计算“两类定位命中 − 背景误报”，取最大值，平分时取高阈值。
val 曲线平缓，选出 0.35（相比 0.5：calling +3、误报 +2）。选定后 test 只看一次，见 `threshold_select/test_at_chosen.json`。
