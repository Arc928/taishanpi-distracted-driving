# 方案二执行结果

**用户已经完成人工修正，并已重新训练。** 新流程记录见 [REVIEWED_RUN.md](REVIEWED_RUN.md)。下文为第一轮自动标注试验的历史记录，其模型使用了未经完整修正的自动框，不能作为可靠种子模型。

已完成以下命令行步骤：

1. 从 `dataset/manifest.csv` 中只抽取 `train` 的 50 张 `texting`、50 张 `calling`、50 张背景，生成 [seed/images/train](/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/seed/images/train)。`dataset/images/val` 和 `dataset/labels/val` 的 SHA-256 快照前后一致。
2. 使用 YOLO-World 的 `cell phone` 提示词自动预标注；漏检时使用 `person holding a cell phone` 作为较粗的候选框。100 张正样本中，79 张使用第一提示词，21 张尝试后备提示词，最终 2 张没有检测框。
3. 将 YOLO 标签转换为 X-AnyLabeling 原生 JSON，目录为 [seed/xlabel](/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/seed/xlabel)。应用已尝试打开该目录。这里的框是自动建议，必须逐张检查。
4. 用 YOLO11n 在 150 张种子图上训练 30 epoch，模型为 [best.pt](/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/runs/seed_yolo11n/weights/best.pt)。种子集上的拟合指标 mAP50=0.581、mAP50-95=0.402；这不是独立验证结果。
5. 用初版模型对剩余 2132 张 `train` 图片做了置信度 0.50 的预标注，结果在 [remaining/xlabel](/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/remaining/xlabel)，388 张有预测框、1744 张为空标签，报告在 [remaining/report.csv](/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/remaining/report.csv)。

## 你检查种子标签后

X-AnyLabeling 保存的是 JSON。请检查并修改新目录 `seed_corrected/xlabel`。修完后导出为独立的训练目录：

```bash
cd /Users/arc928/Documents/ChatGPT/泰山派
./.venv/bin/python yolo_phone/finalize_seed.py
```

然后重新训练：

```bash
./.venv/bin/python yolo_phone/train_seed.py
```

检查 `remaining/xlabel` 时，同样可以同步到 `remaining/labels/train`。建议先只保留你确认正确的高置信度框，再合并到正式训练集；`remaining` 的空标签包含“背景或模型漏检”两种情况，不能直接当作人工确认的背景。

第一版剩余图片统计：玩手机 417 张中 79 张有框，打电话 423 张中 233 张有框，背景 1292 张中 76 张有框。上述只是按原始整图行为标签统计的命中与候选误报，不能代表检测框精度。助手已逐图查看全部 150 张种子图，另存 100 个统一行为框及 50 张背景到 `seed_corrected`，供用户手动修正；未训练这个修正版。

## 已保存的可复现脚本

- `prepare_seed.py`：固定随机种子 42，抽取 50/50/50，拒绝覆盖已有 seed。
- `auto_label_seed.py`：YOLO-World 自动生成种子框提案。
- `train_seed.py`：MPS 上训练 YOLO11n 30 epoch。
- `pseudo_label_remaining.py`：使用初版模型生成剩余 train 图片的伪标签。
- `sync_xlabel_to_yolo.py`：将 X-AnyLabeling JSON 同步回 YOLO txt。
