# 人工修正种子后再训练

请在 X-AnyLabeling 中打开：

```text
/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/seed_corrected/xlabel
```

这是完整的 150 张种子图片，均来自 train。助手已逐图查看并改成统一行为框，但这些仍是待你确认的标注。旧 `seed`、模型和 `remaining` 全部保留；独立 val 保持不变。

- `texting`：框持机的手与手机，不包含扶方向盘的另一只手。
- `calling`：框持机的手、手机以及附近的脸部，避免框整个人。
- 背景：保留空标注，检查有没有漏掉目标行为。
- 仅识别驾驶员行为，乘客玩手机不属于当前任务。
- 类别固定为 `texting`、`calling`；在这里保存 JSON 即可，暂时不用导出 txt。

修完后告诉助手“种子修完了”。之后执行：

```bash
cd /Users/arc928/Documents/ChatGPT/泰山派
./.venv/bin/python yolo_phone/finalize_seed.py
./.venv/bin/python yolo_phone/train_seed.py
./.venv/bin/python yolo_phone/pseudo_label_remaining.py
```

导出脚本检查矩形框、类别、坐标范围及缺失标签，并将修正后的 JSON 导出到独立的 `seed_reviewed`。它只验证文件结构，不能代替你确认框是否准确。对于看不清目标的图片，可以记录文件名并告诉助手剔除，避免将漏标正样本误当背景。

新模型默认保存为 `runs/seed_reviewed_yolo11n/weights/best.pt`，新的剩余图片预标注保存在 `remaining_reviewed`，默认阈值 0.50。后续比较玩手机、打电话的图片命中、错类和背景误报；不能把“有框数”或种子集的 mAP 当成独立检测精度。独立 val 尚缺真实框，最终检测精度仍需人工标注的验证集评估。

种子确认后再推进下一轮。剩余图片中无框的正样本标记为待补标，不直接作为背景；伪标签先抽查，再决定是否加入下一轮训练。
