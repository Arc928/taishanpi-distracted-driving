# 3MDAD 测试集人工审核（2026-10-21 起）

目的：给 3MDAD test 440 张（10 位司机）补人工检测框，之后能算真实检测 mAP，
并把“手机根本看不见”的图和模型真正的漏检分开统计。

`baseline.json` 记录审核开始前全部 440 个 JSON 的哈希（都还是 pending、未修改）。
它只是起点快照，不接受任何框。由 `prepare_test_review.py` 生成。

## 在哪里标

直接在原审核目录里改，X-AnyLabeling 打开：

| 批次 | 张数 | 内容 |
|---|---:|---|
| `gemini38flash_3mdad_v1/review_dataset/review/test/test_01/` | 280 | 240 张手机动作（191 张有 Gemini/Grounding DINO 预填框）+ 40 张背景 |
| `gemini38flash_3mdad_v1/review_dataset/review/test/test_02/` | 160 | 全部背景候选，无预填 |

## 每张图怎么处理

- **能看见司机在用手机**：按项目标准画框，或修正预填框。texting 框持机手和手机；
  calling 框持机手、手机和附近脸部。实际行为与源类别不同就按实际改类别。
- **预填框**只是候选：位置不对、框到乘客或扶方向盘的手，都要修改或删除。确认无误可以直接保留。
- **源动作是手机、但手机完全看不见**（例如被方向盘或手挡住、在画面外）：保持无框。
  评估时它会单列为“不可见”，不算背景，也不计入命中率和 mAP。
- **背景图**：一般保持无框。如果司机其实在用手机，就画框，评估时会按正样本算。
- **实在判断不了**（太模糊、看不清是不是手机）：框不框都行，同时在 `exclusions.csv`
  加一行 `id,原因`，评估时会整张排除。

审核时不要参考模型预测或之前的失败拼图，这样测试标注才能独立于模型。

## 审完以后

告诉助手“测试集审完了”。助手会先跑只看标注、不跑模型的检查，把“不可见”图拼成一张图给你复核：

```bash
./.venv/bin/python yolo_phone/evaluate_reviewed_holdout.py --split test --labels-only \
  --output yolo_phone/test_review_3mdad_20261021/labels_check_<日期>
```

确认后再正式评估。权重已经事先固定为官方 100 轮第 79 轮和混合第 27 轮，不在 test 上调阈值、也不用 test 挑模型：

```bash
./.venv/bin/python yolo_phone/evaluate_reviewed_holdout.py --split test \
  --confirmation '<用户的完成说明>' --output yolo_phone/test_reviewed_eval_<日期>
```

输出包括：
- 全部、正面、侧面三组的检测 mAP50 / mAP50-95，按类别给出；
- @0.5 图像级的类别命中和定位命中（同类且 IoU≥0.5），以及背景误报；
- “不可见”图的单独统计；
- 每张图的审核决定和是否相对基线修改过。

已用审核过的 val 集验证过这套流程：官方 100 轮复算 mAP50 0.9108 / mAP50-95 0.6003，与训练记录一致。
State Farm 452 张独立 eval 仍然没有人工框，不在这次范围内。
