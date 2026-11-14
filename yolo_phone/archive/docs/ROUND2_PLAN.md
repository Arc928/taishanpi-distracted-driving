# 第二轮补数据方案

数据集已重新下载，路径不变：`/Users/arc928/Downloads/state-farm-distracted-driver-detection`。
所有软链接可用；`dataset/images/val`、`dataset/labels/val` 共 1021 个文件的 SHA-256 与删除前完全一致，
`validation_check` 记录的哈希也一致。`manifest.csv` 2926 行全部可解析。

## 先排除的两个假设

**分辨率不是原因。** 用当前权重对 5 位验证司机重跑，imgsz 640/960/1280：

| 分辨率 | texting@0.5 | calling@0.5 | calling@0.25 |
|---|---:|---:|---:|
| 640 | 130/136 | **108/131** | 114/131 |
| 960 | 131/136 | 100/131 | 112/131 |
| 1280 | 116/136 | 42/131 | 80/131 |

640 最好，放大反而恶化。继续用 640 训练和推理。

**候选质量不是主因。** 检查第一轮筛选出的 `calling_auto` 147 张：
全部是“唯一一个正确类别、置信度 >= 0.80、没有 >= 0.5 的错类竞争框”，筛选逻辑本身没有问题。

## 打电话为什么弱

`remaining_selftrained/raw_predictions.json` 中，未进第一轮训练的 205 张打电话图片分桶：

| 桶 | 数量 | 含义 |
|---|---:|---|
| `calling_auto` | 147 | 高置信、干净，第一轮因配额没进训练 |
| `calling_mid` | 27 | 0.5-0.8，检出但不够自信 |
| `calling_near` | 8 | 0.3-0.5，接近阈值 |
| `calling_hard_miss` | 23 | < 0.3，模型盲区（p015 10 张、p050 4 张） |

玩手机对应：45 张高置信、10 张中置信、2 张边缘、10 张盲区。
背景误报候选：9 张 >= 0.8、26 张 0.5-0.8、23 张 0.25-0.5。

主要盲区集中在 `p015`、`p050`、`p042`、`p066`、`p072` 这几位司机，
和验证集里 p081 崩掉是同一类问题的两种表现：**模型对没见过的持机姿势和手机外观泛化不足**。
高置信伪标签只是重复模型已知的东西，补训练必须覆盖它不会的部分。

## 这一轮要标注什么

`round2_review/` 共 **140 张**，只从 train 抽取，已按补充价值分组：

| 分组 | 张数 | 标注要求 |
|---|---:|---|
| `calling_hard_miss` | 23 | 必标。模型完全没检出的打电话 |
| `calling_near` + `calling_mid` | 35 | 必标。有框但不自信，修正后可提高稳定性 |
| `texting_hard_miss` + `texting_mid` + `texting_near` | 22 | 必标。玩手机的同类盲区 |
| `bg_fp_high` + `bg_fp_mid` | 35 | 必标。容易被误判成目标的背景 |
| `bg_control` | 25 | 控制组。应保持空标签，用于确认模型没有乱报 |

标注方式与上一轮一致：只画 `texting` / `calling` 两个类别；背景保持空标签；
`images/train/` 下每张图已经带好模型预填的 JSON，打开 X-AnyLabeling 只需改错、删多余框、补漏框。

**只标 train 的图片。** p081 等 5 位验证司机继续留作评估，不进入任何训练。

## 另外 192 张可以直接用（省人工）

`round2_plan/candidate_pools.json` 里另有 147 张 calling + 45 张 texting，
符合和第一轮完全相同的干净标准（唯一正确框、置信度 >= 0.80、无错类竞争）。
这批可以在合并时直接加入并抽查若干张，不需要逐张重画。

## 标注完成后

```bash
cd /Users/arc928/Documents/ChatGPT/泰山派

# 1) 可选：把确认是背景的图片文件名写进 round2_review/backgrounds.txt（每行一个文件名）
#    未写且无框的图片会被自动排除，不会当背景误用

# 2) 合并 round1 + 本轮人工修正 + 可直接采用的高置信样本
./.venv/bin/python yolo_phone/build_round2_dataset.py

# 3) 从当前权重继续训练（640 分辨率）
./.venv/bin/python yolo_phone/train_seed.py \
  --data yolo_phone/selftrain_round2/data.yaml \
  --model yolo_phone/runs/selftrain_round1_yolo11n/weights/best.pt \
  --name selftrain_round2_yolo11n --epochs 20

# 4) 重新在 5 位验证司机上评估，重点盯 calling 和 p081
./.venv/bin/python yolo_phone/check_validation.py \
  --weights yolo_phone/runs/selftrain_round2_yolo11n/weights/best.pt \
  --output yolo_phone/validation_round2
```

目标不是把 calling 全局命中率再抬几个点，而是让 `p081`、`p015`、`p050` 这类
未见姿势上的漏检明显减少。评估以 5 位独立司机为准，train 上的数字不作为结论。
