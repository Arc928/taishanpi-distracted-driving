# 分心驾驶检测：当前工作入口

YOLO11n 检测驾驶员用手机，两类固定为 0 `texting`、1 `calling`，训练与推理尺寸 640。
数据为 State Farm（车内侧视）+ 3MDAD（侧面/正面）混合，按司机划分，详细约定与历史见根目录 `AGENTS.md`。

## 当前状态（2026-11-02）

- **训练集**：`selftrain_multiview_targeted_round2_20261021/`（train 3449 / val 212）。
  - 在上一版 `selftrain_multiview_gemini_reviewed_20261010/`（train 2952 / val 200）基础上，
    加入针对性补充的困难背景、正面 AC6 texting 和侧面 AC8 calling。
- **网站权重**：针对性补充训练第 51 轮，默认阈值 0.35（按 val 选定）。
  - 文件：`runs/multiview_targeted_round2_20261021_yolo11n/weights/best.pt`
  - SHA-256 前缀 `eb63ae94`
  - 之前用的是官方起点 100 轮第 79 轮（`6607c22b`），文件仍保留。
- **测试集**：3MDAD test 440 张已人工审核，评估在 `evals/test_reviewed_eval_20261021/`。
  - 第 79 轮：mAP50 0.919；阈值 0.5 下 texting 84/98、calling 89/94、背景误报 24/200。
  - 150 轮实验第 150 轮（`runs/multiview_official_resume150_20261012_yolo11n/weights/last.pt`）：
    命中相同，背景误报 10/200，但 val mAP50 较低。见 `evals/test_reviewed_eval_resume150_20261021/`。
  - 测试集不用于选模或调阈值。
- **主要短板**：正面视角左手打字（AC6）漏检；手靠近脸的动作（AC12/14/15/16）造成背景误报。
- **针对性补充训练（已完成，网站已切换）**：
  - 训练集 `selftrain_multiview_targeted_round2_20261021/`（train 3449 / val 212）。
  - 权重 `runs/multiview_targeted_round2_20261021_yolo11n/weights/best.pt`（第 51 轮）。
  - test：mAP50 0.937；texting 85/98、calling 85/94、背景误报 5/200。
  - 与 e79 的对比见 `experiments/official_targeted_round2_20261021/` 和 `evals/test_reviewed_eval_targeted_round2_20261021/`。

## 目录

| 目录 | 用途 |
|---|---|
| `selftrain_multiview_targeted_round2_20261021/` | 最新训练集：上一版 + 针对性补充（`targeted_3mdad_round2_20261021/` 审核后导出） |
| `selftrain_multiview_gemini_reviewed_20261010/` | 上一版混合训练集（含导出确认记录，现行脚本仍引用） |
| `gemini38flash_3mdad_v1/` | 3MDAD 审核目录 `review_dataset/`（train/val/test JSON）与训练时快照 |
| `multiview_statefarm_3mdad_v1/` | 3MDAD 原始选样、司机划分与审计记录 |
| `test_review_3mdad_20261021/` | 测试集审核前基线、`exclusions.csv` 与审核说明 |
| `evals/` | 人工框测试评估 `test_reviewed_eval_*/`，以及外部数据推理 `external_checks/`（不参与训练） |
| `runs/` | 各次训练的 best/last、results.csv、args.yaml（中间检查点已清理） |
| `experiments/` | 官方起点各实验的计划、日志与说明 |
| `dataset_23train_3eval_1of8/` | State Farm 1/8 抽样基础集与 3 司机独立 eval 快照 |
| `local_demo/` | 本机摄像头/图片/视频测试网页 |
| `archive/` | 历史数据（含 State Farm 阶段 seed/selftrain/review/remaining/validation）、旧报告与清理记录 |

## 当前配置（唯一来源）

`config/current.yaml` 记录当前训练集、网站权重及其 SHA-256、默认阈值、imgsz 与 NMS IoU。
网站服务、`train_seed.py` 和 `check_validation.py` 的默认值都从这里读取。
- 切换网站模型：改 `weights.path` 与 `weights.sha256`；阈值只能按 val 选定后改 `confidence`。改完重启服务并提交 git。
- 哈希不一致时服务拒绝启动；`server.py --weights <路径>` 可临时覆盖，不做哈希比对。
- 类别 id 顺序固定在 `annotation_io.NAMES`，不在配置中。
- 各轮专用脚本（`prepare_targeted_round.py`、`evaluate_reviewed_holdout.py` 等）的默认值是那一轮的历史记录，不跟随当前配置。

## 常用命令

已有输出目录一律拒绝覆盖，重跑须换新名称。评估输出建议放在 `yolo_phone/evals/<新目录>`。

```bash
# 人工框评估：test 或 val；--labels-only 只检查标注、不跑模型
./.venv/bin/python yolo_phone/evaluate_reviewed_holdout.py --split test \
  --weights <标签>=<权重.pt> --confirmation '<审核完成说明>' --output yolo_phone/<新目录>

# 3MDAD 审核导出为训练集
./.venv/bin/python yolo_phone/export_multiview.py \
  --dataset yolo_phone/gemini38flash_3mdad_v1/review_dataset \
  --output yolo_phone/<新训练目录> --confirmation '<复核完成说明>'

# 训练
./.venv/bin/python yolo_phone/train_seed.py --data <data.yaml> --model <起点.pt> \
  --name <新运行名> --epochs <轮数> --batch 16

# 数据安全单测（须在 yolo_phone/ 下运行）
cd yolo_phone && ../.venv/bin/python -m unittest test_multiview_dataset test_current_config; cd ..

# 本机测试网页
./.venv/bin/python yolo_phone/local_demo/start.py
./.venv/bin/python yolo_phone/local_demo/stop.py
```

网址通常为 http://127.0.0.1:8765，以 `local_demo/runtime.json` 为准。
训练场景为车内摄像头，不代表电脑正面摄像头的准确率。
