# 分心驾驶检测：当前工作入口

YOLO11n 检测驾驶员用手机，两类固定为 0 `texting`、1 `calling`，训练与推理尺寸 640。
数据为 State Farm（车内侧视）+ 3MDAD（侧面/正面）混合，按司机划分，详细约定与历史见根目录 `AGENTS.md`。

## 当前状态（2026-11-02）

- **训练集**：`selftrain_multiview_targeted_round2_20261021/`（train 3449 / val 212）。
  - 在上一版 `selftrain_multiview_gemini_reviewed_20261010/`（train 2952 / val 200）基础上，
    加入针对性补充的困难背景、正面 AC6 texting 和侧面 AC8 calling。
- **网站权重**：针对性补充训练第 52 轮（旧记录写作“第 51 轮”，见 AGENTS 第 17 节），默认阈值 0.35（按 val 选定）。
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
  - 权重 `runs/multiview_targeted_round2_20261021_yolo11n/weights/best.pt`（第 52 轮，旧记录称 e51）。
  - test：mAP50 0.937；texting 85/98、calling 85/94、背景误报 5/200。
  - 与 e79 的对比见 `experiments/official_targeted_round2_20261021/` 和 `evals/test_reviewed_eval_targeted_round2_20261021/`。

## 数据集登记表

`config/datasets.yaml` 登记全部数据集（含 `archive/data/` 下的历史目录）的 id、路径、类型、状态
（active / protected / historical / archived）、上游来源和 manifest 哈希。脚本按 id 定位数据集，
`artifact(目录名)` 也只查登记表，未登记就报错，不再自动回退到归档目录。
- 新建数据集：在登记表加一条（含 parents 和 manifest_sha256），否则 `test_dataset_registry.py` 会失败。
- 移动目录：改对应的 `path:` 一行即可；data.yaml 中的绝对 `path:` 需同步更新。
- manifest 哈希与磁盘不符时单测失败，须先查明数据被谁改动。

## 原始数据位置

`config/sources.yaml` 记录 State Farm 与 3MDAD 原始数据的位置（`root`，可用环境变量 `STATEFARM_ROOT`、`MDAD_ROOT` 覆盖），
以及清单 `source_image` 列和软链接里写下的原前缀（`recorded`，历史记录，不改）。脚本读取原图时自动把 `recorded` 换算到当前位置。
原始数据换位置后：改 `root` 或设环境变量，然后预览并改写项目内的软链接：

```bash
./.venv/bin/python yolo_phone/relink_sources.py --source mdad          # 预览
./.venv/bin/python yolo_phone/relink_sources.py --source mdad --apply  # 执行，记录写入 archive/relink-*.json
```

新目标有任何一个不存在就整体拒绝。项目内部的链接已于 2026-11-14 全部改为相对链接（`--internal-relative`），项目文件夹整体移动不会失效。
data.yaml 的 `path:` 保持绝对路径（Ultralytics 会把相对路径接到它自己的数据目录下），由单测检查它与所在目录一致。

## 目录

现行数据集都在 `data/` 下（2026-11-14 移入），历史数据在 `archive/data/`；位置以 `config/datasets.yaml` 为准。

| 目录 | 用途 |
|---|---|
| `data/selftrain_multiview_targeted_round2_20261021/` | 最新训练集：上一版 + 针对性补充（`targeted_3mdad_round2_20261021/` 审核后导出） |
| `data/selftrain_multiview_gemini_reviewed_20261010/` | 上一版混合训练集（含导出确认记录，现行脚本仍引用） |
| `data/gemini38flash_3mdad_v1/` | 3MDAD 审核目录 `review_dataset/`（train/val/test JSON）与训练时快照 |
| `data/multiview_statefarm_3mdad_v1/` | 3MDAD 原始选样、司机划分与审计记录 |
| `data/test_review_3mdad_20261021/` | 测试集审核前基线、`exclusions.csv` 与审核说明 |
| `data/targeted_3mdad_round2_20261021/` | 针对性补充第 2 轮审核目录 |
| `evals/` | 人工框测试评估 `test_reviewed_eval_*/`，以及外部数据推理 `external_checks/`（不参与训练） |
| `runs/` | 各次训练的 best/last、results.csv、args.yaml（中间检查点已清理） |
| `experiments/` | 官方起点各实验的计划、日志与说明 |
| `data/dataset_23train_3eval_1of8/` | State Farm 1/8 抽样基础集与 3 司机独立 eval 快照 |
| `local_demo/` | 本机摄像头/图片/视频测试网页 |
| `archive/` | 历史数据（含 State Farm 阶段 seed/selftrain/review/remaining/validation）、旧报告与清理记录 |

## 当前配置（唯一来源）

`config/current.yaml` 记录当前训练集、网站权重及其 SHA-256、默认阈值、imgsz 与 NMS IoU。
网站服务、`train_seed.py` 和 `check_validation.py` 的默认值都从这里读取（`train.py` 只认训练计划）。
- 切换网站模型：改 `weights.path` 与 `weights.sha256`；阈值只能按 val 选定后改 `confidence`。改完重启服务并提交 git。
- 哈希不一致时服务拒绝启动；`server.py --weights <路径>` 可临时覆盖，不做哈希比对。
- 类别 id 顺序固定在 `core/labels.py` 的 `NAMES`，不在配置中。
- 各轮专用脚本（`prepare_targeted_round.py`、`evaluate_reviewed_holdout.py` 等）的默认值是那一轮的历史记录，不跟随当前配置。

## 代码结构

顶层 `*.py` 是命令行入口；可复用的逻辑放在 `core/`：

| 模块 | 内容 |
|---|---|
| `core/paths.py` | 项目根 `BASE`、`artifact()` 定位活跃或归档数据 |
| `core/io.py` | CSV/JSON 读写、SHA-256 |
| `core/labels.py` | 固定类别 id、3MDAD 动作映射、X-AnyLabeling→YOLO 转换与标签校验 |
| `core/splits.py` | 司机划分、选帧、跨划分泄漏检查、State Farm eval 快照 |
| `core/metrics.py` | 人工框评估的 IoU、逐图状态与汇总 |
| `core/config.py` | 读取并校验 `config/current.yaml` |
| `core/registry.py` | 按 id 查 `config/datasets.yaml` 中登记的数据集 |

训练统一用 `train.py --plan`：超参只写在计划文件里，结果（`resolved_plan.json` 含 git 提交、`status.json`、`result.json`）
写在计划旁边，最佳轮次按 Ultralytics 实际保存 best.pt 的规则（val mAP50-95 最大）记录。
`train_seed.py` 只留作快速继续训练；`experiments/` 里各实验自带的 `run.py`/`train.py` 是历史记录，续训类实验仍需专门脚本。

`continue_split.py`、`annotation_io.py` 仅为兼容 experiments 中历史脚本的旧导入，新代码直接从 `core` 导入。

## 常用命令

已有输出目录一律拒绝覆盖，重跑须换新名称。评估输出建议放在 `yolo_phone/evals/<新目录>`。

```bash
# 人工框评估：test 或 val；--labels-only 只检查标注、不跑模型
./.venv/bin/python yolo_phone/evaluate_reviewed_holdout.py --split test \
  --weights <标签>=<权重.pt> --confirmation '<审核完成说明>' --output yolo_phone/<新目录>

# 3MDAD 审核目录首次导出为训练集（v1 用法）
./.venv/bin/python yolo_phone/export_multiview.py \
  --dataset yolo_phone/gemini38flash_3mdad_v1/review_dataset \
  --output yolo_phone/<新训练目录> --confirmation '<复核完成说明>'

# 在已有训练集上合并一轮审核过的补充数据（第 2 轮起的用法）
./.venv/bin/python yolo_phone/export_targeted_round.py \
  --base yolo_phone/<基础训练集> --targeted yolo_phone/<补充审核目录> \
  --round <轮次标签，如 targeted_round3> --confirmation '<用户原话>' --output yolo_phone/<新训练目录>

# 训练：复制 experiments/plan_template.yaml 到 experiments/<新实验>/plan.yaml，改名称与参数
./.venv/bin/python yolo_phone/train.py --plan yolo_phone/experiments/<新实验>/plan.yaml --dry-run
nohup ./.venv/bin/python -u yolo_phone/train.py --plan yolo_phone/experiments/<新实验>/plan.yaml \
  > yolo_phone/experiments/<新实验>/training.log 2>&1 &

# 数据安全与配置单测（项目根目录运行）
./.venv/bin/python -m unittest discover -s yolo_phone -p 'test_*.py'

# 本机测试网页
./.venv/bin/python yolo_phone/local_demo/start.py
./.venv/bin/python yolo_phone/local_demo/stop.py
```

网址通常为 http://127.0.0.1:8765，以 `local_demo/runtime.json` 为准。
训练场景为车内摄像头，不代表电脑正面摄像头的准确率。
