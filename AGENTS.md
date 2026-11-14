# 项目说明（AGENTS.md）

本文件记录“分心驾驶检测”项目的当前约定。当前入口见 `yolo_phone/README.md`，
历史过程位于 `yolo_phone/archive/docs/`（State Farm 阶段记录 `SPLIT23_ROUND2.md` 也在此）。
2026-11-02 起部分目录已移动，第 3–12 节中的旧路径按第 16 节映射查找。

## 1. 目标与标注

使用 State Farm Distracted Driver 数据训练 YOLO11n，后续部署方向为瑞芯微 RKNN（资料在 `work/`）。
原数据只有整图分类，没有检测框；必须区分分类标签、人工框和模型伪标签。

| 原始类别 | 行为 | 固定目标 id |
|---|---|---|
| c1 / c3 | 左右手玩手机 | 0 `texting` |
| c2 / c4 | 左右手打电话 | 1 `calling` |
| c0 / c5-c9 | 其他行为 | 背景，空标签 |

人工确认的实际行为可以纠正弱整图类别，但不得改变目标 id 顺序。
`p014_c2_img_86001` 已由用户纠正为 texting，当前训练标签为 0，并保留源 c2 记录。

标注范围：texting 框持机手和手机，不含扶方向盘的手；calling 框持机手、手机及附近脸部，
不框整个人。只识别驾驶员，乘客拿手机不属于任务。

## 2. 环境与固定数据源

直接使用现有 `./.venv/bin/python`，无需重建环境。
Python 3.12.14、PyTorch 2.14.1（MPS 可用，Apple M1 Pro）、Ultralytics 8.4.172。
版本锁定在 `yolo_phone/requirements-lock.txt`。
X-AnyLabeling 4.1.0 位于 `/Applications/X-AnyLabeling.app`。

固定源目录（不要移动或删除）：

```text
/Users/arc928/Downloads/state-farm-distracted-driver-detection
  driver_imgs_list.csv
  imgs/train/c0..c9/     # 22424 张、26 位司机
  imgs/test/            # 无标签，本项目不用
```

## 3. 当前目录（2026-11-14 数据集移入 data/ 后）

```text
yolo_phone/
  config/                         # current.yaml 当前配置；datasets.yaml 数据集登记表
  core/                           # 公用逻辑包；顶层 *.py 为命令行入口
  data/
    selftrain_multiview_targeted_round2_20261021/ # 最新训练集 train 3449 / val 212
    selftrain_multiview_gemini_reviewed_20261010/ # 上一版混合训练集，现行脚本仍引用
    targeted_3mdad_round2_20261021/ # 针对性补充的审核目录
    gemini38flash_3mdad_v1/         # 3MDAD 审核目录 review_dataset/（train/val/test）
    multiview_statefarm_3mdad_v1/   # 3MDAD 选样、司机划分与审计
    test_review_3mdad_20261021/     # 测试集审核前基线与 exclusions.csv
    dataset_23train_3eval_1of8/     # State Farm 基础抽样集与 3 司机独立 eval
  evals/                          # 人工框测试评估 test_reviewed_eval_*、external_checks
  experiments/                    # 各训练实验的计划、日志、val 对照与阈值选择；plan_template.yaml
  runs/                           # 各次训练的 best/last、results.csv、args.yaml
  local_demo/                     # 本机摄像头、图片、视频测试网页
  archive/                        # 历史数据、阶段文档、旧报告及清理记录
```

本文其他各节提到的数据集目录名都不变，现位于 `data/` 下；记录中的旧绝对路径按 `archive/cleanup-2026-11-14-move.json` 映射。

下文第 3–12 节的权重路径与指标为当时记录；最新权重见第 15 节。
State Farm 阶段的 seed/selftrain/review/remaining/validation 目录已在 `archive/data/`。
还保留新种子与第一轮的权重。更早旧五司机方案的模型权重已删除。
第三轮全量方案（12251 张）及其脚本已停止、删除；不再使用该方案。

2026-10-09 用户确认整理：18 个历史数据目录移至 `archive/data/`，10 个旧脚本移至
`archive/scripts/`，7 份过程记录移至 `archive/docs/`，旧杂项日志在 `archive/scratch/`。
自动预览图和停用的 YOLO-World/CLIP 权重移入废纸篓，项目内减少约 602 MiB；尚未清空废纸篓。
路径映射、废纸篓位置与核验结果见 `archive/cleanup-2026-10-09.json`。
旧 seed 的真实图片仍在 `archive/data/seed/images/train/`，依赖它的软链接已修复。
人工 JSON、标签、预测 JSON、统计和保留模型均按哈希核验为未改变。
历史配置中的路径已适配归档位置，原配置及两份当前 manifest 原版在 `archive/config-originals/`。

当前脚本（2026-11-14 整理后，见第 17 节）：
- 顶层：`train.py`、`evaluate_reviewed_holdout.py`、`check_validation.py`、`prepare_targeted_round.py`、
  `export_targeted_round.py`、`relink_sources.py`。
- `builders/`：已执行完毕的一次性构建脚本 `prepare_dataset.py`、`prepare_multiview.py`、`export_multiview.py`、
  `prepare_test_review.py`。
- `tests/`：全部单测，在项目根运行 `./.venv/bin/python -m unittest discover -s yolo_phone/tests -t yolo_phone`。
- 公用逻辑在 `core/`：`paths`、`io`、`labels`、`splits`、`metrics`、`config`、`registry`、`sources`。
- `continue_split.py`、`annotation_io.py`、`train_seed.py` 已删除，可从 git 历史找回。
归档旧脚本已于 2026-11-02 删除（移入废纸篓，见第 16 节）。

## 4. 数据与验证约定

**独立 eval 司机固定为 p016、p021、p026，其余 23 位为 train。**
用户已确认该划分，保持 1/8 抽样、seed=42、每位司机每个原始类别取 `ceil(数量/8)`。
train 2474 张（源 texting 507 / calling 512 / 背景 1455），eval 452 张（96 / 92 / 264）。
基础集正样本缺少检测框，不能直接拿其 `data.yaml` 训练；训练使用派生的已标注数据集。

**eval 司机任何图片都不能进入训练，一张都不行，包括背景。**
只从 `dataset_23train_3eval_1of8/manifest.csv` 的 `split == 'train'` 选样本，必须按司机划分。
新 eval 快照：`dataset_23train_3eval_1of8/eval_files_sha256.json`（716 个现有文件）。
旧五司机数据已移至 `archive/data/dataset/`，其图像和标签保持内容不变；
快照为 `archive/data/seed/val_snapshot.json` 和
`archive/data/validation_check/val_files_sha256.json`（1021 个文件）。
修改验证文件前先核对哈希。
旧划分的 p039、p081 已获准在新划分进入训练，旧五司机汇总不再是新模型的独立评估。

**漏检正样本不能当背景。** 源 c1-c4 无框且未确认有效检测标注的图片必须排除。
**伪标签门槛**：唯一正确类框、置信度 >=0.8（严格场景 0.85），无 >=0.5 的错类竞争框。
当前筛选对 >=0.5 的同类多框也排除。低置信和漏检送人工，不降低门槛刷数量。
**训练与推理尺寸固定 640。**

## 5. 当前训练与结果

原 150 张人工种子及第二轮补标已复用为 250 张种子，训练 30 轮。
第一轮新划分自训练集 1422 张，训练 20 轮。
用户随后明确回复“修完了”，最近 140 张复核中采用 127 张：
27 张修改正样本、46 张接受预填、54 张背景；13 张仍无框的源正样本排除。
15 张旧伪标签被人工结果覆盖，新增 112 张，当前训练集 1534 张：
实际标签 texting 507 / calling 499 / 背景 528，23 位 train 司机均覆盖。
用户原始 JSON 保持原样；导出快照和确认记录在当前集 `reviewed_json/` 与 `human_review_receipt.json`。

第二轮配置最多 20 轮、batch 16、MPS；patience=10 在第 11 轮触发提前停止。
旧种子拟合指标最佳为第 1 轮，依既有规则使用其 best.pt，未根据独立 eval 挑模型。
训练配置里的 val 是 250 张种子，与 train 重叠，mAP 仅表示拟合保持。

同三位独立司机、阈值 0.5 的图像级结果：

| 指标 | 新划分第一轮 | 人工补标后第二轮 |
|---|---:|---:|
| texting 命中 | 96/96 | 93/96 |
| calling 命中 | 87/92 | 92/92 |
| 背景误报 | 2/264 | 2/264 |

texting 新增漏检均为 p016 c3；1 张 p026 calling 图同时出现 texting 错类框。
calling 全数命中不等于每个框都正确。真实检测 mAP 需要独立 eval 的人工检测框。
训练司机诊断：p039 calling 6/17→13/17、p081 calling 17/22→21/22；
p081 背景误报 1/62→4/62。它们与训练重叠，不能作为独立泛化结论。

历史结论仍有效：
- 旧 p081 全程未参与训练，其问题是未见司机的姿态/外观组合，绿色手机不是主因。
  当时训练池已有 48 张绿色手机样本，其中 47 张进入训练。
- 分辨率对照 calling 命中 640/960/1280 为 108/100/42，增大分辨率没有帮助。
- 高置信伪标签自训练有天花板，补独立司机与困难姿态比堆同司机容易样本更重要。
- 旧五司机 @0.5 第一轮/第二轮：texting 130/136→132/136、calling 108/131→106/131、
  背景误报 11/377→7/377；这些是历史图像级统计。

## 6. 常用命令

已有目录拒绝覆盖，重新运行须指定新名称；直接使用当前环境。

```bash
# 当前划分的抽样方式；现有目录已存在，重跑须换输出名
./.venv/bin/python yolo_phone/builders/prepare_dataset.py --sample-every 8 --seed 42 \
  --val-subjects p016 p021 p026 --output yolo_phone/dataset_23train_3eval_1of8

# 训练（2026-11-14 起）：超参只写在 experiments/<实验>/plan.yaml，模板 experiments/plan_template.yaml
./.venv/bin/python yolo_phone/train.py --plan yolo_phone/experiments/<实验>/plan.yaml --dry-run

# 当前三司机独立评估；脚本默认就是新划分
./.venv/bin/python yolo_phone/check_validation.py \
  --weights <权重.pt> --output yolo_phone/validation_<新名字>

# 本机测试启动 / 停止
./.venv/bin/python yolo_phone/local_demo/start.py
./.venv/bin/python yolo_phone/local_demo/stop.py
```

本机网址通常为 http://127.0.0.1:8765，实际运行状态见 `local_demo/runtime.json`。
预览与推理独立刷新，实时上传最长边 640、模型尺寸 640，默认阈值 0.5。
30 FPS 测试视频实测预览 30、检测约 29.5 FPS，非真实摄像头准确率验证。
用户实际使用电脑前置/USB 摄像头正面拍摄；这与车内侧视训练场景不同，当前模型泛化差。
改善该场景需要新增实际场景的人工标注训练与独立评估数据，方案尚未授权构建。

## 7. 后续事项

2026-10-10 已按用户请求测试外部 `Multi-Class Driver Behavior Image Dataset/texting_phone/` 的全部 1561 张。
输出为 `yolo_phone/external_checks/multiclass_texting_20261010/`，脚本 `test_external_texting.py`，
使用最新 best.pt、640、阈值 0.5；只做推理，源文件未改，未加入训练。
1505 张与原 State Farm 文件 SHA-256 一致，其中 178 张与实际训练图重合。
另外 56 张经全部 22424 个原始文件的大小及哈希核验，无精确重复。
@0.5：总体 texting 命中 1460/1561，State Farm 部分 1458/1505，其他场景仅 2/56；
@0.25 其他场景也仅 3/56，说明跨场景表现很弱。总体分数不能当独立泛化成绩。
助手查看两张其他场景命中，其中一张框含扶方向盘的手；有框不代表定位正确。
这些仍是整图类别诊断，不是检测 mAP，且只测 texting，不能据此估计背景误报率。

当前清理已完成，测试服务保持运行。下一轮训练方案尚未指定。
可从现有 train 优先补与 texting 漏检、背景混淆相似的困难姿态；eval 不能进入训练。
若改为针对正面摄像头采集新数据，先明确用户授权与新数据的训练/评估划分。

## 8. 2026-10-10 多视角数据重建（最新）

用户已授权下载 3MDAD 并按项目重建，明确选择“混合 State Farm＋3MDAD”。
原始新数据固定在 `/Users/arc928/Downloads/3MDAD/`：白天侧面/正面 RGB 各 111017 张，
共 50 位参与者，800 个头手 .mat 标注；CRC 全量通过，逐文件 SHA-256 在原始目录。
头手框不等于本项目行为检测框，不能直接改成 texting/calling 标签。

新目录 `yolo_phone/multiview_statefarm_3mdad_v1/` 已完成：
- 原有效 State Farm 1534 张保留，实际标签 507 texting / 499 calling / 528 背景；
  标签副本哈希、来源及 p014 人工纠正均保留，不把旧伪标签升级为人工真值。
- 3MDAD 按 seed=42 固定 35 train / 5 val / 10 test，具体名单见 `subject_splits.json`。
  此名单约束全部 222034 张源图；同一司机所有帧、所有视角同组。
- 每司机每视角 AC5/6/7/8 各取 3 个时间分位中点帧，AC1/2/3/10/11/12/13/14/15/16 各取 1 帧。
  AC4 GPS 与 AC9 拍照暂不采样；共 2200 张，train 1540 / val 220 / test 440。
- train 1540 张已用最新 best.pt、640 生成候选框（仅 26 张有 >=0.5 的预填框）；
  全部仍待人工审核，没有自动接纳伪标签。val/test 没有模型预填。
- 原 State Farm 452 张独立 eval 单独保留，原 716 文件快照核对通过；不用于训练。
- 新旧选样的司机与 SHA-256 跨划分泄漏检查通过，见 `audit.json`。

这是待标注候选集，**没有可直接训练的 data.yaml，不能把未审核/无框图片当背景训练**。
第一批 `review/train/train_01/` 280 张，覆盖 35 位司机、两个视角、四个手机动作。
审核后需在 `review_status.csv` 明确 positive/background/exclude，默认 pending。
源正样本无框只可排除；源背景允许人工纠正为正样本。JSON 修改本身不代表复核完成。

新增脚本 `prepare_multiview.py`、`export_multiview.py`，现有输出拒绝覆盖。
导出要求明确复核完成记录、完整已审核 val 和两类+背景的新 train；未审核样本不导出。
test 完整审核前不写入训练配置；test 不用于选模或调阈值。验证用 3MDAD 独立 val，
不能再把旧 seed 拟合检查当独立验证。9 个数据安全单测及实际拒绝未审核导出的检查已通过。
本次未训练、未更换本机服务模型；当前可训练集及最新权重仍是第二轮 State Farm 版本。
3MDAD 许可为 CC BY-NC-ND 4.0，教育/研究、非商业用途。

## 9. 2026-10-10 混合训练（已完成）

用户指定 `gemini38flash_3mdad_v1/review_dataset` 与 State Farm 重新训练，随后说明
“我应该是刚修正完，你可以看下修正的时间确认一下”。已核对六个训练批次均有内容修改，
训练最后保存 20:03:34；val 连续保存 20:03:44–20:07:56，与用户完成说明一致。
原始 JSON/CSV 未修改；在 `gemini38flash_3mdad_v1/training_review_snapshot_20261010/`
记录 train/val 的复核决策，test 仍 pending，不把时间戳本身等同于人工真值。

已导出 `selftrain_multiview_gemini_reviewed_20261010/`：train 2952 张
（原 State Farm 1534 + 3MDAD 1418），val 为独立 5 位 3MDAD 司机的 200 张。
无框源正样本排除 train 122、val 20，不能视为背景。val 仅覆盖原 120 张源正样本中的 100 张，
指标报告必须说明筛选范围；3MDAD test 不用于本次训练选模。State Farm 独立 eval 保持不变。
用户说明、哈希、时间与决策见导出目录的 `human_review_receipt.json`、`review_completion_evidence.json`。

运行 `runs/multiview_gemini_reviewed_20261010_yolo11n/` 已启动：从原第二轮 best.pt 继续，
YOLO11n、640、batch 16、MPS、最多 30 轮、patience 10。实际参数见 args.yaml。
Ultralytics 扫描 train 2952 / val 200，0 损坏；当前状态见导出目录 `training_status.json`、
`training.log` 与运行目录 `results.csv`。未更换本机测试服务模型；原最新权重仍保留。

2026-10-10 约 21:19 已正常完成全部 30 轮（约 65.4 分钟），进程已退出，best.pt 来自第 27 轮。
新权重：`yolo_phone/runs/multiview_gemini_reviewed_20261010_yolo11n/weights/best.pt`。
最终复验 200 张 val：P 0.936 / R 0.789 / mAP50 0.914 / mAP50-95 0.580；
texting mAP50 0.883、calling 0.944。只代表筛选后的验证集，未包含被排除的 20 张源正样本。
尚未进行 3MDAD 独立 test 或新模型 State Farm 独立 eval；本机服务模型未切换。
后续最新训练权重以上述混合模型为准，第 3/5/6 节第二轮路径与指标为历史基线。

## 10. 2026-10-10 混合模型留出图片测试

用户要求使用测试集，已固定第 27 轮 best.pt，640、conf=0.5，完成全部 3MDAD test 440 张
（10 位司机）及 State Farm 独立 eval 452 张。输出 `yolo_phone/test_multiview_20261010/`，
脚本 `evaluate_multiview_holdout.py`。未排除 49 张无框测试正样本；没有测试调参或选模。

3MDAD 源整图类别：texting 82/120，calling 90/120，背景误报 14/200。
正面 texting 29/60、calling 48/60、背景误报 9/100；
侧面 texting 53/60、calling 42/60、背景误报 5/100。背景 14 张误报中 13 张为 calling。
State Farm：texting 96/96、calling 90/92、背景误报 0/264。
这些是源动作整图诊断，不是检测 mAP；test 审核状态仍 pending，没有自动接纳未确认检测框。
源图/标注/模型未改，司机与像素哈希隔离通过。本机服务模型仍未切换。

2026-10-10 用户要求切换最新模型，本机服务已改用混合模型第 27 轮 best.pt，
默认权重同步更新于 local_demo/server.py，重启后权重 SHA-256 与测试报告一致。
/health 和 /api/predict 冒烟验证通过；实际地址见 local_demo/runtime.json。

## 11. 2026-10-12 官方起点 100 轮权重测试

用户发来 100 轮曲线并要求测试，定位为 `runs/multiview_official_resume100_20261011_yolo11n/`。
该实验从官方 yolo11n.pt 起点训练，后从已完成 66 轮的检查点续训到总计 100；
完整过程见 `experiments/official_resume100_20261011/README.md`。best.pt 为第 79 轮，
SHA-256 `6607c22b48553ddc3ecf432905f87ef87a1b7b13c08bd9c6c6d4914fc3b37c76`。
验证 mAP50 0.91080、mAP50-95 0.60045。类别和加载正常；原训练清单与全部图像/标签哈希一致。

本次用 best.pt，640、conf 0.5，固定测试相同 892 张图片，结果在 `test_official_resume100_20261012/`。
3MDAD texting 86/120、calling 105/120、背景误报 24/200；
正面 31/60、55/60、11/100；侧面 55/60、50/60、13/100。侧面 calling 有 1 张只出现 texting 错类框。
State Farm texting 96/96、calling 91/92、背景误报 2/264。
旧混合第 27 轮对应 3MDAD 82/120、90/120、14/200；State Farm 96/96、90/92、0/264。
本次打电话命中提升但背景误报增多，正面 texting 改善有限。
均为源整图类别诊断，不是检测 mAP；49 张无框测试正样本仍计入。
测试集已在前期开发中查看，现应视为固定留出集回归对照，而非完全未见的最终评估。
本次没有调阈值或依据 test 挑其他权重，也没有切换本机服务。

2026-10-12 用户明确要求切换网站权重，本机网站已切换为官方起点 100 轮实验的第 79 轮 best.pt。
默认权重同步更新 local_demo/server.py，重启后的权重哈希与本次测试报告一致；
首页、/health 和 /api/predict 均验证通过。网址以 local_demo/runtime.json 为准。

## 12. 2026-10-21 测试集人工审核与人工框评估

失败分析：官方 100 轮模型的 3MDAD 测试漏检集中在正面 AC6（27/29 张正面 texting 漏检），
以及侧面 AC8。训练集中正面 AC6 仅 48 张，侧面 AC8 仅 50 张，其他组合约 100 张；
val 正面 AC6 仅 2 张。部分漏检图中手机完全不可见，现有整图指标把它们计为漏检。
用户选择先补测试集人工框：
- `test_review_3mdad_20261021/` 存审核前基线（440 个 JSON 哈希，均 pending 未修改）、
  `exclusions.csv` 和审核说明；生成脚本为 `prepare_test_review.py`。
- 用户在原 `gemini38flash_3mdad_v1/review_dataset/review/test/` 中审核。
- 新脚本 `evaluate_reviewed_holdout.py` 以人工框为真值，计算全部、正面、侧面三组 mAP，
  以及 @0.5 图像级的类别命中和定位命中（IoU≥0.5）。
- 源手机动作无框的图记为 not_visible 单列，不作背景；`--labels-only` 不跑模型。
- 权重事先固定为官方 100 轮 e79 和混合 e27。
- 已用已审核 val 复现训练 mAP（0.9108/0.6003），新增 2 个单测，共 11 个通过。
用户确认 440 张均已逐张看过；有 6 个 JSON 相对基线修改。
助手指出 6 张 AC8 手机疑似可见，用户决定不补框（“手机不明显”）。
审核结果：可见正样本 192（texting 98 / calling 94）、背景 200、不可见 48。
结果在 `test_reviewed_eval_20261021/`。官方 100 轮 e79 的指标如下：
- 检测 mAP50 0.919、mAP50-95 0.536；正面 0.871、侧面 0.956。
- 正面 texting AP50 0.790，侧面 0.955。
- @0.5 图像级：texting 84/98、calling 89/94、背景误报 24/200。
- 正面 AC6 可见 9 张只命中 1 张，另有 21 张不可见。
混合 e27：mAP50 0.910、mAP50-95 0.545；@0.5 texting 81/98、calling 81/94、背景误报 14/200。
类别命中基本都定位正确。主要问题是正面左手打字的漏检，以及手靠近脸的动作（AC12/14/15/16）误报。

2026-10-21 补测 150 轮实验 `runs/multiview_official_resume150_20261012_yolo11n/`。
该实验从 100 轮实验 e96 恢复原优化器状态，patience=0 跑到 150；val 未超过 e79，
其 best.pt 与 100 轮 e79 张量完全相同。第 150 轮 last.pt 的结果见 `test_reviewed_eval_resume150_20261021/`：
- 人工框测试：mAP50 0.914、mAP50-95 0.540；texting 84/98、calling 89/94，与 e79 相同；
  背景误报 10/200（e79 为 24/200）。
- val：mAP50 0.842（e79 0.911）；背景误报 4/99（e79 8/99）；texting 38/47、calling 46/54。
网站模型未切换；是否切换须依据 val，不依据 test。

## 13. 2026-10-21 项目清理

用户全选清理方案，详见 `yolo_phone/archive/cleanup-2026-10-21.json` 与 `archive/README.md`。
- 移入专用废纸篓文件夹（尚未清空）：
  - 32 个中间检查点；
  - 低学习率 150 run；
  - `open_vocab_3mdad_v1`；
  - `__pycache__`、`.DS_Store`。
- 移入 archive：
  - `experiments/official_low_lr150_20261012` → `archive/experiments/`；
  - 第 10、11 节的旧报告及官方起点测试报告 → `archive/reports/`。
- 项目从约 1.3G 降到约 486M（不含 .venv）。
- 保留：所有 run 的 best/last；官方 yolo11n.pt；全部人工审核数据与评估结果。
  网站权重哈希核验不变。
- 首轮脚本误把低学习率 run 嵌套移入同名废纸篓目录，已在废纸篓内还原层级。
- `yolo_phone/README.md` 已重写为当前状态。
- 清理时发现本机网站进程（runtime.json 记录的 pid）已不在运行，与清理无关，未自动重启。

## 14. 2026-10-21 针对性补充候选（待人工审核）

用户同意按“补数据优先”继续优化。助手更正了此前的误判：官方起点 70 轮实验前 60 轮已开 mosaic，
val 从第 20 轮起停在 0.90–0.92。用户选择暂不做新的训练配置对照，等补充数据审完后再训练。

`yolo_phone/targeted_3mdad_round2_20261021/` 由 `prepare_targeted_round.py` 生成，共 725 张待审核：
- train 正面 AC6 280 张（35 司机 × 8 帧）、侧面 AC8 105 张；
- val 正面 AC6 40 张；
- train 困难背景 300 张：e79 在 94035 张 train 背景动作帧中有 5972 张误报 ≥0.5，
  按“司机 × 视角 × 动作”分散抽取。
- 只有 train 正样本有模型候选框（conf≥0.25），没有自动接受；test 司机未读取，隔离检查通过。
- 审核规则同测试集：手机不可见或不明显则不画框，此类源正样本导出时排除。
  决定由框推导，需要用户的完成确认；新增 1 个单测后共 12 个通过。

## 15. 2026-10-21 针对性补充导出与训练（已完成）

用户说“审完了”。核对结果：正样本批次修改 151 个 JSON；hardneg 300 张未改且全空。
助手标出 13 张直接接受预填、但手机不明显的框，用户回复“2”，全部写入
`targeted_3mdad_round2_20261021/exclusions.csv` 排除。
由 `export_targeted_round.py` 导出 `selftrain_multiview_targeted_round2_20261021/`：
- train 3449 张：原 2952 + 新增 300 困难背景、45 calling（侧面 AC8）、152 texting（正面 AC6）。
- val 212 张：原 200 + 12 张正面 AC6 texting。无框源正样本排除，不作背景。
- 司机及 SHA-256 隔离检查（含 3MDAD test）通过；确认原话写入 `human_review_receipt.json`。

旧权重在新 val 上（conf 0.001、iou 0.7）：e79 mAP50 0.894 / mAP50-95 0.581，
e150 last 0.821 / 0.517。新 val 与旧 val 指标不可直接比较。
训练 `runs/multiview_targeted_round2_20261021_yolo11n/`，配置同官方起点 70 轮实验，改为 80 轮，
2026-10-21 19:15 启动，预计 3–3.5 小时。计划和状态见
`experiments/official_targeted_round2_20261021/`。按新 val 选模，test 只做最终回归对照；网站模型未切换。

22:41 跑满 80 轮。best.pt 为第 51 轮（fitness 最高），SHA-256 `eb63ae94…`；跑 test 前已固定。
新 val @0.5 定位命中（`experiments/official_targeted_round2_20261021/val_compare/`，脚本 `val_compare.py`）：
- e79：mAP50 0.894 / mAP50-95 0.581；texting 43/59、calling 45/54、背景误报 8/99。
- 新 e51：0.913 / 0.573；texting 49/59、calling 42/54、背景误报 1/99；正面 AC6 14 张中命中 11 张（e79 为 7 张）。
- 新 e80 last：0.924 / 0.542；texting 46/59、calling 48/54、背景误报 1/99。
test 回归 `test_reviewed_eval_targeted_round2_20261021/`（e79 对 e51）：
- mAP50 0.919→0.937、mAP50-95 0.536→0.569；正面 0.871→0.931，侧面 0.956→0.931。
- @0.5：texting 84→85/98，calling 89→85/94，背景误报 24→5/200，不可见出框 19→11/48。
- 正面 AC6 可见 9 张命中 1→3。calling 少的主要是正面 AC7，其 AP50 反而升高（0.952→0.981），
  多为置信度降到 0.5 以下。阈值只能依据 val 调，不能依据 test 调。网站模型尚未切换。

2026-10-22 用户选择“先做 2”：在 val 上为 e51 定阈值。规则事先写在
`experiments/official_targeted_round2_20261021/threshold_select.py`：在 0.20–0.60 间每 0.05 取一个阈值，
计算“两类定位命中 − 背景误报”，取最大值，平分时取高阈值。结果在 `threshold_select/`。
- val 曲线很平（0.20–0.50 得分 90–91），选出 0.35：texting 49、calling 45、误报 3/99（0.5 为 49/42/1）。
- 选定后 test 只算一次（用已保存的 conf≥0.1 原始预测）：@0.35 texting 85/98、calling 88/94（定位）、
  背景误报 9/200、不可见出框 13/48；@0.5 为 85、84、5、11。e79@0.5 为 82、89、24、19。
当时网站模型和默认阈值均未改。

2026-10-22 用户要求“切换”。核对发现 00:41 已有改动（非本会话所为，来源未知）：
`local_demo/server.py` 默认权重改为新 e51 best.pt，默认阈值 0.35；`index.html` 滑块默认 0.35，
服务已用新权重重启（pid 以 runtime.json 为准）。本会话核验：
- 权重 SHA-256 `eb63ae94…` 与评估一致；
- /health 返回 default_confidence 0.35，首页滑块为 0.35；
- 用 val 图片调用 /api/predict 正常返回检测框。

## 16. 2026-11-02 项目结构整理

用户全选四项，记录与逐文件 SHA-256 见 `yolo_phone/archive/cleanup-2026-11-02.json`。
- 移入 `archive/data/`：`seed_23train_reviewed`、`selftrain_23train_round2`、`review_23train_round1`、
  `remaining_23train_round2`、`validation_23train_round2`；`SPLIT23_ROUND2.md` 移入 `archive/docs/`。
- 移入 `archive/scripts/` 的 6 个旧脚本：`prepare_split_review`、`merge_split_review`、
  `compare_split_validation`、`pseudo_label_remaining`、`evaluate_multiview_holdout`、`test_external_texting`。
  `continue_split.py` 被现行脚本导入，未归档。
- 移入 `evals/`：3 个 `test_reviewed_eval_*` 和 `external_checks/`。
- `__pycache__`、`.DS_Store` 移入废纸篓 `taishan-cleanup-2026-11-02-e8ed97e4`（尚未清空）。
- 改了 27 个文件里的路径，原版在 `archive/config-originals/2026-11-02/`：
  - 被移动目录及两个旧 selftrain 归档里的 data.yaml；
  - 评估目录 `yolo/` 下的 yaml/txt；
  - `prepare_multiview.py`、`train_seed.py` 改用 `artifact()`；
  - `threshold_select.py` 和实验 README。
- 未改：训练集 manifest、审核回执、评估 JSON 内的绝对路径，它们是历史记录，按清理 JSON 映射定位。
- 核验：16414 个移动文件内容一致，软链接全部可解析，State Farm 716 文件 eval 快照通过；
  12 个单测通过（须在 `yolo_phone/` 下运行），网站权重哈希不变。整理时本机网站未在运行。

2026-11-02 用户随后要求“用不到的脚本啥的帮我删掉”，选择两项，移入废纸篓
`taishan-delete-2026-11-02-57b5cd98`（尚未清空，可恢复），记录见 `archive/cleanup-2026-11-02-delete.json`：
- 第 15 节针对性训练 run 的 16 个中间检查点 epoch0–75.pt（324 MiB），删后不能从中间轮次续训；
  best.pt（e51，`eb63ae94…`）与 last.pt 哈希核验不变。
- `archive/scripts/` 全部 16 个旧脚本及空目录 `archive/scratch/`。
保留：顶层全部现行脚本（含 `check_validation.py`、`prepare_dataset.py`、`prepare_test_review.py`）、
experiments 中的 run/train 记录脚本。项目（不含 .venv）约 544M。

## 17. 2026-11-14 Git 基线与统一当前配置

用户要求先做 Git 基线，再“接着做”统一配置。
- 初始提交 `6213ffdf`（master）：跟踪代码、文档、人工审核 JSON、标签、manifest、回执与评估记录。
  不跟踪权重、jpg（含原始数据软链接）、`.cache`、`source_inventory_3mdad.csv`、`mining_scores.jsonl`、
  `training.log`、汇报导出物、`runtime.json`；规则在根 `.gitignore`。
  原 `yolo_phone/.gitignore` 会忽略 `runs/`、`dataset_*/`、`archive/data/seed/`，已删除并合并。
- 新增 `yolo_phone/config/current.yaml` 与 `current_config.py`，作为当前训练集、网站权重（含 SHA-256）、
  阈值 0.35、imgsz 640、NMS IoU 0.5 的唯一来源。网站服务与页面滑块、`train_seed.py`、
  `check_validation.py` 的默认值改读此文件；权重哈希不符时拒绝启动。`--conf` 默认 0.5 未改，以便和历史结果对照。
- 类别统一引用同一份 `NAMES`（`prepare_dataset`、`prepare_multiview`、`check_validation`、server），现位于 `core.labels`。
- 各轮专用脚本的默认路径保持原样，属于历史复现记录。
- 新增 `test_current_config.py`，共 16 个单测通过。网站冒烟验证通过：/health 返回 0.35 与 `eb63ae94…`，
  /api/predict 正常，页面滑块显示 0.35。验证后停止服务，恢复为原先未运行的状态。

2026-11-14 用户再次说“接着做”，完成架构建议第 3 项：抽出 `yolo_phone/core/` 包。
- 函数原样搬入，未改逻辑：
  - `core.io`：csv_rows、write_csv、write_json、digest；
  - `core.labels`：NAMES、3MDAD 的 POSITIVE/BACKGROUND、json_to_yolo、validate_label；
  - `core.splits`：EVAL_DRIVERS、split_subjects、select_frames、check_disjoint、snapshot、check_snapshot；
  - `core.metrics`：ground_truth、categorize、iou、image_status、summarize；
  - `core.paths`：BASE、artifact；`core.config`：原 `current_config.py`。
- 现行脚本、网站服务和单测改从 core 导入。删除 `project_paths.py` 和 `current_config.py`；
  `continue_split.py`、`prepare_multiview.py`、`evaluate_reviewed_holdout.py`、`annotation_io.py` 仍导出原名称，
  experiments 历史脚本的导入不需要改。`check_validation.py` 的重复 csv_rows/val_snapshot 改用 core。
- 单测可在项目根运行：`./.venv/bin/python -m unittest discover -s yolo_phone -p 'test_*.py'`，16 个通过。
- 核验：
  - 全部脚本及 val_compare/threshold_select 均能导入，State Farm eval 快照通过；
  - 用重构后代码 `--labels-only` 重跑 test：审核决策 440 行、标注哈希、类别计数与 10-21 结果完全一致；
  - 用 core.metrics 从 10-21 逐图结果重算 e79、e27 的 @0.5 图像级汇总，与历史 summary.json 一致。

2026-11-14 用户第三次说“接着做”，完成架构建议第 4 项：统一训练与导出入口。
- 新增 `yolo_phone/train.py --plan <plan.yaml> [--dry-run]` 与模板 `experiments/plan_template.yaml`：
  - 超参只写在计划的 `train:` 块，原样传给 Ultralytics；
  - imgsz 取 config（640），data/project/name/imgsz/model/resume 不许写在 `train:` 里；
  - 起点权重须与计划中的 SHA-256 一致，data.yaml 类别须与固定 id 一致，运行目录已存在则拒绝；
  - 一个计划只对应一次训练，输出 `resolved_plan.json`（含 git 提交与是否有未提交改动）、`status.json`、`result.json`。
  - 用模板（参数同第 2 轮）试跑：最终 19 个训练参数与 `official_targeted_round2_20261021/train.py` 逐项一致。
- `export_targeted_round.py` 去掉写死的第 2 轮信息：`--base`、`--round` 改为必填，新增 `--test-manifest`，
  类别与 data.yaml 改由 NAMES 生成。按第 2 轮原始输入重跑到临时目录，与现有
  `selftrain_multiview_targeted_round2_20261021/` 的 manifest（3661 行）、标签、509 个审核 JSON、
  图片链接、summary 与审核回执完全一致。`export_multiview.py` 是首轮的整目录导出，用途不同，保留。
- `prepare_targeted_round.py` 的选样计划（POSITIVE_PLAN、挖掘权重）仍是第 2 轮专用，下一轮按需求再参数化。
- 新增 `test_train_plan.py`，共 21 个单测通过。
- **轮次标签更正**：网站权重 `eb63ae94…` 实为第 52 轮，不是第 51 轮。
  - 证据：其 val 复算 mAP50 0.9133 / mAP50-95 0.5732（`val_compare/report.json`），
    对应 `results.csv` 第 52 轮（0.9133 / 0.5738）；第 51 轮为 0.9275 / 0.5724。
  - 原因：Ultralytics 8.4.172 保存 best.pt 只看 mAP50-95，第 52 轮最大；
    旧 `run.py` 用 0.1×mAP50 + 0.9×mAP50-95 公式记录最佳轮次，算出第 51 轮。
  - 权重文件及所有评估都基于实际文件，结论不变，只是标签差一轮。
  - 第 15 节等历史记录中的“e51”保留原文，README 与 config 已注明。
  - 其他 run 两种规则选出的轮次相同，混合 e27 一致。

2026-11-14 用户第四次说“接着做”，完成架构建议第 5 项的登记表部分：新增数据集登记表 `yolo_phone/config/datasets.yaml`。
- 共 32 条：8 个现行/历史数据集、多视角集内的 State Farm eval 副本 `eval_statefarm`、23 个 `archive/data/` 目录。
  每条记录路径、类型、状态（active/protected/historical/archived）、上游来源（parents）和 manifest SHA-256。
- 新增 `core/registry.py`。
  - `config/current.yaml` 的 `dataset` 改为登记表 id `train_targeted_round2`；
  - 各脚本写死的数据集目录改为按 id 查找；
  - `artifact()` 改为只查登记表，未登记即报错，不再静默回退到归档目录。
- 新增 `test_dataset_registry.py`，检查以下几项，共 26 个单测通过：
  - 登记路径存在、manifest 哈希与磁盘一致；
  - 磁盘上所有含 manifest 的目录和归档目录都已登记；
  - 来源关系无循环；当前训练集为 active。
  - 首次运行即发现未登记的 `multiview_statefarm_3mdad_v1/eval_statefarm`，已补登记为 protected。
- 核验：
  - 全部脚本及 experiments 中的 val_compare、threshold_select 均能导入；
  - labels-only 重跑 test 的审核决策与 10-21 结果逐字节一致；
  - 训练计划试跑解析出同一 data.yaml。
- 未做：把数据集目录实际移入 `data/`。登记表建好后，移动只需改 path 和 data.yaml，留待用户决定。

2026-11-14 用户选择“移进 data/”：7 个现行数据集目录从 `yolo_phone/` 顶层移入 `yolo_phone/data/`，
记录与移动前逐文件 SHA-256、链接目标见 `archive/cleanup-2026-11-14-move.json`。
- 移动前确认：项目内没有软链接指向这些目录；它们内部的 23022 个链接都指向 `~/Downloads` 原始数据或
  `archive/data`，移动后不受影响。没有运行中的训练或网站进程。
- 修改：
  - 3 个 data.yaml 的绝对 path；登记表 9 个 path；
  - `threshold_select.py`、`val_compare.py` 改按登记表 id 定位；`official_resume150` 的 eval 快照路径；
  - 训练计划模板，以及 `test_dataset_registry.py` 改为遍历 `data/` 下的嵌套 manifest。
  - 原版在 `archive/config-originals/2026-11-14/`。
- 未改：manifest、审核回执、评估 JSON、experiments plan.json、runs/*/args.yaml 中的绝对路径，
  它们是历史记录，按移动记录定位。Ultralytics `.cache` 下次读取时自动重建。
- 核验：
  - 47468 个条目中只有 3 个 data.yaml 按计划改动，其余内容与链接目标不变，链接全部可解析；
  - 26 个单测通过，State Farm eval 快照通过；
  - labels-only 重跑 test 审核决策逐字节一致；按第 2 轮输入重跑导出，与现训练集一致；
  - 当前权重在新路径 val 上 mAP50 0.9133 / mAP50-95 0.5732，与 10-21 结果一致。

2026-11-14 用户第六次说“接着做”，完成架构建议第 6 项：原始数据路径改为可配置。
- 新增 `config/sources.yaml` 与 `core/sources.py`：
  - 每个原始数据源记录 `recorded`（清单与软链接中的原绝对前缀，不改）和 `root`（本机位置，可用 `STATEFARM_ROOT`、`MDAD_ROOT` 覆盖）。
  - `localize()` 读取原图时把 recorded 前缀换算到 root；数据未移动时结果不变。
- 改读配置：
  - `prepare_dataset.py`、`prepare_multiview.py` 的 `--source` 默认值改读配置，原先写死 `/Users/.../Downloads`；
  - `prepare_multiview`、`prepare_targeted_round`、`export_targeted_round` 读原图、预测、建链接前先 `localize`；
  - 清单里记录的值不变。
- 新增 `relink_sources.py`，默认只预览；`--apply` 才改写，并写记录到 `archive/relink-*.json`；新目标缺一个就整体拒绝。
  - 当前预览：两个原始数据源都无需改写；`--internal-relative` 可改写 `archive/data` 里 665 个项目内绝对链接，未执行。
  - 把 MDAD_ROOT 设为不存在的位置时正确拒绝，未改动文件。
- data.yaml 的 `path:` 保持绝对：Ultralytics 8.4 把相对 path 接到自己的数据目录下。改为单测检查 path 与所在目录一致。
- 新增 `test_sources.py`，`test_dataset_registry.py` 加 data.yaml 检查，共 32 个单测通过。
  按第 2 轮输入重跑导出，manifest 与 3661 个链接目标和现训练集一致。
- 随后用户要求处理项目内绝对链接：已用 `relink_sources.py --internal-relative --apply` 将 `archive/data` 中 665 个
  指向项目内部的绝对链接改为相对链接（其中 65 个是 git 跟踪的 JSON 链接），记录在 `archive/relink-20261114-163359.json`。
  核验：全部 63907 个链接改写前后解析到同一文件、无失效；项目内已无绝对内部链接；32 个单测通过。
  其余链接指向 `~/Downloads` 原始数据，仍为绝对路径，由 `config/sources.yaml` 与 `--source` 模式处理。

2026-11-14 用户要求“这些脚本现在如果有没用的就删掉，或者说整理的干净一点”：
- 删除（`git rm`，可从历史找回）：
  - `annotation_io.py`、`continue_split.py`：兼容转发；
  - `train_seed.py`：已被 `train.py --plan` 取代。
  - experiments 中 `threshold_select.py`、`val_compare.py`、`official_resume150/run.py` 改从 core 导入，
    原版在 `archive/config-originals/2026-11-14b/`。
  - `archive/experiments/official_low_lr150_20261012/run.py` 仍导入已删模块，属归档记录，不再运行。
- 移入 `builders/`（附 README 对照登记表 id）：`prepare_dataset`、`prepare_multiview`、`export_multiview`、`prepare_test_review`。
  脚本开头加入 yolo_phone 到导入路径；`prepare_dataset` 默认输出改为 `data/dataset`。
- 移入 `tests/`：5 个单测文件；运行命令改为 `discover -s yolo_phone/tests -t yolo_phone`。
- 顶层只留 6 个现行入口。核验：32 个单测通过；10 个脚本 `--help` 均可启动；3 个实验脚本可导入；无残留导入。
