# 历史归档

2026-10-09 按用户确认的清理方案整理。本目录保存历史数据与过程，当前工作入口在上一级 README。

- `data/`：18 个历史数据、标注、预测和评估目录，包括旧五司机划分。
- `scripts/`：原有 10 个历史脚本，2026-11-02 已随新增的 6 个一并删除（移入废纸篓，见文末）。
- `docs/`：7 份历史阶段记录，反映当时目录结构。
- `scratch/`：旧分辨率对照日志。
- `config-originals/`：更新路径前的历史 YAML 和两份当前 manifest 的原始副本。
- `cleanup-2026-10-09.json`：原路径→归档路径、废纸篓清理清单及完整性核验。

人工标注、标签、JSON、统计 CSV 与保留的模型内容逐一校验为相同哈希。
旧五司机 eval 的 1021 个文件与新三司机 eval 的 716 个文件快照均一致。
旧 `seed/images/train` 的真实图片仍保存在 `data/seed/images/train`，相关软链接已修复。

归档数据的 `data.yaml` 已适配新位置；原版本在 `config-originals/`。
历史 JSON、统计和文档中的绝对路径保留原始记录，可用清理记录里的 `archived_paths` 映射定位。
当前迁移和比较脚本通过 `project_paths.artifact()` 自动定位归档内容。
历史脚本保留当时路径假设，不能作为当前默认入口；其中 YOLO-World 自动标注流程已停用。

自动预测图、拼图、训练图表与旧 YOLO-World/CLIP 权重移到了本次专用废纸篓文件夹，
其位置见清理 JSON 的 `trash_root`。这些内容可恢复；尚未清空废纸篓。
评估的原始 `raw_predictions.json` 与统计记录保留，可据此重新绘制预览。

## 2026-10-21 第二次清理

用户选择四项，记录在 `cleanup-2026-10-21.json`，其中有每个移入废纸篓文件的 SHA-256。
- 移入专用废纸篓文件夹：
  - 32 个中间检查点（各实验 epochN.pt 与 resume_epoch*.pt，约 648 MiB）；
  - 已放弃的 `runs/multiview_official_low_lr150_20261012_yolo11n`；
  - 被 gemini38flash 审核目录取代的 `open_vocab_3mdad_v1`；
  - 全部 `__pycache__` 与 `.DS_Store`。
- 移入本目录，哈希核验一致：
  - `experiments/official_low_lr150_20261012`；
  - `reports/` 下 3 份旧整图口径测试报告：test_multiview_20261010、
    test_official_init_20261011、test_official_resume100_20261012。
- 所有 run 的 best.pt/last.pt/results.csv/args.yaml 均保留。删除检查点后，不能再带优化器状态从中间轮次续训。

## 2026-11-02 第三次清理

用户全选四项，记录在 `cleanup-2026-11-02.json`（移动清单、逐文件 SHA-256、路径修改前后哈希）。
- `data/` 新增 State Farm 阶段 5 个目录：seed_23train_reviewed、selftrain_23train_round2、
  review_23train_round1、remaining_23train_round2、validation_23train_round2。
- `docs/` 新增 SPLIT23_ROUND2.md；`scripts/` 新增 6 个旧脚本（导入关系保留原样，不能直接运行）。
- 被修改路径的配置原版在 `config-originals/2026-11-02/`，按原相对路径存放。
- 评估结果移到上一级 `evals/`。

2026-11-02 随后按用户选择删除（移入废纸篓 `taishan-delete-2026-11-02-57b5cd98`，记录 `cleanup-2026-11-02-delete.json`）：
- 本目录 `scripts/` 全部 16 个旧脚本与空目录 `scratch/`；
- 针对性训练 run 的 16 个中间检查点（324 MiB）。
