# 一次性数据集构建脚本

这些脚本都已执行完毕，产物登记在 `config/datasets.yaml`，现在保留只为说明数据来源，日常流程不需要运行。
现有输出目录一律拒绝覆盖；重跑须指定新输出名。

| 脚本 | 生成的数据（登记表 id） |
|---|---|
| `prepare_dataset.py` | `statefarm_base_1of8`：State Farm 1/8 抽样与 3 司机独立 eval |
| `prepare_multiview.py` | `mdad_selection_v1`：3MDAD 选样、司机划分与待审候选 |
| `export_multiview.py` | `train_mixed_v1`：首个混合训练集（从审核目录整目录导出） |
| `prepare_test_review.py` | `mdad_test_review_baseline`：3MDAD test 审核前基线 |

各数据目录 README 中写的旧命令（如 `yolo_phone/export_multiview.py`）现对应本目录下同名脚本。
