# State Farm + 3MDAD 多视角数据集 v1

这是已完成划分、抽样和预标注的待复核数据集，尚不能直接训练。

- 继承 State Farm 1534 张有效训练样本，保留原标签及其人工/伪标签来源，不将伪标签升级成人工真值。
- 3MDAD 50 人按 seed=42 固定为 35 train / 5 val / 10 test；每位司机所有视角、所有原始帧同组。
- 每个司机、每个视角，AC5/6/7/8 每个动作取 3 个等分区间的中点帧；10 类背景候选各取 1 帧，共 44 张/人、2200 张。
- AC4 操作 GPS、AC9 拍照暂不采样；完整 222034 张来源及划分在 source_inventory_3mdad.csv。
- 同步双视角不会跨 split；按司机和图片 SHA-256 检查泄漏。原 State Farm p016/p021/p026 完全隔离。
- 两类固定 0 texting / 1 calling；训练和推理尺寸 640。

## 标注

在 X-AnyLabeling 打开 `/Users/arc928/Documents/ChatGPT/泰山派/yolo_phone/multiview_statefarm_3mdad_v1/review/train/train_01`（第一批 280 张，覆盖 35 位司机、两个视角和四个手机动作）。
其他批次在 review/train/ 下。验证、测试分别在 review/val/、review/test/，按原图独立标注，未提供模型预填。

texting 框持机手和手机，不含扶方向盘的手；calling 框持机手、手机及附近脸部，不框整个人。只识别司机。
训练预填框仅供辅助，必须逐张核对。空 shapes、没有模型预测，均不表示已确认背景。
原始 .mat 的头/手框不是行为框，未自动转换。

JSON 保存后，在 review_status.csv 将相应 decision 改为：

- `positive`：存在正确的 texting/calling 人工框；允许人工纠正弱行为类别。
- `background`：人工确认无目标，且 JSON 无框；仅允许源背景候选。
- `exclude`：遮挡、动作过渡、不可判断或无框源正样本；note 写明原因。
- `pending`：尚未完成。

只改 JSON 不代表已审核。最终导出还需记录用户明确的复核完成确认。
源 AC5–AC8 没有框必须排除，不会导出为空背景标签。
val 用于新训练的模型选择；test 是最终独立测试，不能用于调阈值或反复挑模型。
原 State Farm 452 张在 eval_statefarm/，仅保持原图像级评估，不能当检测 mAP 真值。

## 导出

使用 `export_multiview.py --dataset <本目录> --output <新目录> --confirmation <明确的复核完成记录>`。
导出需至少有已复核的 3MDAD train 和完整已复核的 val；test 全部复核后才会写入 YOLO test 配置。
未审核图片不会出现在导出的训练目录，已有输出拒绝覆盖。

## 文件说明

- manifest.csv：3734 张选样清单，来源、划分、框状态与路径。
- inherited/：1534 张原训练图片软链接和标签副本。
- review/：2200 张新数据软链接及独立可编辑 JSON；没有未审核 YOLO 空标签。
- train_proposals.json、model_provenance.json：训练区模型预测与权重哈希（若启用预填）。
- selection_hashes.json：划分与选样冻结校验；review_initial_hashes.json：初始 JSON 校验。

3MDAD 原始数据位于 /Users/arc928/Downloads/3MDAD；许可 CC BY-NC-ND 4.0，教育/研究、非商业用途。
本次构建没有修改原数据、现有训练集、独立评估文件，也没有启动训练。
