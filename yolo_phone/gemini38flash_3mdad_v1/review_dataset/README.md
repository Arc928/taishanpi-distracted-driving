# Gemini 3.8 Flash 辅助预标注复核副本

本目录来自 open_vocab_3mdad_v1/review_dataset，所有原图划分与 1534 张 State Farm 标签保持不变。
当前生成进度见上级 status.json；BUILDING.json 存在时表示生成或最终核验尚未完成。

已修改 JSON、checked=true、明确审核状态以及用户指定范围均受保护，不替换为 Gemini 框。
其余 3MDAD 图片使用固定提示词和 Gemini 3.8 Flash 辅助生成框；全部仍须人工审核。
texting 框手机及持机手；calling 框手机、完整持机手和附近脸部。只识别司机。
无框不代表确认背景；源正样本无框应补标或排除，不能当背景训练。

在 X-AnyLabeling 打开 review/train、review/val、review/test 下的相应批次。
review_status.csv 保留原人工状态；Gemini 不自动修改 decision，不自动接受任何标签。
val/test 的候选也须独立人工审核，不得根据 test 表现反复调整提示词、阈值或标注标准。

已修改图片的原始初始哈希保留，Gemini 新候选使用本轮生成哈希；两类来源另见 gemini_preservation.json。
使用 export_multiview.py 前，仍须满足人工复核确认与验证集审核条件。
