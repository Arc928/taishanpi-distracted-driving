# Gemini 3.8 Flash 小批量边框辅助实验

本实验由用户明确授权，仅在独立目录产生候选标注，没有写回原标注或启动训练。

## 输入与保护

- 来源：`yolo_phone/open_vocab_3mdad_v1/review_dataset`。
- 12 张 train 图片，来自 S10、S11；正面/侧面各 6 张，texting/calling 各 6 张。
- 8 张 JSON 与初始预填哈希一致；4 张 JSON 已修改，只作对照，不自动认定为已完成的人工真值。
- 请求仅发送原始图片和统一标注规则，不向模型发送已有边框、人工标注或弱类别提示。
- 未使用 val/test，没有背景样本。本实验不能估计整体精度、误报率或检测 mAP。
- 请求前后核对源 2200 份标注 JSON 和 review_status.csv，变化数量为 0。
- API key 通过终端隐藏输入，仅在运行时内存中使用，不写入脚本、日志或结果。

## 模型与结果

使用用户提供的本地代理 `http://127.0.0.1:8317/v1/chat/completions`。
请求模型 ID 为代理列表中的 `gemini-3.8-flash-high`；返回响应标记为 `gemini-3.8-flash-n`。
这是代理报告的模型标识，本实验不独立验证上游模型身份。

12/12 请求成功，每张返回 1 个候选框，共 6 texting、6 calling。
坐标均通过图像范围检查。所有 JSON 保持 checked=false，没有自动接受的标签。
返回用量合计 32431 token（输入 16932、输出含推理 15499）；未提供费用估算。

## 目视检查结论

助手查看了全部 12 张叠框对比图。结果具备辅助标注价值，但仍需要人工逐张审核：

- `md_RGB2_SUB10ACT5F111`：原 DINO 有车窗位置的多余框；Gemini 只保留司机持机手和手机区域。
- `md_RGB1_SUB10ACT5F111`、`md_RGB1_SUB10ACT7F95`：原候选为空，Gemini 给出了相应司机行为框。
- `md_RGB2_SUB11ACT7F17`：Gemini calling 框下边界偏高，截掉了部分持机手，需要人工向下调整。
- texting 候选整体贴近手机与持机手；calling 框范围与原已修改参考存在差异，需统一“附近脸部”的范围。

上述为助手目视意见，不是用户完成复核的记录，不能据此直接导出训练标签。

## 使用

- `xlabel/`：12 张图片软链接与新的 X-AnyLabeling JSON，全部独立于原标注。
- `comparisons/01.jpg` 至 `06.jpg`：左侧原标注/原 DINO 候选（绿色），右侧 Gemini 候选（橙色）。
- `source_annotations_snapshot/`：本次选取的原 JSON 内容快照。
- `selection.json`、`prompt.txt`、逐图 `*_response.json`：选样、完整提示词和结构化返回。
- `summary.json`、`validation.json`、`source_hashes_before/after.json`：运行、坐标校验与原文件保护记录。

在 X-AnyLabeling 打开本目录 xlabel 即可单独查看和修正。此次结果未合并回主复核数据集。
下一次实验宜补充更多司机、左右手动作及容易误报的背景，并优先处理尚未人工标注的图片。
