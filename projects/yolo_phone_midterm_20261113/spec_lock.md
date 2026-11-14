<!-- ppt-master-schema: spec-lock/v1 -->
# Execution Lock

## canvas
- viewBox: 0 0 1280 720
- format: PPT 16:9

## communication
- primary_language: zh-Hans-CN
- audience: 课题组导师与评审老师及组内同学，熟悉目标检测基本概念，关心进展是否扎实、评估是否可信、计划是否可行
- objective: 汇报阶段进展并如实说明短板，让老师认可指标可信、同意下一阶段以 RKNN 部署和困难样本补充为重点
- core_message: 从只有分类标签的公开数据出发建立了人工 + 模型协同标注与防泄漏评估；测试 mAP50 0.937、背景误报 24/200 → 5/200，下一步上板部署
- consumption_mode: balanced

## mode
- mode: custom
- mode_references: pyramid
- mode_behavior: 以 pyramid 为基底：封面后先用一页交代背景与目标，正文每页标题写成结论句，下方用数据、流程或证据图支撑；问题页与计划页同页对照“短板→对策”，最后一页三句话收束，不做空洞的致谢页。

## visual_style
- visual_style: custom
- visual_style_references: swiss-minimal
- visual_style_behavior: 以 swiss-minimal 为基底：近白底、左对齐模块网格、直角细线分隔，一条深蓝色竖向色带或大号数字承担版面分区；装饰几乎为零，关键指标用超大数字呈现，检测样例图用细线框排成证据格；绿色表示 texting、橙色表示 calling，与样例图框色一致；完全扁平，无阴影无渐变。

## colors
- background: #FFFFFF
- secondary_bg: #F3F5F8
- primary: #1F3A5F
- accent: #E8833A
- secondary_accent: #2EAD6B
- body_text: #1E2430
- secondary_text: #5B6575
- divider: #D5DBE3

## typography
- font_family: Microsoft YaHei, Arial, sans-serif
- title_family: Microsoft YaHei, Arial, sans-serif
- body_family: Microsoft YaHei, Arial, sans-serif
- body: 24
- title: 42
- subtitle: 32
- annotation: 18
- display: 72
- footnote: 14

## icons
- library: tabler-outline
- stroke_width: 2
- inventory: tabler-outline/car, tabler-outline/device-mobile, tabler-outline/phone-call, tabler-outline/database, tabler-outline/tag, tabler-outline/shield-check, tabler-outline/chart-bar, tabler-outline/target, tabler-outline/cpu, tabler-outline/alert-triangle, tabler-outline/arrow-right, tabler-outline/camera, tabler-outline/users, tabler-outline/flag, tabler-outline/check, tabler-outline/x

## images
- ex_texting_hit: images/ex_front_texting_hit.jpg | source=user | crop=no-crop
- ex_calling_hit: images/ex_side_calling_hit.jpg | source=user | crop=no-crop
- ex_texting_miss: images/ex_front_texting_miss.jpg | source=user | crop=no-crop
- ex_background_fp: images/ex_front_background_fp.jpg | source=user | crop=no-crop
- demo: images/demo_screenshot.png | source=user | crop=no-crop
- train_curves: images/train_curves.png | source=user | crop=no-crop
- val_pr_curve: images/val_pr_curve.png | source=user | crop=no-crop

## page_rhythm
- P01: anchor
- P02: breathing
- P03: dense
- P04: dense
- P05: dense
- P06: dense
- P07: dense
- P08: breathing
- P09: dense
- P10: anchor

## page_visualizations
- P03: chart/column_chart
- P07: table/comparison_matrix

## pptx_structure
- mode: flat

## forbidden
- `mask`, `<style>`, `class`, external CSS, `<foreignObject>`, `textPath`, `@font-face`, `<animate*>`, `<set>`, `<script>` / event attributes, `<iframe>`
- HTML named entities in text; write typography as raw Unicode and escape XML reserved characters
