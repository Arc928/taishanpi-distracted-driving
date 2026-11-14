<!-- ppt-master-schema: design-spec/v1 -->
# 驾驶员手机使用行为检测中期汇报 - Design Spec

## I. Project Information

| Item | Value |
| --- | --- |
| Project Name | 驾驶员手机使用行为检测中期汇报 |
| Canvas Format | ppt169 (1280 × 720) |
| Page Count | 10 |
| Primary Language | zh-Hans-CN |
| Target Audience | 课题组导师与评审老师，以及组内同学；熟悉深度学习和目标检测的基本概念，关心阶段进展是否扎实、评估是否可信、后续计划是否可行 |
| Communication Intent | 先汇报阶段进展（数据、标注、评估、模型结果、演示），再如实说明当前短板，最后给出以板端部署为重点的下一阶段计划 |
| Desired Audience Outcome | 老师能看清项目已完成哪些工作、指标可信的原因（按司机划分、人工框测试集），认可当前问题判断，并同意下一阶段以 RKNN 部署和困难样本补充为重点 |
| Core Message / Ask / Action | 从只有分类标签的公开数据出发，建立了人工 + 模型协同的检测框标注流程和严格的防泄漏评估；当前模型测试 mAP50 0.937、背景误报从 24/200 降到 5/200，下一步上板部署 |
| Delivery Context | 主要为有主讲的中期汇报现场投影，约 10 分钟、9 页（应用户要求增至 10 页，加入训练结果图）；次要为会后老师查阅 |
| Artifact Afterlife | 作为中期检查材料留档，供后续结题汇报复用 |
| Reading Mode | balanced |
| Content Strategy | balanced（未填写，按默认：保留全部事实，按结论先行重组） |
| Design Style | 学术网格·结论先行（custom mode 基于 pyramid；custom visual style 基于 swiss-minimal） |
| AI Image Acquisition Path | not applicable（仅使用提供的图片） |
| Generation Mode | continuous |
| Spec Refinement | disabled |
| Speaker Notes | enabled — final Stage-2 proactive policy |
| Custom Animations | disabled — final Stage-2 proactive policy |
| Narration Audio | disabled — final Stage-2 proactive policy |
| Created Date | 2026-11-13 |

## II. Canvas Specification

| Property | Value |
| --- | --- |
| Format | PPT 16:9 |
| Dimensions | 1280 × 720 |
| viewBox | `0 0 1280 720` |
| Margins | 64 px left/right, 56 px top, 48 px bottom |
| Content Area | x 64–1216, y 56–672 |

## III. Visual Theme

### Theme Style

- **Mode**: custom
- **Mode References**: pyramid
- **Mode Behavior**: 以 pyramid 为基底：封面后先用一页交代背景与目标，正文每页标题写成结论句（如“背景误报下降约 80%”），下方用数据、流程或证据图支撑；问题页与计划页同页对照“短板→对策”，最后一页三句话收束，不做空洞的致谢页。
- **Visual style**: custom
- **Visual Style References**: swiss-minimal
- **Visual Style Behavior**: 以 swiss-minimal 为基底：近白底、左对齐模块网格、直角细线分隔，一条深蓝色竖向色带或大号数字承担版面分区；装饰几乎为零，关键指标用超大数字呈现，检测样例图用细线框排成证据格；两类行为固定用绿（texting）与橙（calling）作为语义色，与样例图中的框色一致；完全扁平，无阴影无渐变。
- **Theme**: 深蓝学术网格；页面左缘一条细深蓝竖线 + 页码作为跨页母题，封面与结尾页用整块深蓝色带分区
- **Tone**: 冷静、精确、可信

### Color Scheme

| Role | HEX | Purpose |
| --- | --- | --- |
| Background | #FFFFFF | 页面底色 |
| Secondary background | #F3F5F8 | 数据区、证据格底板 |
| Primary | #1F3A5F | 标题、色带、主要数字 |
| Accent | #E8833A | calling（打电话）语义色、关键变化强调 |
| Secondary accent | #2EAD6B | texting（玩手机）语义色、改善方向 |
| Body text | #1E2430 | 正文 |
| Secondary text | #5B6575 | 图注、来源说明、页码 |
| Divider | #D5DBE3 | 细分隔线、表格线、图片细框 |

## IV. Typography System

### Font Plan

| Role | Character (Reference) | Primary | English if non-English | Fallback tail |
| --- | --- | --- | --- | --- |
| Title | 无衬线，粗体，紧凑 | Microsoft YaHei | Arial | sans-serif |
| Body | 无衬线，常规 | Microsoft YaHei | Arial | sans-serif |

- **Title stack**: Microsoft YaHei, Arial, sans-serif
- **Body stack**: Microsoft YaHei, Arial, sans-serif

### Font Size Hierarchy

| Purpose | Anchor Size (px) |
| --- | ---: |
| Body | 24 |
| Title | 42 |
| Subtitle | 32 |
| Annotation | 18 |
| Display | 72 |
| Footnote | 14 |

## V. Layout Principles

### Deck-wide Direction

- **Hierarchy direction**: 左上结论标题 → 一句要点 → 下方证据（数据、流程、图），视线自左向右、自上而下
- **Composition tendency**: 左对齐模块网格；关键数字放大成为视觉中心；证据图排成整齐网格并配小号图注
- **Cross-page continuity**: 内容页左缘细深蓝竖线、左下页码与小号章节标签；绿 = texting、橙 = calling 在全篇保持一致
- **Spacing posture**: 内容页偏紧凑但留白充足；封面与结尾页大面积留白
- **Spacing anchors**: page margin 64 px · block gap 24 px · column gutter 32 px · corner radius 0 px · body leading 36 px

## VI. Icon Usage Specification

- **Primary bundled library**: tabler-outline
- **Stroke Width**: 2

| Icon Path | Suitable Scenarios |
| --- | --- |
| icons/tabler-outline/car.svg | |
| icons/tabler-outline/device-mobile.svg | |
| icons/tabler-outline/phone-call.svg | |
| icons/tabler-outline/database.svg | |
| icons/tabler-outline/tag.svg | |
| icons/tabler-outline/shield-check.svg | |
| icons/tabler-outline/chart-bar.svg | |
| icons/tabler-outline/target.svg | |
| icons/tabler-outline/cpu.svg | |
| icons/tabler-outline/alert-triangle.svg | |
| icons/tabler-outline/arrow-right.svg | |
| icons/tabler-outline/camera.svg | |
| icons/tabler-outline/users.svg | |
| icons/tabler-outline/flag.svg | |
| icons/tabler-outline/check.svg | |
| icons/tabler-outline/x.svg | |

## VII. Visualization Reference List

| Page | Family | Template | Usage |
| --- | --- | --- | --- |
| P03 | chart | column_chart | 训练集规模随版本增长（250 → 3449 张） |
| P07 | table | comparison_matrix | 上一版 e79 与当前 e51 在同一测试集上的指标对比 |

## VIII. Image Resource List

| Filename | Dimensions | Ratio | Purpose | Type | Image pattern | Crop Policy | Acquire Via | Status | Reference | text_policy | page_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ex_front_texting_hit.jpg | 640×480 | 1.33 | 证据：正面玩手机被正确检出 | Photo | 2×2 证据格之一，完整显示并配小号图注 | no-crop | user | Existing | 测试集 md_RGB2_SUB2ACT6F21，e51 @0.5 检出 texting 0.56，白虚线为人工框 | none | local |
| ex_side_calling_hit.jpg | 640×480 | 1.33 | 证据：侧面打电话被正确检出 | Photo | 2×2 证据格之一，完整显示并配小号图注 | no-crop | user | Existing | 测试集 md_RGB1_SUB6ACT7F25，calling 0.83 | none | local |
| ex_front_texting_miss.jpg | 640×480 | 1.33 | 证据：正面左手打字漏检 | Photo | 2×2 证据格之一，完整显示并配小号图注 | no-crop | user | Existing | 测试集 md_RGB2_SUB41ACT6F26，无检测框，白虚线为人工框 | none | local |
| ex_front_background_fp.jpg | 640×480 | 1.33 | 证据：喝水被误判为打电话 | Photo | 2×2 证据格之一，完整显示并配小号图注 | no-crop | user | Existing | 测试集 md_RGB2_SUB9ACT14F76，误报 calling 0.61 | none | local |
| demo_screenshot.png | 661×1368 | 0.48 | 本机演示网页界面截图 | Screenshot | 竖版截图放在页面一侧完整显示，另一侧列功能与性能要点 | no-crop | user | Existing | local_demo 性能验证截图（早期权重，仅展示界面） | none | local |
| train_curves.png | 2400×1200 | 2.00 | 训练过程：每轮训练损失与验证集指标曲线 | Chart | 宽幅放在页面左侧主区域完整显示，下方配图注 | no-crop | user | Existing | Ultralytics results.png，runs/multiview_targeted_round2_20261021_yolo11n，80 轮 | none | local |
| val_pr_curve.png | 2250×1500 | 1.50 | 验证集 PR 曲线（best.pt 第 51 轮复验） | Chart | 放在右侧完整显示，下方列出三个 AP 数字 | no-crop | user | Existing | Ultralytics BoxPR_curve.png，val 212 张：texting 0.854 · calling 0.973 · 全部 0.913 | none | local |

## IX. Content Outline

### Part 1: 开场

#### Slide 01 - 封面

- **Audience move**: 不了解本课题 → 知道题目、阶段与汇报人
- **Relationships**: none
- **Composition**: 左侧整块深蓝色带承载题目，右侧大面积留白放副标题与汇报信息
- **Title**: 基于 YOLO11n 的驾驶员手机使用行为检测
- **Core message**: 中期阶段性成果汇报
- **Content**:
  - 副标题：中期汇报 · 阶段性成果
  - 汇报人：（待填）· 指导老师：（待填）· 2026 年 11 月
- **Cover impact**: 一行关键数字作为钩子——“测试 mAP50 0.937 · 背景误报 24 → 5”；构图为参考

#### Slide 02 - 背景与目标

- **Audience move**: 知道题目 → 清楚要检测什么、为什么、最终跑在哪里
- **Relationships**: 两类检测目标（texting、calling）并列；它们服务于“分心驾驶预警”这一目标；部署平台是最终落点
- **Composition**: 上方一句背景；中部两张并列的类别卡（绿/橙）写清框选范围；底部一条“模型 → 部署平台”说明
- **Title**: 目标：在车内画面中实时识别驾驶员玩手机与打电话
- **Core message**: 任务是两类目标检测，最终部署到泰山派 RK3566
- **Content**:
  - 背景：分心驾驶是交通事故的重要诱因，驾驶时使用手机最常见
  - texting（玩手机）：框持机手 + 手机，不含扶方向盘的手 · calling（打电话）：框持机手 + 手机 + 附近脸部
  - 只看驾驶员，乘客使用手机不算
  - 模型 YOLO11n（轻量），输入 640 · 目标平台：泰山派（瑞芯微 RK3566，带 NPU）

### Part 2: 已完成工作

#### Slide 03 - 数据与标注

- **Audience move**: 以为有现成标注 → 明白“只有分类标签、没有框”是核心难点，以及如何解决
- **Relationships**: 两个数据来源并列；标注流程四步有先后顺序；训练集规模随版本递增
- **Composition**: 左栏两个数据源 + 四步标注流程；右栏训练集规模柱状图
- **Title**: 公开数据只有分类标签，我们用人工 + 模型协同补出检测框
- **Core message**: 建立了从分类标签到检测框的标注流程，训练集扩展到 3449 张
- **Content**:
  - 数据源：State Farm（22424 张、26 位司机、车内侧视，只有整图类别）· 3MDAD（50 人，侧面 + 正面）
  - 流程：① 人工标注种子 250 张 → ② 模型伪标签（置信度 ≥ 0.8 且无竞争错类框）→ ③ 低置信与漏检送人工复核 → ④ 新数据模型预填、人工逐张审核
  - 训练集规模：人工种子 250 · 自训练第一轮 1422 · 人工补标 1534 · 加入 3MDAD 2952 · 针对性补充 3449
- **Visualization**: dataset-growth — 柱状图，五个版本的训练集张数
- **Native-ready**: dataset-growth=yes

#### Slide 04 - 评估方法

- **Audience move**: 对指标可信度存疑 → 相信测试结果没有“背答案”
- **Relationships**: 三个集合（train / val / test）按司机互斥；val 与 test 用途对照（选模调阈值 vs 只做对照）；测试集组成为部分—整体
- **Composition**: 上方三段式 train / val / test 分区条，标出司机隔离；下方两栏：防泄漏措施 与 测试集组成
- **Title**: 按司机划分 + 哈希校验，测试集 440 张全部人工标框
- **Core message**: 评估设计保证测试图片从未进入训练，指标可信
- **Content**:
  - 按司机划分：同一司机的所有图片只在一个集合 · SHA-256 校验测试图片未进入训练
  - val 212 张：选模型、定阈值 · test 440 张（3MDAD 10 位司机）：只做最终对照
  - test 组成：可见正样本 192（texting 98 / calling 94）· 背景 200 · 手机不可见 48 张单独统计，不当背景
  - 两类指标：检测 mAP50（框的质量）· 图像级命中（阈值 0.5，类别与位置都对）

#### Slide 05 - 迭代与发现

- **Audience move**: 不知道优化走过哪些路 → 理解为什么最后选择“针对性补数据”
- **Relationships**: 三条早期发现并列；失败分析导致针对性补数据（因果）；补充数据三部分组成整体
- **Composition**: 左栏三条发现；右栏“失败分析 → 补数据”因果链，补充数据用三段数字呈现
- **Title**: 补困难样本比加大分辨率、堆简单样本更有效
- **Core message**: 失败分析定位了两类短板，据此补充 725 张针对性数据
- **Content**:
  - 早期发现：640 / 960 / 1280 分辨率下打电话命中 108 / 100 / 42（加大无益）· 伪标签自训练有天花板 · 新司机与困难姿态更有价值
  - 失败分析：漏检集中在“正面视角左手打字”（AC6）· 误报集中在“手靠近脸”的动作（喝水、整理头发等）
  - 针对性补充 725 张候选：正面左手打字 280 · 侧面打电话 105 · 困难背景 300 · val 40，人工审核后加入训练

#### Slide 06 - 训练过程

- **Audience move**: 只知道补了数据 → 看到训练过程收敛正常、验证集表现，以及最终权重如何选出
- **Relationships**: 训练曲线（过程）与 PR 曲线（结果）并列；best.pt 由 fitness 在验证集上选出
- **Composition**: 左侧宽幅训练曲线为主，右侧 PR 曲线与三个 AP 数字，底部一行训练设置
- **Title**: 训练 80 轮收敛平稳，按验证集 fitness 选出第 51 轮
- **Core message**: 补数据后的训练过程正常收敛，第 51 轮在验证集上 mAP50 0.913
- **Content**:
  - 训练曲线：每轮训练损失与验证集指标；最后 10 轮关闭 mosaic 增强，训练损失随之下降
  - 验证集（212 张）PR 曲线：texting AP50 0.854 · calling 0.973 · 全部 mAP50 0.913
  - 设置：官方 yolo11n.pt 起点 · 80 轮 · batch 16 · 输入 640 · AdamW · Apple M1 Pro（MPS）
  - best.pt 按 fitness（0.1×mAP50 + 0.9×mAP50-95）在验证集上选择，test 不参与
- **Images**: train_curves.png 与 val_pr_curve.png 均完整显示

#### Slide 07 - 主要结果

- **Audience move**: 不知道效果如何 → 看到误报大幅下降、正面视角改善，并在样例中直观看到成功与失败
- **Relationships**: e79 与 e51 在同一测试集上对比；四张样例图分成成功（2）与失败（2）两组
- **Composition**: 左侧指标对比表 + 一个大号要点数字（误报 24 → 5）；右侧 2×2 证据格（上排命中、下排漏检/误报）
- **Title**: 背景误报下降约 80%，正面视角 mAP50 提升到 0.931
- **Core message**: 针对性补数据后，测试 mAP50 0.919 → 0.937，背景误报 24 → 5
- **Content**:
  - 对比表（同一 440 张人工框测试集，e79 → e51）：mAP50 0.919 → 0.937 · mAP50-95 0.536 → 0.569 · 正面 mAP50 0.871 → 0.931 · 侧面 mAP50 0.956 → 0.931 · texting 命中 84/98 → 85/98 · calling 命中 89/94 → 85/94 · 背景误报 24/200 → 5/200
  - 说明：calling 命中减少主要是置信度落到 0.5 以下，正面 calling AP50 反而上升（0.952 → 0.981）
  - 图注：正面玩手机 命中 / 侧面打电话 命中 / 正面左手打字 漏检 / 喝水 误报为打电话；白色虚线为人工框
  - 脚注：选模与阈值只依据 val；test 为固定留出集的回归对照
- **Visualization**: model-compare — 对比表，行是指标、列是 e79 与 e51
- **Native-ready**: model-compare=yes
- **Images**: 四张样例图同尺寸 2×2 排列，上排两张成功、下排两张失败，各配一行图注

#### Slide 08 - 演示系统

- **Audience move**: 只看到离线指标 → 看到模型已能在本机实时运行
- **Relationships**: 三种输入方式并列；性能数据属于本机测试
- **Composition**: 左侧竖版截图完整显示；右侧功能与性能要点，性能数字放大
- **Title**: 本机网页演示已可实时检测，单帧推理约 7 ms
- **Core message**: 完成了支持摄像头、图片、视频输入的实时演示系统
- **Content**:
  - 输入：摄像头 · 图片 · 视频；实时画框并显示置信度
  - 阈值可调，默认 0.35（在 val 上按事先定好的规则选出）
  - 实测：画面 30 帧/秒 · 检测约 29.5 帧/秒 · 单帧推理约 7 ms
  - 脚注：测试平台 Apple M1 Pro，非开发板；截图为早期权重下的界面
- **Images**: demo_screenshot.png 竖版完整显示

### Part 3: 问题与计划

#### Slide 09 - 问题与计划

- **Audience move**: 了解成果 → 认同当前短板判断，并接受下一阶段以部署和补数据为重点
- **Relationships**: 三个问题与三项计划一一对应（短板 → 对策）
- **Composition**: 左右两栏对照：左“当前问题”、右“下一步计划”，中间用箭头连接对应行
- **Title**: 下一阶段重点：上板部署，同时补齐正面与手近脸的短板
- **Core message**: 三个短板各有对应计划，部署是下一阶段首要任务
- **Content**:
  - 问题 1：正面视角左手打字仍有漏检（正面 texting 命中 31/39）→ 计划：继续补充正面左手困难样本
  - 问题 2：手靠近脸的动作仍会误报为打电话 → 计划：补充喝水、整理头发等困难背景
  - 问题 3：跨场景泛化弱（早期模型在外部 56 张其他场景图中只检出 2 张；电脑正面摄像头效果差）→ 计划：采集贴近实际安装位置的数据并建立独立测试集
  - 部署：导出 ONNX → 转换 RKNN（INT8 量化）→ 泰山派 RK3566 上测速度与精度

#### Slide 10 - 总结

- **Audience move**: 听完细节 → 带走三条结论
- **Relationships**: 三条结论并列（流程、评估、结果与下一步）
- **Composition**: 深蓝色带呼应封面，三条结论以大号序号排列
- **Title**: 总结
- **Core message**: 流程、评估、模型三方面已打好基础，下一阶段转向板端部署
- **Content**:
  - 建立了人工 + 模型协同的标注流程，训练集扩展到 3449 张
  - 建立了按司机划分、哈希防泄漏的评估体系，测试集 440 张全部人工标框
  - 当前模型测试 mAP50 0.937、背景误报 24 → 5，并完成本机实时演示；下一步上板部署
- **Closing impact**: 第三条结论落在“下一步上板部署”；构图为参考

## X. Speaker Notes Requirements

- **Generation**: enabled
- **Filename**: match each SVG filename under `notes/`
- **Content**: 依据每页最终内容与 sources/deck_source.md 撰写讲稿，只用其中的事实与数字，不新增数据
- **Total duration**: 约 10 分钟（每页约 1 分钟，第 3、6 页可稍长）
- **Notes style**: 正式但口语化，先说本页结论，再用两三句话解释
- **Presentation purpose**: 先汇报阶段进展，再如实说明短板，最后给出以板端部署为重点的下一阶段计划
