# 混合数据训练：2026-10-10

来源：gemini38flash_3mdad_v1/review_dataset，加上其已继承的 State Farm 1534 张（不重复加入）。

用户训练请求及“我应该是刚修正完，你可以看下修正的时间确认一下”记录在 human_review_receipt.json。
核对六个训练批次均有内容修改，最后保存于 20:03:34；验证集修改从 20:03:44 到 20:07:56。
时间用于佐证用户完成说明，不将修改时间本身当作人工真值。原始 JSON 和 CSV 均未修改。
复核快照在 ../gemini38flash_3mdad_v1/training_review_snapshot_20261010/。

训练 2952 张：State Farm 1534 + 3MDAD 1418；验证 200 张，为 3MDAD 的 5 位独立司机。
排除仍无框的源正样本：train 122 / val 20；这些没有作为背景导出。
验证结果只针对保留的 200/220 张，不能掩盖被排除的 20 张正样本；源正样本覆盖 100/120。
State Farm p016/p021/p026 的 452 张独立评估继续隔离。3MDAD test 没有进入本次配置。

YOLO11n 从原第二轮 best.pt 继续训练，640、batch 16、最多 30 轮、patience 10、MPS。
运行目录：../runs/multiview_gemini_reviewed_20261010_yolo11n/。
具体训练参数以运行目录 args.yaml 为准，进度见 training.log 和运行目录 results.csv。
review_completion_evidence.json 保存逐文件修改时间、哈希和决策；reviewed_json/ 保存导出时的标注副本。

## 训练已完成

2026-10-10 约 21:19 正常完成 30 轮，训练约 65.4 分钟；best.pt 来自第 27 轮。
最终验证：P 0.936、R 0.789、mAP50 0.914、mAP50-95 0.580；仅为筛选后 200 张 val 的指标。
尚未执行最终独立 test，也未切换本机服务模型。完成状态和权重哈希见 training_status.json。
