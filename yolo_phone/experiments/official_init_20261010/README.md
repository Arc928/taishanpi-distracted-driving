# 官方 YOLO11n 起点对照

仅更换起始权重为官方 COCO yolo11n.pt，使用与上一轮相同的混合数据、划分和 30 轮训练上限。
640、batch 16、MPS、patience 10。官方发布文件与本地 SHA-256 已比对一致。
训练结果在 ../../runs/multiview_official_yolo11n_20261010/。进度见 training.log，生命周期状态见 status.json。
完成后 result.json 比较两次训练各自最佳验证指标；同预算对照不保证两者都充分收敛。
验证仅为筛选后的 200 张，排除了 20 张无框源正样本；不使用 test 选模，不切换演示服务。
