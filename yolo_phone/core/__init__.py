"""Shared building blocks for the distracted-driving YOLO project.

Scripts at the yolo_phone/ top level are command-line entry points; reusable logic lives here:
  paths    项目根目录与归档定位
  io       CSV/JSON 读写、SHA-256
  labels   固定类别 id、3MDAD 动作映射、X-AnyLabeling→YOLO 转换与标签校验
  splits   司机划分、按时间选帧、跨划分泄漏检查、eval 快照
  metrics  人工框评估的 IoU、逐图状态与汇总
  config   config/current.yaml（当前训练集、权重、阈值）
"""
