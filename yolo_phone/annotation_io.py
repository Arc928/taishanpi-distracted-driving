"""兼容旧导入：标注转换已移入 core.labels（experiments/ 中的历史脚本仍从这里导入）。"""
from core.labels import NAMES, json_to_yolo  # noqa: F401
