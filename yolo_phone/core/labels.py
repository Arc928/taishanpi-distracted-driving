"""Fixed class ids, 3MDAD action mapping and label conversion/validation."""
import json
import math

# 固定目标 id，顺序不得改变（见 AGENTS.md 第 1 节）
NAMES = {0: 'texting', 1: 'calling'}

# 3MDAD 动作编号：AC5/6 打字、AC7/8 打电话；其余采样动作为背景候选（AC4、AC9 不采样）
POSITIVE = {5: 'texting', 6: 'texting', 7: 'calling', 8: 'calling'}
BACKGROUND = {1, 2, 3, 10, 11, 12, 13, 14, 15, 16}


def json_to_yolo(path):
    """Convert one X-AnyLabeling JSON file to YOLO label text."""
    data = json.loads(path.read_text(encoding='utf-8'))
    width, height = float(data['imageWidth']), float(data['imageHeight'])
    lines = []
    for shape in data.get('shapes', []):
        if shape.get('shape_type') != 'rectangle':
            raise ValueError(f'{path}: only rectangle boxes are supported')
        label = shape.get('label')
        if label not in NAMES.values():
            raise ValueError(f'{path}: unexpected label {label!r}')
        xs, ys = zip(*shape['points'])
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f'{path}: invalid box {x1, y1, x2, y2}')
        cls = 0 if label == 'texting' else 1
        lines.append(f'{cls} {(x1+x2)/2/width:.6f} {(y1+y2)/2/height:.6f} '
                     f'{(x2-x1)/width:.6f} {(y2-y1)/height:.6f}\n')
    return ''.join(lines)


def validate_label(text, row):
    """Reject empty positives and malformed or out-of-range YOLO boxes."""
    lines = text.splitlines()
    if not lines and row['class_name'] != 'background':
        raise ValueError(f"正样本不能为空标签: {row['image']}")
    for line in lines:
        values = line.split()
        if len(values) != 5 or values[0] not in ('0', '1'):
            raise ValueError(f"非法标签: {row['image']}")
        x, y, w, h = map(float, values[1:])
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError('非有限坐标')
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"坐标越界: {row['image']}")
        if not (-1e-6 <= x-w/2 and x+w/2 <= 1+1e-6
                and -1e-6 <= y-h/2 and y+h/2 <= 1+1e-6):
            raise ValueError(f"框越界: {row['image']}")
