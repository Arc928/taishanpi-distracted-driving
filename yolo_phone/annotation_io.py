"""Shared X-AnyLabeling-to-YOLO conversion used by current dataset tools."""
import json

NAMES = {0: 'texting', 1: 'calling'}


def json_to_yolo(path):
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
