"""Image-level scoring against human-reviewed boxes.

A source phone action with no reviewed box is "not_visible": it is reported
separately and never counted as background.
"""
from collections import Counter

VIEWS = ('all', 'front', 'side')


def ground_truth(text, width=640, height=480):
    boxes = []
    for line in text.splitlines():
        c, x, y, w, h = line.split()
        x, y, w, h = float(x)*width, float(y)*height, float(w)*width, float(h)*height
        boxes.append((int(c), [x-w/2, y-h/2, x+w/2, y+h/2]))
    return boxes


def categorize(row, gt, excluded):
    if row['id'] in excluded:
        return 'exclude'
    if gt:
        return 'positive'
    return 'background' if row['source_behavior'] == 'background_candidate' else 'not_visible'


def iou(a, b):
    w = max(0, min(a[2], b[2])-max(a[0], b[0]))
    h = max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-w*h
    return w*h/union if union > 0 else 0


def image_status(category, gt, preds):
    """preds are already filtered to the reporting threshold."""
    classes = {p['class_id'] for p in preds}
    if category == 'background':
        return 'false_positive' if preds else 'clear'
    if category == 'not_visible':
        return 'fired' if preds else 'silent'
    if not classes & {c for c, _ in gt}:
        return 'wrong_class_only' if preds else 'missed'
    located = all(any(p['class_id'] == c and iou(p['xyxy'], b) >= .5 for p in preds) for c, b in gt)
    return 'localized_hit' if located else 'class_hit_poor_box'


def summarize(rows):
    out = {}
    for view in VIEWS:
        group = [r for r in rows if view == 'all' or r['view'] == view]
        s = {}
        for name in ('texting', 'calling', 'mixed'):
            pos = [r for r in group if r['category'] == 'positive' and r['gt_class'] == name]
            if pos:
                c = Counter(r['status'] for r in pos)
                s[name] = {'total': len(pos), 'class_hit': c['localized_hit']+c['class_hit_poor_box'],
                           'localized_hit': c['localized_hit'], 'wrong_class_only': c['wrong_class_only'],
                           'missed': c['missed']}
        bg = [r for r in group if r['category'] == 'background']
        s['background'] = {'total': len(bg), 'false_positive': sum(r['status'] == 'false_positive' for r in bg)}
        nv = [r for r in group if r['category'] == 'not_visible']
        s['not_visible'] = {'total': len(nv), 'fired': sum(r['status'] == 'fired' for r in nv)}
        out[view] = s
    return out
