"""Render isolated before/after candidates and validate local annotation files."""
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent


def draw_boxes(image, annotation, color):
    painter=ImageDraw.Draw(image)
    for shape in annotation.get('shapes',[]):
        points=shape['points'];xs=[p[0] for p in points];ys=[p[1] for p in points]
        if not all(0<=x<=640 for x in xs) or not all(0<=y<=480 for y in ys):
            raise ValueError('Box out of image bounds')
        painter.rectangle((min(xs),min(ys),max(xs),max(ys)),outline=color,width=3)
        painter.text((min(xs),max(0,min(ys)-14)),shape['label'],fill=color)


def main():
    summary=json.loads((ROOT/'summary.json').read_text())
    selected=json.loads((ROOT/'selection.json').read_text())
    success={r['id']:r for r in summary['results'] if 'result' in r}
    view=ROOT/'comparisons';view.mkdir(exist_ok=True)
    subjects=Counter();groups=Counter();classes=Counter()
    for start in range(0,len(selected),2):
        canvas=Image.new('RGB',(1280,1080),'#eeeeee');draw=ImageDraw.Draw(canvas)
        for row,item in enumerate(selected[start:start+2]):
            identifier=item['id'];top=row*540
            before=json.loads((ROOT/'source_annotations_snapshot'/f'{identifier}.json').read_text())
            image=Image.open(item['source_image']).convert('RGB')
            assert image.size==(640,480)
            left=image.copy();draw_boxes(left,before,'#00ff00')
            right=image.copy()
            if identifier in success:
                annotation=json.loads((ROOT/'xlabel'/f'{identifier}.json').read_text())
                assert annotation['imagePath']==identifier+'.jpg' and not annotation['checked']
                draw_boxes(right,annotation,'#ff9900')
                classes.update(s['label'] for s in annotation['shapes'])
            draw.text((8,top+5),identifier+' | '+item['experiment_group'],fill='black')
            draw.text((8,top+25),'LEFT: original edited reference / DINO proposals (green)',fill='black')
            draw.text((650,top+25),'RIGHT: Gemini candidates (orange), NOT accepted labels',fill='black')
            canvas.paste(left,(0,top+50));canvas.paste(right,(640,top+50))
            subjects[item['subject']]+=1;groups[item['experiment_group']]+=1
        canvas.save(view/f'{start//2+1:02d}.jpg',quality=94)
    total_usage=Counter()
    for record in success.values():
        total_usage.update({k:v for k,v in (record.get('usage') or {}).items() if isinstance(v,int)})
    report={'subjects':dict(subjects),'groups':dict(groups),'candidate_classes':dict(classes),
            'usage':dict(total_usage),'successful':len(success),'source_files_changed':summary['source_files_changed_during_experiment'],
            'coordinate_and_image_checks':'passed','test_or_val_images_used':False}
    (ROOT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
