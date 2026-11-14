"""Resumable, isolated Gemini preannotation; protect all observed human edits."""
import argparse
import base64
import concurrent.futures
import csv
import getpass
import hashlib
import importlib.util
import json
import os
import shutil
import threading
import time
from collections import Counter
from pathlib import Path

import requests

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'open_vocab_3mdad_v1/review_dataset'
OUTPUT=ROOT/'review_dataset'
PILOT=ROOT.parent/'experiments/gemini38flash_boxes_v1/run.py'
spec=importlib.util.spec_from_file_location('gemini_pilot',PILOT)
pilot=importlib.util.module_from_spec(spec);spec.loader.exec_module(pilot)
MODEL=pilot.MODEL
ENDPOINT=pilot.ENDPOINT
PROMPT=pilot.PROMPT+'\nFor calling, include the ENTIRE visible phone-holding hand, from fingertips through the palm to the wrist. Do not crop the lower part of that hand while enclosing the nearby face. The rectangular box may necessarily contain intervening background; keep it as tight as the required components permit.'
LOCK=threading.Lock()


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def csv_rows(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def selection():
    rows=[r for r in csv_rows(SOURCE/'manifest.csv') if r['source']=='3mdad']
    initial=json.loads((SOURCE/'review_initial_hashes.json').read_text())
    decisions={r['id']:r['decision'] for r in csv_rows(SOURCE/'review_status.csv')}
    excluded=set(json.loads((ROOT/'additional_exclusions.json').read_text())) if (ROOT/'additional_exclusions.json').exists() else set()
    prepared=[]
    for r in rows:
        path=SOURCE/r['annotation_json'];data=json.loads(path.read_text());digest=sha(path)
        reasons=[]
        if digest!=initial[r['id']]:reasons.append('changed_since_initial_prefill')
        if data.get('checked'):reasons.append('checked_true')
        if decisions[r['id']]!='pending':reasons.append('explicit_review_decision')
        if r['id'] in excluded or r['review_batch'] in excluded:reasons.append('user_exclusion')
        prepared.append({**r,'source_json_sha256':digest,'protected_reasons':reasons})
    return prepared


def prepare():
    if (ROOT/'selection.json').exists():raise ValueError('Already prepared; use --run to resume')
    rows=selection()
    if OUTPUT.exists():raise ValueError('Output exists; refusing overwrite')
    shutil.copytree(SOURCE,OUTPUT,symlinks=True)
    (ROOT/'source_metadata').mkdir()
    for name in ('README.md','summary.json','audit.json','model_provenance.json','prefill_hashes.json','review_initial_hashes.json'):
        p=OUTPUT/name
        if p.exists():shutil.copy2(p,ROOT/'source_metadata'/name)
    for name in ('summary.json','audit.json','model_provenance.json','prefill_hashes.json'):
        p=OUTPUT/name
        if p.exists():p.unlink()  # These are copied stale generated records, not user files.
    (ROOT/'responses').mkdir()
    (ROOT/'prompt.txt').write_text(PROMPT)
    save(ROOT/'selection.json',rows)
    save(ROOT/'source_hashes_before.json',{r['annotation_json']:r['source_json_sha256'] for r in rows})
    save(OUTPUT/'BUILDING.json',{'status':'gemini_preannotation_in_progress_not_training_ready'})
    save(ROOT/'status.json',{'stage':'prepared','total_images':len(rows),'protected':sum(bool(r['protected_reasons']) for r in rows),
                          'eligible':sum(not r['protected_reasons'] for r in rows),'success':0})
    (OUTPUT/'README.md').write_text('''# Gemini 3.8 Flash 辅助预标注复核副本

本目录来自 open_vocab_3mdad_v1/review_dataset，所有原图划分与 1534 张 State Farm 标签保持不变。
当前生成进度见上级 status.json；BUILDING.json 存在时表示生成或最终核验尚未完成。

已修改 JSON、checked=true、明确审核状态以及用户指定范围均受保护，不替换为 Gemini 框。
其余 3MDAD 图片使用固定提示词和 Gemini 3.8 Flash 辅助生成框；全部仍须人工审核。
texting 框手机及持机手；calling 框手机、完整持机手和附近脸部。只识别司机。
无框不代表确认背景；源正样本无框应补标或排除，不能当背景训练。

在 X-AnyLabeling 打开 review/train、review/val、review/test 下的相应批次。
review_status.csv 保留原人工状态；Gemini 不自动修改 decision，不自动接受任何标签。
val/test 的候选也须独立人工审核，不得根据 test 表现反复调整提示词、阈值或标注标准。

已修改图片的原始初始哈希保留，Gemini 新候选使用本轮生成哈希；两类来源另见 gemini_preservation.json。
使用 export_multiview.py 前，仍须满足人工复核确认与验证集审核条件。
''')
    print((ROOT/'status.json').read_text(),flush=True)


def run(workers):
    key=getpass.getpass('API key (hidden): ')
    rows=json.loads((ROOT/'selection.json').read_text())
    assert (ROOT/'prompt.txt').read_text()==PROMPT
    initial=json.loads((ROOT/'source_metadata/review_initial_hashes.json').read_text())
    results={};began=time.time()

    def changed(row):
        p=SOURCE/row['annotation_json']
        if sha(p)!=row['source_json_sha256']:return True
        if json.loads(p.read_text()).get('checked'):return True
        # A user may mark a review decision while the batch is running.
        status={r['id']:r['decision'] for r in csv_rows(SOURCE/'review_status.csv')}
        return status[row['id']]!='pending'

    def protect(row,reason):
        shutil.copy2(SOURCE/row['annotation_json'],OUTPUT/row['annotation_json'])
        return {'id':row['id'],'status':'protected','reason':reason,'source_sha256':sha(SOURCE/row['annotation_json'])}

    def process(row):
        if row['protected_reasons']:return protect(row,','.join(row['protected_reasons']))
        if changed(row):return protect(row,'source_modified_during_run')
        response_path=ROOT/'responses'/f'{row["id"]}.json'
        if response_path.exists():
            result=json.loads(response_path.read_text())
            if result['prompt_sha256']!=hashlib.sha256(PROMPT.encode()).hexdigest():raise ValueError('Prompt changed across resume')
        else:
            payload={'model':MODEL,'temperature':0,'max_tokens':1600,'messages':[{'role':'user','content':[
                {'type':'text','text':PROMPT},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode((SOURCE/row['image']).read_bytes()).decode()}}
            ]}]}
            for attempt in range(3):
                try:
                    started=time.time()
                    response=requests.post(ENDPOINT,headers={'Authorization':'Bearer '+key},json=payload,timeout=(15,180))
                    if not response.ok:raise RuntimeError(f'HTTP_{response.status_code}')
                    raw=response.json();parsed=pilot.parse_response(raw['choices'][0]['message']['content'])
                    result={'id':row['id'],'status':'success','requested_model':MODEL,'returned_model':raw.get('model'),
                            'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),'result':parsed,'usage':raw.get('usage'),
                            'seconds':round(time.time()-started,2),'attempt':attempt+1}
                    save(response_path,result)
                    break
                except Exception as error:
                    if attempt==2:return {'id':row['id'],'status':'failed','error_type':type(error).__name__}
                    time.sleep(4*(attempt+1))
        if changed(row):return protect(row,'source_modified_during_run')
        shapes=[]
        for obj in result['result']['objects']:
            y1,x1,y2,x2=obj['box_2d'];x1,x2=x1*.64,x2*.64;y1,y2=y1*.48,y2*.48
            shapes.append({'label':obj['label'],'shape_type':'rectangle','points':[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                           'flags':{},'group_id':None,'description':'Gemini candidate; pending human review, not ground truth.'})
        save(OUTPUT/row['annotation_json'],{'version':'4.1.0','flags':{},'checked':False,'shapes':shapes,
             'imagePath':Path(row['image']).name,'imageData':None,'imageWidth':640,'imageHeight':480,
             'description':result['result'].get('note','')})
        return {**result,'box_count':len(shapes),'json_sha256':sha(OUTPUT/row['annotation_json'])}

    def progress(stage):
        counts=Counter(r['status'] for r in results.values())
        save(ROOT/'status.json',{'stage':stage,'total_images':len(rows),'eligible_initial':sum(not r['protected_reasons'] for r in rows),
             'processed':len(results),'counts':dict(counts),'elapsed_seconds':round(time.time()-began),'workers':workers,
             'boxes':sum(r.get('box_count',0) for r in results.values()),'training_started':False})

    progress('running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(process,row):row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row=futures[future]
            try:result=future.result()
            except Exception as error:result={'id':row['id'],'status':'failed','error_type':type(error).__name__}
            results[row['id']]=result
            progress('running')
            if len(results)%25==0:print((ROOT/'status.json').read_text(),flush=True)

    # Re-check all inputs after requests, preserving edits made by the user meanwhile.
    current=selection();current_by_id={r['id']:r for r in current}
    baseline=dict(initial);preservation={};source_after={}
    for row in rows:
        now=current_by_id[row['id']]
        source_after[row['annotation_json']]=now['source_json_sha256']
        if now['protected_reasons'] or now['source_json_sha256']!=row['source_json_sha256']:
            results[row['id']]=protect(row,'protected_at_final_check')
            preservation[row['id']]={'reason':now['protected_reasons'],'sha256':sha(OUTPUT/row['annotation_json'])}
        elif results[row['id']]['status']=='success':
            baseline[row['id']]=sha(OUTPUT/row['annotation_json'])
    shutil.copy2(SOURCE/'review_status.csv',OUTPUT/'review_status.csv')
    save(ROOT/'source_hashes_after.json',source_after)
    save(OUTPUT/'review_initial_hashes.json',baseline)
    save(OUTPUT/'gemini_preservation.json',preservation)
    save(ROOT/'results.json',list(results.values()))
    counts=Counter(r['status'] for r in results.values())
    usage=Counter()
    for r in results.values():usage.update({k:v for k,v in (r.get('usage') or {}).items() if type(v)==int})
    save(OUTPUT/'model_provenance.json',{'model':MODEL,'endpoint':ENDPOINT,'prompt_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
         'prompt_file':str(ROOT/'prompt.txt'),'image_size':[640,480],'coordinate_scale':1000,'prefilled_splits':['train','val','test'],
         'same_parameters_all_splits':True,'automatically_accepted':0,'protected_original_annotations':len(preservation),
         'returned_models':sorted({r['returned_model'] for r in results.values() if 'returned_model' in r})})
    frozen=json.loads((OUTPUT/'selection_hashes.json').read_text())
    assert all(sha(OUTPUT/name)==value for name,value in frozen.items())
    inherited=[r for r in csv_rows(OUTPUT/'manifest.csv') if r['source']=='statefarm']
    assert len(inherited)==1534 and all(sha(OUTPUT/r['label'])==r['label_sha256'] for r in inherited)
    assert all(sha(OUTPUT/row['annotation_json'])==source_after[row['annotation_json']] for row in rows if row['id'] in preservation)
    save(OUTPUT/'summary.json',{'status':'awaiting_human_review_not_training_ready','total_3mdad':len(rows),'counts':dict(counts),
         'original_dataset':str(SOURCE),'inherited_training':len(inherited),'split_unchanged':True,'training_started':False,
         'auto_accepted':0,'usage':dict(usage),'source_changed_during_run':sum(source_after[r['annotation_json']]!=r['source_json_sha256'] for r in rows)})
    save(OUTPUT/'audit.json',{'frozen_selection_verified':True,'inherited_labels_verified':True,
         'protected_annotations_match_source':True,'pending_annotations_not_auto_accepted':True})
    if not counts['failed']:(OUTPUT/'BUILDING.json').unlink(missing_ok=True)
    progress('complete' if not counts['failed'] else 'partial_failure_resume_needed')
    print((ROOT/'status.json').read_text(),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--run',action='store_true')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.prepare:prepare()
    if args.run:run(args.workers)
