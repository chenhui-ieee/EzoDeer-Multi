"""Evaluate the validation-selected checkpoint; fixed-threshold metrics use confidence 0.25."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

from ..runtime import settings,detection_run,platform_details
ARGS=settings('detection')
HERE=ARGS.output
RUN=detection_run(HERE)
NAMES=['male','female','unknown']

def iou_xywh(a,b):
    ax,ay,aw,ah=a;bx,by,bw,bh=b
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return inter/(aw*ah+bw*bh-inter+1e-12)

def fixed_metrics(split,predictions):
    by_image=collections.defaultdict(list)
    for p in predictions:
        if p['score']>=0.25:by_image[str(p['image_id'])].append(p)
    count=np.zeros((3,3),dtype=int) # true positives, false positives, false negatives
    rows=[]
    for path in sorted((HERE/'yolo_data'/'labels'/split).glob('*.txt')):
        # Cached images all have a 960-pixel width and a 540-pixel height.
        gt=[]
        for line in path.read_text().splitlines():
            cls,xc,yc,w,h=map(float,line.split())
            gt.append((int(cls),[(xc-w/2)*960,(yc-h/2)*540,w*960,h*540]))
        used=set(); local=np.zeros((3,3),dtype=int)
        for p in sorted(by_image[path.stem],key=lambda x:-x['score']):
            cls=p['category_id']-1
            candidates=[(iou_xywh(p['bbox'],b),i) for i,(c,b) in enumerate(gt) if c==cls and i not in used]
            overlap,index=max(candidates,default=(0,-1))
            if overlap>=0.5:used.add(index);local[cls,0]+=1
            else:local[cls,1]+=1
        for i,(cls,b) in enumerate(gt):
            if i not in used:local[cls,2]+=1
        count+=local
        rows.append(dict(image=path.stem,**{f'{name}_{metric}':int(local[c,k]) for c,name in enumerate(NAMES) for k,metric in enumerate(['tp','fp','fn'])}))
    result={}
    for name,values in list(zip(NAMES,count))+[('all_micro',count.sum(0))]:
        tp,fp,fn=map(int,values)
        result[name]=dict(tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else 0,
            recall=tp/(tp+fn) if tp+fn else 0,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    with (HERE/f'{split}_per_image_fixed_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result

def main():
    assert (HERE/'training_completed.json').exists(), 'Training must complete before evaluation.'
    best=RUN/'weights'/'best.pt'
    summary=dict(checkpoint=str(best),checkpoint_sha256=hashlib.sha256(best.read_bytes()).hexdigest())
    model=YOLO(str(best))
    summary['parameters']=sum(p.numel() for p in model.model.parameters())
    summary['gflops_at_960']=float(get_flops(model.model,imgsz=960))
    for split in ['val','test']:
        model=YOLO(str(best));shapes=set()
        def capture_shape(validator):shapes.add(tuple(validator.batch['img'].shape[1:]))
        # Capture actual loader shapes without modifying the validator.
        def capture_loader(validator):
            for batch in validator.dataloader:
                shapes.add(tuple(batch['img'].shape[1:]))
                break
        model.add_callback('on_val_start',capture_loader)
        metrics=model.val(data=str(HERE/'dataset.yaml'),split=split,imgsz=960,batch=16,
            device=0,workers=8,conf=0.001,iou=0.7,max_det=300,half=False,rect=True,
            augment=False,plots=True,save_json=True,project=str(HERE/'evaluation'),name=split,exist_ok=False)
        prediction_file=Path(metrics.save_dir)/'predictions.json'
        predictions=json.loads(prediction_file.read_text())
        entry=dict(ultralytics_metrics={k:float(v) for k,v in metrics.results_dict.items()},
            speed_ms_per_image={k:float(v) for k,v in metrics.speed.items()},
            observed_first_batch_tensor_shapes=sorted(shapes),
            per_class_ap={},fixed_confidence_025_iou_050=fixed_metrics(split,predictions))
        for i,c in enumerate(metrics.box.ap_class_index):
            entry['per_class_ap'][NAMES[int(c)]]=dict(ap50=float(metrics.box.ap50[i]),ap50_95=float(metrics.box.ap[i]))
        summary[split]=entry
        (HERE/'evaluation_metrics.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
