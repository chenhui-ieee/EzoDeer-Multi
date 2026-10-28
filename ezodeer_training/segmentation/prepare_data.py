"""Build mutually exclusive semantic masks from current COCO RLE annotations."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import csv
import hashlib
import json
import cv2
import numpy as np
from pycocotools import mask as coco

from ..runtime import settings,platform_details
ARGS=settings('segmentation')
HERE=ARGS.output
SOURCE=ARGS.data_root
CACHE=HERE/'data'
CLASSES=['background','male','female','unknown']

def read_mask(d):
    im=d['images'][0];h,w=im['height'],im['width']
    label=np.zeros((h,w),dtype=np.uint8)
    for a in d['annotations']:
        rle=a['segmentation'];assert isinstance(rle,dict)
        binary=coco.decode(coco.frPyObjects(rle,h,w)).astype(bool)
        assert not np.any(label[binary]), 'Overlapping semantic labels.'
        assert int(binary.sum())==a['area']
        label[binary]=a['category_id']
    return label

def process(item):
    split,path=item;raw=path.read_bytes();d=json.loads(raw);im=d['images'][0]
    label=read_mask(d);counts=np.bincount(label.ravel(),minlength=4)
    image_path=SOURCE/split/'images'/im['file_name']
    image=cv2.imread(str(image_path));assert image.shape[:2]==label.shape
    resized=cv2.resize(image,(960,540),interpolation=cv2.INTER_LINEAR)
    small=cv2.resize(label,(960,540),interpolation=cv2.INTER_NEAREST)
    assert cv2.imwrite(str(CACHE/'images'/split/(path.stem+'.png')),resized)
    assert cv2.imwrite(str(CACHE/'masks'/split/(path.stem+'.png')),small)
    row=dict(split=split,image=im['file_name'],group=path.stem.rsplit('_',1)[0],width=im['width'],height=im['height'],
        regions=len(d['annotations']),seg_sha256=hashlib.sha256(raw).hexdigest(),
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        **{f'pixels_{name}':int(counts[c]) for c,name in enumerate(CLASSES)})
    return row,np.bincount(small.ravel(),minlength=4)

def main():
    for split in ['train','val','test']:
        for kind in ['images','masks']:(CACHE/kind/split).mkdir(parents=True,exist_ok=True)
    items=[(s,p) for s in ['train','val','test'] for p in sorted((SOURCE/s/'seg').glob('*.json'))]
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(process,items))
    rows=[x[0] for x in results];summary={};groups={}
    for split in ['train','val','test']:
        indices=[i for i,r in enumerate(rows) if r['split']==split];selected=[rows[i] for i in indices]
        groups[split]={r['group'] for r in selected}
        summary[split]=dict(images=len(selected),groups=len(groups[split]),regions=sum(r['regions'] for r in selected),
            empty_images=sum(sum(r[f'pixels_{n}'] for n in CLASSES[1:])==0 for r in selected),
            native_class_pixels=[sum(r[f'pixels_{n}'] for r in selected) for n in CLASSES],
            resized_class_pixels=np.sum([results[i][1] for i in indices],axis=0).tolist())
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    assert len(rows)==1980
    with (HERE/'data_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (HERE/'data_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
