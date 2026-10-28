"""Convert released COCO boxes into YOLO labels without changing the source data."""
from pathlib import Path
import collections
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import cv2
import yaml

from ..runtime import settings,detection_run,platform_details
ARGS=settings('detection')
HERE=ARGS.output
SOURCE=ARGS.data_root
DEST = HERE / 'yolo_data'

def convert(item):
    split, path = item
    annotation = SOURCE / split / 'bbox' / (path.stem + '.json')
    obj = json.loads(annotation.read_text())
    image = cv2.imread(str(path))
    assert image is not None, path
    h, w = image.shape[:2]
    assert (w, h) == (obj['images'][0]['width'], obj['images'][0]['height'])
    lines, classes = [], collections.Counter()
    for box in obj['annotations']:
        x, y, bw, bh = box['bbox']
        assert bw > 0 and bh > 0
        # Conversion roundoff can place edges a fraction of a pixel outside the image.
        assert x >= -0.02 and y >= -0.02 and x + bw <= w + 0.02 and y + bh <= h + 0.02, (path, box)
        x1, y1, x2, y2 = max(0, x), max(0, y), min(w, x+bw), min(h, y+bh)
        category = box['category_id'] - 1
        assert category in [0, 1, 2]
        values = [(x1+x2)/2/w, (y1+y2)/2/h, (x2-x1)/w, (y2-y1)/h]
        lines.append(str(category) + ' ' + ' '.join(f'{v:.10f}' for v in values))
        classes[category] += 1
    # Lossless PNG cache after deterministic aspect-preserving resize. Original files remain untouched.
    scale = 960 / max(h, w)
    image = cv2.resize(image, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_LINEAR)
    output = DEST / 'images' / split / (path.stem + '.png')
    assert cv2.imwrite(str(output), image)
    label = DEST / 'labels' / split / (path.stem + '.txt')
    label.write_text('\n'.join(lines) + ('\n' if lines else ''))
    return dict(split=split, image=path.name, group=path.stem.rsplit('_',1)[0],
                width=w, height=h, boxes=len(lines), male=classes[0], female=classes[1], unknown=classes[2],
                image_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest())

def main():
    for split in ['train','val','test']:
        for kind in ['images','labels']:
            (DEST/kind/split).mkdir(parents=True, exist_ok=True)
    items = [(split,p) for split in ['train','val','test'] for p in sorted((SOURCE/split/'images').glob('*.jpg'))]
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(convert, items))
    with (HERE/'data_manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary = {}
    groups = {}
    for split in ['train','val','test']:
        selected=[r for r in rows if r['split']==split]
        groups[split]={r['group'] for r in selected}
        summary[split]=dict(images=len(selected),groups=len(groups[split]),
            boxes=sum(r['boxes'] for r in selected),empty_images=sum(r['boxes']==0 for r in selected),
            **{k:sum(r[k] for r in selected) for k in ['male','female','unknown']})
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    assert sum(r['boxes'] for r in rows)==4934 and len(rows)==1980
    (HERE/'data_summary.json').write_text(json.dumps(summary,indent=2))
    (HERE/'dataset.yaml').write_text(yaml.safe_dump(dict(path=str(DEST),train='images/train',val='images/val',test='images/test',names={0:'male',1:'female',2:'unknown'}),sort_keys=False))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
