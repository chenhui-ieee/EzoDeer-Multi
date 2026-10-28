"""Independently verify exported confusion matrices, native predictions and checkpoint."""
import csv
import hashlib
import json
import cv2
import numpy as np
import torch
from .common import HERE,make_model,segmentation_metrics
from .prepare_data import SOURCE,read_mask

def main():
    results=json.loads((HERE/'evaluation_metrics.json').read_text())
    summary=json.loads((HERE/'data_summary.json').read_text())
    history=list(csv.DictReader((HERE/'history.csv').open()))
    best=max(history,key=lambda r:float(r['val_foreground_miou']))
    assert int(best['epoch'])==results['selected_epoch']
    checkpoint=HERE/'weights'/'best.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==results['checkpoint_sha256']
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=make_model();model.load_state_dict(saved['state_dict'],strict=True)
    assert model.segmentation_head[0].out_channels==4
    for split in ['val','test']:
        rows=list(csv.DictReader((HERE/f'{split}_per_image_metrics.csv').open()))
        assert len(rows)==summary[split]['images']
        cm=sum(np.array([[int(r[f'cm_{a}_{b}']) for b in range(4)] for a in range(4)]) for r in rows)
        assert cm.tolist()==results[split]['native_resolution']['confusion_matrix']
        assert cm.sum(1).tolist()==summary[split]['native_class_pixels']
        assert len(list((HERE/'predictions'/split).glob('*.png')))==len(rows)
        recalculated=segmentation_metrics(cm)
        for metric in ['foreground_miou','foreground_mean_dice','pixel_accuracy','binary_deer_iou','binary_deer_dice']:
            assert abs(recalculated[metric]-results[split]['native_resolution'][metric])<1e-12
    # Read saved PNG predictions for all six published examples and reproduce each confusion matrix.
    selected=json.loads((HERE/'qualitative_example_selection.json').read_text())['images']
    by_name={r['image']:r for r in csv.DictReader((HERE/'test_per_image_metrics.csv').open())}
    for filename in selected:
        stem=filename.rsplit('.',1)[0]
        pred=cv2.imread(str(HERE/'predictions'/'test'/(stem+'.png')),cv2.IMREAD_GRAYSCALE)
        gt=read_mask(json.loads((SOURCE/'test'/'seg'/(stem+'.json')).read_text()))
        assert pred.shape==gt.shape and pred.min()>=0 and pred.max()<=3
        cm=np.bincount(gt.ravel().astype(np.int64)*4+pred.ravel(),minlength=16).reshape(4,4)
        row=by_name[filename]
        assert cm.tolist()==[[int(row[f'cm_{a}_{b}']) for b in range(4)] for a in range(4)]
    message='PASS: checkpoint strict load and SHA-256; validation-only selection; native class pixel totals; pooled per-image confusion matrices; metric formulas; all prediction file counts; six saved prediction PNGs reproduce reported native confusion matrices.'
    (HERE/'verification.txt').write_text(message+'\n');print(message)

if __name__=='__main__':main()
