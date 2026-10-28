"""Evaluate validation-selected DeepLabV3+ at working and original resolutions."""
from pathlib import Path
import csv
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .common import HERE,CLASSES,DeerDataset,make_model,segmentation_metrics,update_confusion
from .prepare_data import read_mask,SOURCE

@torch.inference_mode()
def main():
    assert (HERE/'training_completed.json').exists()
    torch.set_num_threads(8)
    checkpoint=HERE/'weights'/'best.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=make_model().cuda();model.load_state_dict(saved['state_dict'],strict=True);model.eval()
    result=dict(model='DeepLabV3+ / ResNet-50',checkpoint=str(checkpoint),selected_epoch=saved['epoch'],
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        parameters=sum(p.numel() for p in model.parameters()),
        native_resolution_protocol='bilinearly upsample 4-channel logits from 960x540 content to original HxW, then argmax; no TTA')
    manifest={(r['split'],Path(r['image']).stem):r for r in csv.DictReader((HERE/'data_manifest.csv').open())}
    for split in ['val','test']:
        loader=DataLoader(DeerDataset(split),batch_size=8,shuffle=False,num_workers=4,pin_memory=True)
        native_cm=torch.zeros((4,4),dtype=torch.int64,device='cuda');working_cm=torch.zeros_like(native_cm)
        folder=HERE/'predictions'/split;folder.mkdir(parents=True,exist_ok=True)
        per_image=[];inference_seconds=0;start=time.time()
        for batch_index,(image,target,stems) in enumerate(loader):
            image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
            torch.cuda.synchronize();t=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16):logits=model(image)
            torch.cuda.synchronize();inference_seconds+=time.perf_counter()-t
            update_confusion(working_cm,logits.argmax(1),target)
            for j,stem in enumerate(stems):
                path=SOURCE/split/'seg'/(stem+'.json');raw=path.read_bytes()
                assert hashlib.sha256(raw).hexdigest()==manifest[split,stem]['seg_sha256'], 'Labels changed since preparation.'
                annotation=json.loads(raw);original=read_mask(annotation);h,w=original.shape
                resized=F.interpolate(logits[j:j+1,:,:540,:].float(),size=(h,w),mode='bilinear',align_corners=False)
                pred=resized.argmax(1)[0];gt=torch.from_numpy(original.astype(np.int64)).cuda()
                cm=torch.zeros_like(native_cm);update_confusion(cm,pred,gt);native_cm+=cm
                native=segmentation_metrics(cm.cpu().numpy())
                array=pred.cpu().numpy().astype(np.uint8)
                assert cv2.imwrite(str(folder/(stem+'.png')),array)
                row=dict(image=stem+'.jpg',height=h,width=w,foreground_miou=native['foreground_miou'],
                    **{f'{n}_iou':native['per_class'][n]['iou'] for n in CLASSES})
                row.update({f'cm_{a}_{b}':int(native['confusion_matrix'][a][b]) for a in range(4) for b in range(4)})
                per_image.append(row)
            if (batch_index+1)%10==0:print(f'{split}: {len(per_image)}/{len(loader.dataset)} native images evaluated',flush=True)
        result[split]=dict(native_resolution=segmentation_metrics(native_cm.cpu().numpy()),
            working_resolution=segmentation_metrics(working_cm.cpu().numpy()),images=len(loader.dataset),
            total_elapsed_seconds=time.time()-start,
            batch8_forward_ms_per_image=1000*inference_seconds/len(loader.dataset))
        with (HERE/f'{split}_per_image_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(per_image[0]));writer.writeheader();writer.writerows(per_image)
        (HERE/'evaluation_metrics.json').write_text(json.dumps(result,indent=2))
        print(split,json.dumps(result[split]),flush=True)
    print('EVALUATION COMPLETE',flush=True)

if __name__=='__main__':main()
