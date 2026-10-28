"""Model, dataset, loss and metric definitions shared by training and evaluation."""
from pathlib import Path
import sys
from ..runtime import settings,platform_details
ARGS=settings('segmentation')
HERE=ARGS.output
import random
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp

CLASSES=['background','male','female','unknown']
MEAN=torch.tensor([.485,.456,.406])[:,None,None]
STD=torch.tensor([.229,.224,.225])[:,None,None]

def make_model(pretrained=False):
    model=smp.DeepLabV3Plus(encoder_name='resnet50',encoder_depth=5,encoder_weights=None,
        encoder_output_stride=16,decoder_channels=256,decoder_atrous_rates=(12,24,36),
        in_channels=3,classes=4,activation=None)
    if pretrained:
        weights=Path(ARGS.pretrained).expanduser() if ARGS.pretrained else HERE/'resnet50_imagenet.pth'
        if not weights.exists():
            if ARGS.pretrained:raise FileNotFoundError(weights)
            torch.hub.download_url_to_file('https://download.pytorch.org/models/resnet50-0676ba61.pth',str(weights),hash_prefix='0676ba61')
        model.encoder.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    return model

class DeerDataset(Dataset):
    def __init__(self,split,augment=False):
        self.split=split;self.augment=augment
        self.paths=sorted((HERE/'data'/'images'/split).glob('*.png'))
    def __len__(self):return len(self.paths)
    def __getitem__(self,index):
        path=self.paths[index]
        image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        mask=cv2.imread(str(HERE/'data'/'masks'/self.split/path.name),cv2.IMREAD_GRAYSCALE)
        if self.augment:
            if random.random()<.5:image=image[:,::-1].copy();mask=mask[:,::-1].copy()
            if random.random()<.8:
                # Mild photometric augmentation; no geometry or target-dependent crops.
                gain=random.uniform(.8,1.2);offset=random.uniform(-20,20)
                image=np.clip(image.astype(np.float32)*gain+offset,0,255).astype(np.uint8)
        image=torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1))).float()/255
        image=(image-MEAN)/STD
        mask=torch.from_numpy(mask.astype(np.int64))
        # 960x540 content is padded to 960x544 (divisible by encoder stride).
        return F.pad(image,(0,0,0,4),value=0),F.pad(mask,(0,0,0,4),value=255),path.stem

class SegmentationLoss(nn.Module):
    def __init__(self,weights):
        super().__init__();self.register_buffer('weights',torch.tensor(weights,dtype=torch.float32))
    def forward(self,logits,target):
        logits=logits.float()
        ce=F.cross_entropy(logits,target,weight=self.weights,ignore_index=255)
        valid=target!=255
        onehot=F.one_hot(target.masked_fill(~valid,0),4).permute(0,3,1,2).float()*valid[:,None]
        prob=logits.softmax(1)*valid[:,None]
        inter=(prob*onehot).sum((0,2,3));den=(prob+onehot).sum((0,2,3))
        present=onehot.sum((0,2,3))[1:]>0
        dice_loss=(1-(2*inter[1:]+1)/(den[1:]+1))[present].mean() if present.any() else logits.sum()*0
        return ce+dice_loss

def update_confusion(matrix,pred,target):
    valid=target!=255
    matrix+=torch.bincount(target[valid]*4+pred[valid],minlength=16).reshape(4,4)

def segmentation_metrics(confusion):
    a=np.asarray(confusion,dtype=np.float64);tp=np.diag(a);gt=a.sum(1);pred=a.sum(0)
    def div(x,y):return np.divide(x,y,out=np.zeros_like(x),where=y>0)
    iou=div(tp,gt+pred-tp);dice=div(2*tp,gt+pred);p=div(tp,pred);recall=div(tp,gt)
    btp=a[1:,1:].sum();bfp=a[0,1:].sum();bfn=a[1:,0].sum()
    return dict(confusion_matrix=a.astype(np.int64).tolist(),per_class={n:dict(iou=float(iou[c]),dice=float(dice[c]),
        precision=float(p[c]),recall=float(recall[c]),gt_pixels=int(gt[c]),pred_pixels=int(pred[c])) for c,n in enumerate(CLASSES)},
        foreground_miou=float(iou[1:].mean()),foreground_mean_dice=float(dice[1:].mean()),
        all_class_miou=float(iou.mean()),pixel_accuracy=float(tp.sum()/a.sum()),
        binary_deer_iou=float(btp/(btp+bfp+bfn)),binary_deer_dice=float(2*btp/(2*btp+bfp+bfn)))
