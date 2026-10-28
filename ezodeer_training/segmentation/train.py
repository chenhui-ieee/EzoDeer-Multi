"""Train a single-seed DeepLabV3+ baseline; select only on validation foreground mIoU."""
import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .common import ARGS,HERE,DeerDataset,SegmentationLoss,make_model,segmentation_metrics,update_confusion,smp

def worker_init(worker_id):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)
    import cv2
    cv2.setNumThreads(0)

@torch.inference_mode()
def validate(model,loader,criterion):
    model.eval();cm=torch.zeros((4,4),dtype=torch.int64,device='cuda');loss=0
    for image,target,_ in loader:
        image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
        with torch.autocast('cuda',dtype=torch.float16):output=model(image)
        loss+=criterion(output,target).item()*image.size(0)
        update_confusion(cm,output.argmax(1),target)
    return loss/len(loader.dataset),segmentation_metrics(cm.cpu().numpy())

def main():
    if not torch.cuda.is_available():raise RuntimeError('The reference training recipe requires a CUDA-capable GPU.')
    if (HERE/'training_completed.json').exists() or (HERE/'weights'/'best.pth').exists():raise FileExistsError('Use a fresh output directory; previous training artifacts are preserved.')
    downloaded_model=make_model(True)
    del downloaded_model
    torch.set_num_threads(8);seed=42;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    config=dict(model='DeepLabV3+',encoder='resnet50',pretrained='ImageNet ResNet50 IMAGENET1K_V1',
        image_size=[960,540],padded_tensor_size=[960,544],batch_size=8,max_epochs=50,patience=12,
        optimizer='AdamW',encoder_lr=1e-4,decoder_lr=1e-3,weight_decay=1e-4,
        schedule='one-epoch warmup, polynomial decay power 0.9 to 1% at epoch 50',seed=seed,
        amp=True,num_workers=8,loss='train-frequency-weighted cross entropy + foreground soft Dice (equal weighting)',
        augmentation='horizontal flip p=0.5; brightness/contrast gain [0.8,1.2], offset [-20,20] with p=0.8',
        selection='highest validation foreground mIoU at cached working resolution',ignore_index=255)
    summary=json.loads((HERE/'data_summary.json').read_text());freq=np.asarray(summary['train']['resized_class_pixels'],dtype=float)
    frequency=freq/freq.sum();weights=np.clip(np.sqrt(np.median(frequency[1:])/frequency),.25,4.0).tolist()
    config['cross_entropy_class_weights']=weights
    (HERE/'config.json').write_text(json.dumps(config,indent=2))
    (HERE/'platform.json').write_text(json.dumps(dict(os=platform.platform(),python=platform.python_version(),
        torch=torch.__version__,cuda=torch.version.cuda,smp=smp.__version__,gpu=torch.cuda.get_device_name(0),
        cpu=__import__('ezodeer_training.runtime',fromlist=['platform_details']).platform_details()['cpu'],gpu_details=__import__('ezodeer_training.runtime',fromlist=['platform_details']).platform_details()['driver'],
        initial_weights_sha256=hashlib.sha256((__import__('pathlib').Path(ARGS.pretrained).expanduser() if ARGS.pretrained else HERE/'resnet50_imagenet.pth').read_bytes()).hexdigest()),indent=2))
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(DeerDataset('train',True),batch_size=8,shuffle=True,num_workers=8,pin_memory=True,
        persistent_workers=True,worker_init_fn=worker_init,generator=generator)
    val_loader=DataLoader(DeerDataset('val'),batch_size=8,shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True,worker_init_fn=worker_init)
    model=make_model(True).cuda();criterion=SegmentationLoss(weights).cuda()
    encoder=list(model.encoder.parameters());other=list(model.decoder.parameters())+list(model.segmentation_head.parameters())
    optimizer=torch.optim.AdamW([dict(params=encoder,lr=1e-4),dict(params=other,lr=1e-3)],weight_decay=1e-4)
    total_steps=50*len(train_loader);warmup=len(train_loader)
    def multiplier(step):
        if step<warmup:return .1+.9*step/max(1,warmup)
        progress=(step-warmup)/max(1,total_steps-warmup)
        return .01+.99*max(0,1-progress)**.9
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)
    scaler=torch.amp.GradScaler('cuda');best=-1;bad=0;start=time.time();(HERE/'weights').mkdir(exist_ok=True)
    print('CONFIG',json.dumps(config),flush=True)
    print('Parameters',sum(p.numel() for p in model.parameters()),flush=True)
    with (HERE/'history.csv').open('w',newline='',buffering=1) as f:
        fields=['epoch','elapsed_seconds','train_loss','val_loss','val_foreground_miou','val_foreground_mean_dice','val_all_class_miou','encoder_lr','decoder_lr']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for epoch in range(1,51):
            model.train();train_loss=0
            for batch_index,(image,target,_) in enumerate(train_loader):
                image=image.cuda(non_blocking=True);target=target.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):output=model(image);loss=criterion(output,target)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(optimizer);scaler.update();scheduler.step()
                train_loss+=loss.item()*image.size(0)
                if (batch_index+1)%50==0:print(f'epoch {epoch} batch {batch_index+1}/{len(train_loader)} loss {loss.item():.4f}',flush=True)
            val_loss,metrics=validate(model,val_loader,criterion)
            row=dict(epoch=epoch,elapsed_seconds=time.time()-start,train_loss=train_loss/len(train_loader.dataset),val_loss=val_loss,
                val_foreground_miou=metrics['foreground_miou'],val_foreground_mean_dice=metrics['foreground_mean_dice'],
                val_all_class_miou=metrics['all_class_miou'],encoder_lr=optimizer.param_groups[0]['lr'],decoder_lr=optimizer.param_groups[1]['lr'])
            writer.writerow(row);print(json.dumps(row),flush=True)
            improved=metrics['foreground_miou']>best
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            checkpoint=dict(epoch=epoch,state_dict=state,config=config,validation_metrics=metrics)
            if improved:
                best=metrics['foreground_miou'];bad=0;torch.save(checkpoint,HERE/'weights'/'best.pth')
            else:bad+=1
            # Last checkpoint includes optimizer/scaler state for recovery if needed.
            torch.save(dict(checkpoint,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict()),HERE/'weights'/'last.pth')
            if bad>=12:
                print(f'Early stopping: {bad} epochs without validation foreground mIoU improvement.',flush=True);break
    (HERE/'training_completed.json').write_text(json.dumps(dict(epochs=epoch,elapsed_seconds=time.time()-start,best_validation_foreground_miou=best),indent=2))
    print('TRAINING COMPLETE',flush=True)

if __name__=='__main__':main()
