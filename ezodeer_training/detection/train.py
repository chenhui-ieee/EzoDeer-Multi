"""Fixed single-run detection baseline; test set is not used for model selection."""
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO

from ..runtime import settings,detection_run,platform_details
ARGS=settings('detection')
HERE=ARGS.output
def main():
    torch.set_num_threads(8)
    if not torch.cuda.is_available():raise RuntimeError('The reference training recipe requires a CUDA-capable GPU.')
    if (HERE/'training_completed.json').exists() or (detection_run(HERE)/'weights'/'best.pt').exists():raise FileExistsError('Use a fresh output directory; previous training artifacts are preserved.')
    initial=ARGS.pretrained or 'yolo26n.pt'
    platform_info=dict(os=platform.platform(),python=platform.python_version(),torch=torch.__version__,
        cuda_runtime=torch.version.cuda,ultralytics=ultralytics.__version__,gpu=torch.cuda.get_device_name(0),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,cpu_threads=os.cpu_count(),
        cpu=platform_details()['cpu'],driver=platform_details()['driver'])
    (HERE/'platform.json').write_text(json.dumps(platform_info,indent=2))
    config=dict(data=str(HERE/'dataset.yaml'),epochs=100,patience=30,imgsz=960,batch=16,
        device=0,workers=8,optimizer='AdamW',lr0=0.001,lrf=0.01,momentum=0.9,weight_decay=0.0005,
        warmup_epochs=3.0,seed=42,deterministic=True,amp=True,cache=False,rect=False,
        cos_lr=True,close_mosaic=10,mosaic=1.0,mixup=0.0,copy_paste=0.0,
        hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,degrees=0.0,translate=0.1,scale=0.5,
        shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.5,multi_scale=False,
        project=str(HERE/'runs'),name='yolo26n_960_seed42',exist_ok=False,
        save=True,save_period=10,plots=True,val=True)
    (HERE/'training_config.json').write_text(json.dumps(config,indent=2))
    start=time.time()
    YOLO(str(initial)).train(**config)
    (HERE/'training_completed.json').write_text(json.dumps(dict(elapsed_seconds=time.time()-start),indent=2))
    print('TRAINING COMPLETED',flush=True)

if __name__=='__main__':main()
