"""Write manuscript text and result tables from the completed experiment only."""
from pathlib import Path
import csv
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..runtime import settings,detection_run,platform_details
ARGS=settings('detection')
HERE=ARGS.output
RUN=detection_run(HERE)
def pct(x):return f'{100*x:.2f}'

def main():
    metrics=json.loads((HERE/'evaluation_metrics.json').read_text())
    platform=json.loads((HERE/'platform.json').read_text())
    data=json.loads((HERE/'data_summary.json').read_text())
    completed=json.loads((HERE/'training_completed.json').read_text())
    rows=list(csv.DictReader((RUN/'results.csv').open()))
    best=max(reversed(rows),key=lambda r:float(r['metrics/mAP50-95(B)']))
    duration=completed['elapsed_seconds']/60
    train=metrics['test'];val=metrics['val']
    all_test=train['fixed_confidence_025_iou_050']['all_micro']
    val_micro=val['fixed_confidence_025_iou_050']['all_micro']
    table=['| Subset | Images | Deer boxes | Precision (%) | Recall (%) | F1 (%) | mAP50 (%) | mAP50–95 (%) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    exported=[]
    for split,label in [('val','Validation'),('test','Test')]:
        entry=metrics[split];fixed=entry['fixed_confidence_025_iou_050']['all_micro'];ap=entry['ultralytics_metrics']
        values=[label,data[split]['images'],data[split]['boxes'],pct(fixed['precision']),pct(fixed['recall']),pct(fixed['f1']),pct(ap['metrics/mAP50(B)']),pct(ap['metrics/mAP50-95(B)'])]
        table.append('| '+' | '.join(map(str,values))+' |')
        exported.append(dict(zip(['subset','images','boxes','precision_percent','recall_percent','f1_percent','map50_percent','map50_95_percent'],values)))
    with (HERE/'overall_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(exported[0]));w.writeheader();w.writerows(exported)
    per_class=['| Class | Test instances | TP | FP | FN | Precision (%) | Recall (%) | F1 (%) | AP50 (%) | AP50–95 (%) |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    class_rows=[]
    for name in ['male','female','unknown']:
        fixed=train['fixed_confidence_025_iou_050'][name];ap=train['per_class_ap'][name]
        values=[name,data['test'][name],fixed['tp'],fixed['fp'],fixed['fn'],pct(fixed['precision']),pct(fixed['recall']),pct(fixed['f1']),pct(ap['ap50']),pct(ap['ap50_95'])]
        per_class.append('| '+' | '.join(map(str,values))+' |')
        class_rows.append(dict(zip(['class','instances','tp','fp','fn','precision_percent','recall_percent','f1_percent','ap50_percent','ap50_95_percent'],values)))
    with (HERE/'test_per_class_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)
    (HERE/'result_tables.md').write_text('\n'.join(table)+'\n\n'+'\n'.join(per_class)+'\n')
    c,h,w=train['observed_first_batch_tensor_shapes'][0]
    shape=f'{h} × {w} pixels (height × width), including padding'
    report=f'''# Technical Validation

## Experimental Platforms

Experiments were conducted on a workstation running Ubuntu 24.04.5 LTS with an AMD Ryzen 9 9950X processor (16 cores and 32 threads), approximately 64 GB of installed system memory, and one NVIDIA GeForce RTX 5090 GPU with 32 GB of graphics memory. The software environment comprised Python {platform['python']}, PyTorch {platform['torch']}, CUDA runtime {platform['cuda_runtime']}, and Ultralytics {platform['ultralytics']}. The NVIDIA driver version was 580.178.04. All training and evaluation were performed on the same GPU.

## Implementation Details

A YOLO26n object detector initialized from COCO-pretrained weights was fine-tuned as a baseline for deer localization and sex-category prediction. The detector used three classes: male, female, and unknown. The released category identifiers 1, 2, and 3 were mapped to YOLO class indices 0, 1, and 2, respectively. This experiment evaluated bounding-box annotations only; segmentation annotations were not used, and no segmentation-performance claim is made.

The existing recording-group split was retained without reassignment. The training, validation, and test subsets contained 1,407, 214, and 359 images and 3,460, 647, and 827 deer bounding boxes, respectively. No filename-based recording-group identifier was shared between subsets. The 12, 1, and 21 images with empty bounding-box annotations in the respective subsets were retained as negative training or evaluation samples according to the released labels. Images were not manually relabeled for this experiment.

Bounding boxes were converted from pixel-based [x, y, width, height] coordinates to normalized YOLO coordinates. For efficient loading, source images were resized with bilinear interpolation to a longest side of 960 pixels and stored in a lossless PNG cache. All released images had a 16:9 aspect ratio, yielding cached images of 960 × 540 pixels. Original images and annotations were preserved. Training used an image-size setting of 960 with square 960 × 960 augmented inputs. Validation and testing used aspect-preserving rectangular batches with stride-aligned padding and the same 960-pixel longest-side setting; the observed first test-batch tensor shape, excluding batch size, was {shape}. No test-time augmentation was applied.

Training used a batch size of 16, the AdamW optimizer, an initial learning rate of 0.001, beta1 of 0.9, and weight decay of 0.0005. A cosine learning-rate schedule was configured to reach 1% of the initial learning rate at the end of the 100-epoch budget, with a three-epoch warm-up. Augmentations included mosaic augmentation (probability 1.0), horizontal flipping (probability 0.5), translation (0.1), scale variation (0.5), and HSV color augmentation (hue 0.015, saturation 0.7, and value 0.4). Rotation, shear, perspective transformation, vertical flipping, MixUp, and copy-paste augmentation were disabled. Mosaic augmentation was scheduled to be disabled during the last 10 epochs of the budget; this phase was not reached because early stopping occurred before epoch 91. Automatic mixed precision was enabled, eight data-loader workers were used, and the random seed was fixed at 42 with deterministic execution requested.

The maximum training budget was 100 epochs, with early stopping after 30 consecutive epochs without improvement in validation mAP50–95. Training completed {len(rows)} epochs in approximately {duration:.1f} minutes, including initialization and the framework's final validation. The checkpoint with the highest validation mAP50–95 was selected (epoch {best['epoch']}) and then evaluated on the held-out test subset. No hyperparameter or checkpoint selection was performed using test results. This was one training run with one random seed; variation across repeated training runs was not estimated.

## Evaluation Metrics

Detection accuracy was evaluated using average precision at an intersection-over-union (IoU) threshold of 0.50 (AP50) and average precision averaged over ten IoU thresholds from 0.50 to 0.95 in increments of 0.05 (AP50–95). The corresponding mAP values are unweighted means across the three classes. AP was computed using the Ultralytics evaluator with a prediction-confidence floor of 0.001 and a maximum of 300 detections per image.

Precision, recall, and F1 were additionally computed at a prespecified confidence threshold of 0.25 and an IoU threshold of 0.50. Predictions were processed in descending confidence order and matched one-to-one to unmatched ground-truth boxes of the same class with the greatest IoU. A match was accepted when IoU was at least 0.50. Unmatched predictions and ground-truth annotations were counted as false positives and false negatives, respectively. Precision, recall, and F1 were calculated as TP/(TP+FP), TP/(TP+FN), and 2TP/(2TP+FP+FN). Overall precision, recall, and F1 were micro-averaged by pooling counts across classes and images; per-class metrics were also reported. A wrong sex-category prediction therefore does not count as a correct detection, even if its box overlaps a deer.

The fixed-threshold precision and recall reported here differ in definition from the framework's default confidence-curve summaries, which select an operating point from the evaluated data. The same prespecified confidence threshold was used for validation and test reporting. Prediction outputs, raw evaluator metrics, per-image TP/FP/FN counts, and the exact software and training configurations were retained to support reproducibility.

## Experimental Results

The validation-selected YOLO26n checkpoint achieved a test mAP50 of {pct(train['ultralytics_metrics']['metrics/mAP50(B)'])}% and a test mAP50–95 of {pct(train['ultralytics_metrics']['metrics/mAP50-95(B)'])}%. At confidence 0.25 and IoU 0.50, micro-averaged test precision, recall, and F1 were {pct(all_test['precision'])}%, {pct(all_test['recall'])}%, and {pct(all_test['f1'])}%, respectively. These corresponded to {all_test['tp']} true positives, {all_test['fp']} false positives, and {all_test['fn']} false negatives.

'''+'\n'.join(table)+f'''

Per-class test results are shown below. AP50 was {pct(train['per_class_ap']['male']['ap50'])}% for male, {pct(train['per_class_ap']['female']['ap50'])}% for female, and {pct(train['per_class_ap']['unknown']['ap50'])}% for unknown.

'''+'\n'.join(per_class)+'''

The female class achieved the strongest test AP, whereas performance on the unknown class was very poor: no unknown annotations were correctly detected at the fixed confidence threshold of 0.25. The model therefore does not provide reliable three-category recognition under this split. The difference between validation and test performance also indicates limited generalization beyond the validation recording groups.

These results provide a reproducible baseline for joint bounding-box localization and sex-category prediction under the existing dataset split. They do not measure sex classification in isolation or segmentation quality. The test set contains only 13 unknown annotations, compared with 469 male and 345 female annotations, so estimates for the unknown class are based on limited support. Class proportions also differ between subsets. Furthermore, adjacent frames from the same recording are correlated, and all data originate from two fixed collection locations; these results should not be interpreted as evidence of generalization to unseen locations or independent individual animals. Performance is evaluated against the released annotations, including images with empty annotation lists that have not been independently adjudicated as deer-free.

Model reference: [Ultralytics YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/). Evaluation implementation: [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/).
'''
    (HERE/'Technical_Validation_EN.md').write_text(report)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'savefig.dpi':200})
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for column,label in [('metrics/mAP50(B)','mAP50'),('metrics/mAP50-95(B)','mAP50–95')]:
        axs[0].plot([int(r['epoch']) for r in rows],[100*float(r[column]) for r in rows],label=label)
    axs[0].axvline(int(best['epoch']),color='gray',linestyle='--',label='Selected epoch')
    axs[0].set(xlabel='Epoch',ylabel='Validation mAP (%)',title='Validation accuracy');axs[0].legend()
    for column,label in [('train/box_loss','Box loss'),('train/cls_loss','Classification loss')]:
        axs[1].plot([int(r['epoch']) for r in rows],[float(r[column]) for r in rows],label=label)
    axs[1].set(xlabel='Epoch',ylabel='Loss',title='Training losses');axs[1].legend()
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(HERE/f'training_curves.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));x=np.arange(3);width=.35
    for offset,key,label,color in [(-width/2,'ap50','AP50','#3277AD'),(width/2,'ap50_95','AP50–95','#D87551')]:
        values=[100*train['per_class_ap'][n][key] for n in ['male','female','unknown']]
        rects=ax.bar(x+offset,values,width,label=label,color=color)
        ax.bar_label(rects,fmt='%.2f',padding=3)
    ax.set(xticks=x,xticklabels=['male (n=469)','female (n=345)','unknown (n=13)'],ylabel='AP (%)',ylim=(0,108),title='YOLO26n test performance (960-pixel input)');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(HERE/f'test_per_class_AP.{ext}',bbox_inches='tight')
    plt.close(fig)
    print(report)

if __name__=='__main__':main()
