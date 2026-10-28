"""Write manuscript text, result tables and figures from measured results."""
from pathlib import Path
import csv
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .common import HERE,CLASSES
from .prepare_data import SOURCE,read_mask

COLORS=np.array([[0,0,0],[50,119,173],[216,117,81],[139,132,152]],dtype=np.uint8)
def percent(x):return f'{100*x:.2f}'

def main():
    results=json.loads((HERE/'evaluation_metrics.json').read_text());config=json.loads((HERE/'config.json').read_text())
    platform=json.loads((HERE/'platform.json').read_text());completed=json.loads((HERE/'training_completed.json').read_text())
    summary=json.loads((HERE/'data_summary.json').read_text());history=list(csv.DictReader((HERE/'history.csv').open()))
    test=results['test']['native_resolution'];val=results['val']['native_resolution']
    lines=['| Subset | Images | Foreground mIoU (%) | Foreground mean Dice (%) | All-class mIoU (%) | Pixel accuracy (%) | Binary deer IoU (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    overall=[]
    for split in ['val','test']:
        e=results[split]['native_resolution'];row=[split,results[split]['images'],*[percent(e[k]) for k in ['foreground_miou','foreground_mean_dice','all_class_miou','pixel_accuracy','binary_deer_iou']]]
        lines.append('| '+' | '.join(map(str,row))+' |');overall.append(dict(zip(['subset','images','foreground_miou_percent','foreground_mean_dice_percent','all_class_miou_percent','pixel_accuracy_percent','binary_deer_iou_percent'],row)))
    cls_lines=['| Class | IoU (%) | Dice (%) | Pixel precision (%) | Pixel recall (%) | Ground-truth pixels |',
        '|---|---:|---:|---:|---:|---:|'];class_rows=[]
    for name in CLASSES:
        e=test['per_class'][name];row=[name,*[percent(e[k]) for k in ['iou','dice','precision','recall']],e['gt_pixels']]
        cls_lines.append('| '+' | '.join(map(str,row))+' |');class_rows.append(dict(zip(['class','iou_percent','dice_percent','precision_percent','recall_percent','gt_pixels'],row)))
    for name,rows in [('overall_results',overall),('test_per_class_results',class_rows)]:
        with (HERE/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    tables='\n'.join(lines)+'\n\n'+'\n'.join(cls_lines)+'\n';(HERE/'result_tables.md').write_text(tables)
    report=f'''# Technical Validation: Semantic Segmentation

## Experimental Platforms

Experiments were performed on a workstation with an AMD Ryzen 9 9950X processor, approximately 64 GB of installed RAM, and one NVIDIA GeForce RTX 5090 GPU with 32 GB of memory. The software environment used Python {platform['python']}, PyTorch {platform['torch']}, CUDA runtime {platform['cuda']}, and segmentation-models-pytorch {platform['smp']}. The detection and segmentation baselines were executed as separate experiments.

## Implementation Details

DeepLabV3+ with a ResNet-50 encoder was used as a semantic segmentation baseline. The encoder was initialized with the official PyTorch ImageNet-1K V1 weights, while the decoder and four-class segmentation head were randomly initialized. The encoder output stride was 16, the decoder had 256 channels, and the ASPP atrous rates were 12, 24, and 36. The network contained {results['parameters']:,} parameters. The four classes were background (0), male (1), female (2), and unknown (3).

The existing recording-group split was retained: 1,407 training images, 214 validation images, and 359 test images. Current COCO uncompressed RLE annotations were decoded at their original image resolutions. Same-category regions had already been unioned, and cross-category overlaps had been resolved using the priority male > female > unknown. All decoded masks were checked for overlap before conversion into single-channel class maps. The 12, 1, and 21 images with empty masks in the respective subsets were retained according to their released labels; they have not been independently verified as truly deer-free. The original images and annotations were preserved, and SHA-256 checksums were recorded for data provenance.

Images were resized by bilinear interpolation to 960 × 540 pixels, preserving their 16:9 aspect ratio. Training label maps were resized using nearest-neighbor interpolation. Image tensors were normalized with ImageNet RGB means (0.485, 0.456, 0.406) and standard deviations (0.229, 0.224, 0.225), then padded by four rows to 960 × 544 pixels for encoder-stride compatibility. Padded label pixels were assigned an ignore index of 255. Augmentation comprised horizontal flipping with probability 0.5 and brightness/contrast perturbations with probability 0.8, using gains sampled from 0.8–1.2 and offsets from −20 to 20. No label-dependent crops or test-time augmentation were used.

Training used AdamW with a batch size of 8, an encoder learning rate of 0.0001, a decoder/head learning rate of 0.001, and weight decay of 0.0001. A one-epoch linear warm-up was followed by polynomial learning-rate decay with power 0.9, scheduled toward 1% of the base rate over a maximum of 50 epochs. The loss was the equally weighted sum of class-weighted cross-entropy and foreground soft Dice loss. Cross-entropy weights were derived solely from training pixel frequencies using inverse-square-root scaling and clipping to [0.25, 4.0]; weights in background/male/female/unknown order were {', '.join(f'{w:.4f}' for w in config['cross_entropy_class_weights'])}. Dice loss was averaged across foreground classes present in each minibatch. Automatic mixed precision and gradient-norm clipping at 1.0 were used. The random seed was fixed at 42, and deterministic algorithms were requested with warnings enabled for unsupported operations.

The checkpoint with the highest validation foreground mIoU at the working resolution was selected. Early stopping was configured after 12 epochs without improvement. Training completed {completed['epochs']} epochs in {completed['elapsed_seconds']/60:.1f} minutes, and epoch {results['selected_epoch']} was selected. No test-set results were used to select the model or hyperparameters. This experiment comprised one training run; variation across independent seeds was not estimated.

## Evaluation Metrics

Final validation and test metrics were computed at each image's original released resolution. The four-row input padding was removed from the predicted logits, which were bilinearly upsampled to the original image dimensions before pixel-wise argmax. The reference masks were decoded directly from the original-resolution RLE. Working-resolution metrics were also retained in the machine-readable results, but all values in the tables below refer to original-resolution evaluation.

A global pixel confusion matrix was accumulated across each subset, with rows denoting reference classes and columns predicted classes. Per-class intersection over union (IoU) and Dice were calculated as TP/(TP+FP+FN) and 2TP/(2TP+FP+FN), respectively. Pixel precision and recall were TP/(TP+FP) and TP/(TP+FN). The primary summary measures were foreground mIoU and foreground mean Dice, computed as unweighted means over male, female, and unknown. All-class mIoU additionally included background. Overall pixel accuracy was the fraction of correctly classified pixels. Because background occupies most image pixels, pixel accuracy alone was not considered sufficient evidence of segmentation quality.

Binary deer IoU and Dice were additionally computed by merging the three foreground classes, assessing deer-region segmentation independently of sex-category errors. These binary metrics were derived from the same predictions without retraining or threshold tuning. Metrics were computed from pooled pixel counts rather than averaging per-image ratios.

## Experimental Results

On the held-out test set, the validation-selected model achieved a foreground mIoU of {percent(test['foreground_miou'])}% and a foreground mean Dice of {percent(test['foreground_mean_dice'])}%. All-class mIoU and pixel accuracy were {percent(test['all_class_miou'])}% and {percent(test['pixel_accuracy'])}%, respectively. After merging the foreground classes, binary deer IoU was {percent(test['binary_deer_iou'])}% and binary deer Dice was {percent(test['binary_deer_dice'])}%.

'''+tables+f'''
The class-specific test IoUs were {percent(test['per_class']['male']['iou'])}% for male, {percent(test['per_class']['female']['iou'])}% for female, and {percent(test['per_class']['unknown']['iou'])}% for unknown. These results characterize this single baseline and dataset split and should not be interpreted as demonstrating reliable recognition of all categories. Unknown has only nine image–category mask regions in the test set, and class proportions differ between subsets. Adjacent frames remain correlated within recordings, and both subsets originate from the same two collection locations. The group-based split therefore does not establish generalization to unseen sites or independent individual animals. Resizing may also remove very small annotated regions during training, although final scores use original-resolution reference masks. No instance-segmentation performance is claimed because the labels represent semantic category unions.

Model reference: [DeepLabV3+ paper](https://arxiv.org/abs/1802.02611). Implementation reference: [segmentation-models-pytorch documentation](https://smp.readthedocs.io/en/stable/models.html#deeplabv3).
'''
    (HERE/'Technical_Validation_Semantic_EN.md').write_text(report)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'savefig.dpi':200})
    fig,axs=plt.subplots(1,2,figsize=(11,4));epochs=[int(r['epoch']) for r in history]
    for key,label in [('val_foreground_miou','Foreground mIoU'),('val_foreground_mean_dice','Foreground mean Dice')]:axs[0].plot(epochs,[100*float(r[key]) for r in history],label=label)
    axs[0].axvline(results['selected_epoch'],color='gray',linestyle='--');axs[0].set(xlabel='Epoch',ylabel='Validation score (%)',title='Working-resolution validation');axs[0].legend()
    for key,label in [('train_loss','Train'),('val_loss','Validation')]:axs[1].plot(epochs,[float(r[key]) for r in history],label=label)
    axs[1].set(xlabel='Epoch',ylabel='CE + Dice loss',title='Loss curves');axs[1].legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(HERE/f'training_curves.{ext}',bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    for offset,key,label in [(-.18,'iou','IoU'),(.18,'dice','Dice')]:
        vals=[100*test['per_class'][n][key] for n in CLASSES[1:]];bars=ax.bar(x+offset,vals,.36,label=label)
        ax.bar_label(bars,fmt='%.2f',padding=3)
    ax.set(xticks=x,xticklabels=CLASSES[1:],ylabel='Score (%)',ylim=(0,110),title='Original-resolution test segmentation');ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(HERE/f'test_per_class_scores.{ext}',bbox_inches='tight')
    plt.close(fig)
    qualitative(summary)
    print(report)

def qualitative(summary):
    manifest=list(csv.DictReader((HERE/'data_manifest.csv').open()));selected=[]
    for name in CLASSES[1:]:
        rows=sorted([r for r in manifest if r['split']=='test' and int(r[f'pixels_{name}'])>0],key=lambda r:int(r[f'pixels_{name}']))
        for q in [.5,.75]:
            for r in rows[round((len(rows)-1)*q):]+rows:
                if r['image'] not in [v['image'] for v in selected]:selected.append(r);break
    fig,axs=plt.subplots(len(selected),3,figsize=(16,3.3*len(selected)))
    for i,row in enumerate(selected):
        path=SOURCE/'test'/'images'/row['image'];stem=path.stem
        rgb=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);rgb=cv2.resize(rgb,(960,540))
        gt=read_mask(json.loads((SOURCE/'test'/'seg'/(stem+'.json')).read_text()))
        pred=cv2.imread(str(HERE/'predictions'/'test'/(stem+'.png')),cv2.IMREAD_GRAYSCALE)
        axs[i,0].imshow(rgb)
        for ax,mask in [(axs[i,1],gt),(axs[i,2],pred)]:
            small=cv2.resize(mask,(960,540),interpolation=cv2.INTER_NEAREST)
            overlay=rgb.copy().astype(float);fg=small>0;overlay[fg]=.5*overlay[fg]+.5*COLORS[small[fg]]
            ax.imshow(overlay.astype(np.uint8))
        for j,ax in enumerate(axs[i]):ax.axis('off');ax.set_title(['Original image','Ground-truth overlay','Predicted overlay'][j]+'\n'+row['image'],fontsize=9)
    fig.legend(handles=[Patch(facecolor=COLORS[c]/255,label=CLASSES[c]) for c in [1,2,3]],loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.025,1,1))
    for ext in ['png','pdf']:fig.savefig(HERE/f'test_prediction_examples.{ext}',bbox_inches='tight')
    plt.close(fig)
    (HERE/'qualitative_example_selection.json').write_text(json.dumps(dict(rule='Two distinct test images per class near the 50th and 75th percentiles of ground-truth mask area; no selection by model score.',images=[r['image'] for r in selected]),indent=2))

if __name__=='__main__':main()
