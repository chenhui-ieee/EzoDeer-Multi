# EzoDeer-Multi: Training and Evaluation

Reference training and evaluation code for two tasks:

- **Bounding-box detection with sex-category labels:** YOLO26n initialized with COCO-pretrained weights.
- **Semantic segmentation:** DeepLabV3+ with an ImageNet-pretrained ResNet-50 encoder.

This repository contains code only. Dataset images, annotation collections, original videos, pretrained model files, trained checkpoints, cached data and experiment outputs are not included.

## 1. Download the dataset

**EzoDeer-Multi is publicly available on Kaggle and Figshare. Download it independently from either platform using the dataset record associated with the paper.** Extract the archive before running this code. A Kaggle API key, Figshare token or other credential is not required by these scripts; downloading is performed separately by the user.

The extracted directory must have the following structure:

```text
EzoDeer-Multi/
├── README.md
├── metadata.csv
├── classes.json
├── train/
│   ├── images/*.jpg
│   ├── bbox/*.json
│   └── seg/*.json
├── val/
│   ├── images/*.jpg
│   ├── bbox/*.json
│   └── seg/*.json
└── test/
    ├── images/*.jpg
    ├── bbox/*.json
    └── seg/*.json
```

`--data-root` must point directly to this directory, not to the archive, its parent directory or a training-cache directory. The dataset can be placed anywhere; the code does not contain machine-specific paths.

| Subset | Images | Recording groups | Bounding boxes | Semantic category masks |
|---|---:|---:|---:|---:|
| Training | 1,407 | 150 | 3,460 | 1,882 |
| Validation | 214 | 36 | 647 | 328 |
| Test | 359 | 55 | 827 | 401 |
| Total | 1,980 | 241 | 4,934 | 2,611 |

Only image files are counted. `Thumbs.db`, if present, is a thumbnail cache and is ignored. The original recording-group split is retained; no new random image split is created.

## 2. Install dependencies

Python 3.10 or newer and an NVIDIA CUDA-capable GPU are required for the reference training recipes. Use a separate environment. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a mutually compatible CUDA-enabled PyTorch/torchvision pair appropriate for your GPU and driver using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The reference experiments used Python 3.10.20, PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128 on one NVIDIA RTX 5090 with 32 GB of memory.

Then install this package with the tasks you need:

```bash
# Both tasks:
python -m pip install -e '.[all]'

# Alternatively, install just one task:
# python -m pip install -e '.[detection]'
# python -m pip install -e '.[segmentation]'
```

`requirements.txt` provides an alternative list of application dependencies; install a compatible PyTorch/torchvision pair first. Verify GPU availability with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The supplied recipes use GPU 0. Batch sizes and epoch budgets reproduce the reference experiments and are defined in each task's `train.py`. Changes for another hardware configuration should be recorded alongside the resulting experiment.

## 3. Train and test bounding-box detection

Run the commands below from the repository root, replacing `/path/to/EzoDeer-Multi` with your extracted dataset location:

```bash
python train.py --task detection --data-root /path/to/EzoDeer-Multi --output outputs/bbox
python test.py --task detection --data-root /path/to/EzoDeer-Multi --output outputs/bbox
```

The training command first converts COCO bounding boxes into normalized YOLO labels and creates an aspect-preserving, lossless resized image cache. It then fine-tunes YOLO26n. `yolo26n.pt` is obtained through the official Ultralytics model-loading mechanism if not already available. To use a local initialization file:

```bash
python train.py --task detection --data-root /path/to/EzoDeer-Multi --output outputs/bbox --pretrained /path/to/yolo26n.pt
```

Reference settings: `imgsz=960`, batch size 16, AdamW, initial learning rate 0.001, weight decay 0.0005, three warm-up epochs, cosine learning-rate decay, seed 42, a maximum of 100 epochs and early-stopping patience 30. Augmentation includes mosaic, horizontal flipping, translation, scaling and HSV perturbations. Training inputs are square; evaluation uses aspect-preserving rectangular stride-padded batches. The complete settings are saved in `training_config.json` and `runs/.../args.yaml`.

The best checkpoint is selected using **validation mAP50-95**, not test performance. Testing evaluates the selected checkpoint on validation and test subsets without tuning it. Reported precision/recall/F1 are class-aware and micro-averaged at confidence 0.25 and IoU 0.50; AP metrics use a confidence floor of 0.001. Framework confidence-curve precision/recall summaries are retained separately and have a different definition. Prediction JSON boxes use resized-cache coordinates.

Key outputs:

```text
outputs/bbox/
├── training_config.json
├── platform.json
├── data_manifest.csv
├── data_summary.json
├── runs/yolo26n_960_seed42/
│   ├── weights/best.pt
│   ├── weights/last.pt
│   ├── args.yaml
│   └── results.csv
├── evaluation/{val,test}/
├── evaluation_metrics.json
├── overall_results.csv
├── test_per_class_results.csv
├── result_tables.md
└── Technical_Validation_EN.md
```

## 4. Train and test semantic segmentation

```bash
python train.py --task segmentation --data-root /path/to/EzoDeer-Multi --output outputs/semantic
python test.py --task segmentation --data-root /path/to/EzoDeer-Multi --output outputs/semantic
```

Preparation decodes the released COCO uncompressed RLE masks into mutually exclusive class maps. Background is 0, male is 1, female is 2 and unknown is 3. The released same-category masks are semantic unions, not animal-instance masks. Cross-category overlaps must already be resolved according to the released conventions: male > female > unknown. Preparation checks this and does not silently overwrite conflicting labels.

DeepLabV3+ uses a ResNet-50 encoder initialized with official PyTorch ImageNet-1K V1 weights. The default checkpoint is downloaded from the official PyTorch model host with its expected hash prefix verified. To provide a local initialization file, add `--pretrained /path/to/resnet50-0676ba61.pth` to the training command.

Images are resized to 960 × 540 using bilinear interpolation, masks use nearest-neighbor interpolation, and tensors are padded to 960 × 544. Padded label pixels use ignore value 255. The recipe uses batch size 8, AdamW, encoder/decoder learning rates 0.0001/0.001, weight decay 0.0001, one warm-up epoch, polynomial learning-rate decay, seed 42, at most 50 epochs and patience 12. The loss combines training-frequency-weighted cross-entropy and foreground soft Dice. Augmentation uses horizontal flipping and brightness/contrast perturbations.

The checkpoint is selected using **validation foreground mIoU at working resolution**. Final evaluation removes padding, bilinearly upsamples logits to each original image resolution, and applies argmax before comparison with native RLE reference masks. Main metrics are foreground mIoU and mean Dice over male/female/unknown. All-class mIoU includes background; pixel accuracy and binary deer IoU/Dice are secondary metrics. Predicted original-resolution PNG maps have pixel values 0–3. Neither task uses test-time augmentation.

Key outputs:

```text
outputs/semantic/
├── config.json
├── platform.json
├── data_manifest.csv
├── data_summary.json
├── history.csv
├── weights/best.pth
├── weights/last.pth
├── predictions/{val,test}/*.png
├── evaluation_metrics.json
├── overall_results.csv
├── test_per_class_results.csv
├── result_tables.md
├── test_prediction_examples.png
└── Technical_Validation_Semantic_EN.md
```

Additional result verification is available:

```bash
python -m ezodeer_training.segmentation.verify_results --data-root /path/to/EzoDeer-Multi --output outputs/semantic
```

## 5. Run individual stages

The convenience commands above can be replaced with explicit stages. Use the same dataset and output arguments for all stages:

```bash
python -m ezodeer_training.detection.prepare_data --data-root /path/to/EzoDeer-Multi --output outputs/bbox
python -m ezodeer_training.detection.train --data-root /path/to/EzoDeer-Multi --output outputs/bbox
python -m ezodeer_training.detection.evaluate --data-root /path/to/EzoDeer-Multi --output outputs/bbox
python -m ezodeer_training.detection.make_report --data-root /path/to/EzoDeer-Multi --output outputs/bbox
```

Replace `detection` with `segmentation` and use `outputs/semantic` for the segmentation workflow. Run `python train.py --help` or `python test.py --help` for wrapper options. Use `--skip-prepare` only when the prepared cache already exists. `--skip-report` computes evaluation metrics without producing tables/figures.

The `test.py` command expects a completed run in `--output`, including its selected model and preparation artifacts. It does not include trained models and is not a standalone test of an arbitrary unrelated checkpoint. Use a new output directory for a new training run; existing weights are protected from accidental overwrite. Never load an untrusted checkpoint.

## 6. Reference results and limitations

The following values are from the original completed experiments. Packaging these scripts did not rerun full training:

| Model | Test metric | Original result |
|---|---|---:|
| YOLO26n | mAP50 | 51.50% |
| YOLO26n | mAP50-95 | 37.23% |
| YOLO26n | Fixed-threshold micro F1 | 72.88% |
| DeepLabV3+ / ResNet-50 | Foreground mIoU | 44.04% |
| DeepLabV3+ / ResNet-50 | Foreground mean Dice | 53.59% |
| DeepLabV3+ / ResNet-50 | Binary deer IoU / Dice | 80.03% / 88.90% |

The original detector completed 77 epochs and selected epoch 47. Segmentation completed 22 epochs and selected epoch 10. Unknown-category performance was weak and its test support was limited. The experiments used one seed, and data originate from two fixed collection sites. Results do not establish generalization to unseen sites or independent animals. Hardware, dependencies and numerical kernels may affect reproduction. Images with empty annotation lists are retained according to released labels and have not been independently confirmed as deer-free.

## 7. Publication and dependencies

No passwords, API tokens, raw videos, dataset files, proprietary fonts or third-party source trees are intentionally bundled. `.gitignore` excludes data, videos, local environments, outputs and model files. A file inventory and credential/path pattern check can be run before uploading:

```bash
python scripts/audit_release.py --report ../training_code_audit.json
```

This repository does not download data from Kaggle/Figshare, access their APIs, or upload/publish anything. Dataset use is governed by its published Kaggle/Figshare release terms. Dependencies retain their upstream licenses; no code license or dataset license is selected on the authors' behalf. A content audit cannot establish legal ownership or authorization.
