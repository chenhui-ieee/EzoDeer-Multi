"""Convenience entry points for preparing, training and testing the reference recipes."""
import argparse
from pathlib import Path
import subprocess
import sys

def main(mode):
    parser=argparse.ArgumentParser(description=f'{mode.title()} EzoDeer-Multi reference models.')
    parser.add_argument('--task',choices=['detection','segmentation'],required=True)
    parser.add_argument('--data-root',type=Path,required=True,help='Extracted dataset root directly containing train/, val/ and test/.')
    parser.add_argument('--output',type=Path,help='Same experiment directory for training and testing; defaults to outputs/<task>.')
    parser.add_argument('--pretrained',help='Optional local COCO YOLO26n or ImageNet ResNet-50 checkpoint used for initialization.')
    if mode=='train':parser.add_argument('--skip-prepare',action='store_true',help='Use an existing prepared image/mask cache.')
    else:parser.add_argument('--skip-report',action='store_true',help='Compute metrics without generating result tables and figures.')
    args=parser.parse_args();output=(args.output or Path('outputs')/args.task).expanduser().resolve()
    data=args.data_root.expanduser().resolve()
    if not all((data/s).is_dir() for s in ['train','val','test']):parser.error('data-root must directly contain train/, val/ and test/.')
    if mode=='train':
        if (output/'training_completed.json').exists() or (output/'weights'/'best.pth').exists() or list(output.glob('runs/*/weights/best.pt')):
            parser.error('Training artifacts already exist; use a fresh output directory. Existing runs are preserved.')
        stages=([] if args.skip_prepare else ['prepare_data'])+['train']
    else:
        if not (output/'training_completed.json').exists():parser.error('No completed training run was found at --output. Train first or use the correct experiment directory.')
        stages=['evaluate']+([] if args.skip_report else ['make_report'])
    for stage in stages:
        command=[sys.executable,'-m',f'ezodeer_training.{args.task}.{stage}','--data-root',str(data),'--output',str(output)]
        if args.pretrained:command.extend(['--pretrained',args.pretrained])
        print(f'Running {args.task}/{stage}',flush=True)
        subprocess.run(command,check=True)

if __name__=='__main__':raise SystemExit('Use train.py or test.py from the repository root.')
