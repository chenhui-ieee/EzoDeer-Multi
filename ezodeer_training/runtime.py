"""Portable command-line paths; no user-specific paths or implicit dataset discovery."""
import argparse
from pathlib import Path
import os

_CACHE={}

def settings(kind):
    if kind in _CACHE:return _CACHE[kind]
    parser=argparse.ArgumentParser(description=f'EzoDeer-Multi {kind} utilities. See the repository README for stage commands.')
    parser.add_argument('--data-root',type=Path,default=os.environ.get('EZODEER_DATA_ROOT'),
        help='Dataset directory directly containing train/, val/, test/, metadata.csv and classes.json.')
    parser.add_argument('--output',type=Path,default=Path('outputs')/kind,help='Experiment/output directory; use the same directory across stages.')
    parser.add_argument('--pretrained',help='Optional local pretrained weight file. Detection defaults to the official yolo26n.pt download.')
    args=parser.parse_args()
    if args.data_root is None:parser.error('--data-root is required (or set EZODEER_DATA_ROOT).')
    args.data_root=Path(args.data_root).expanduser().resolve()
    if not all((args.data_root/s).is_dir() for s in ['train','val','test']):parser.error('data-root must contain train/, val/ and test/.')
    args.output=args.output.expanduser().resolve();args.output.mkdir(parents=True,exist_ok=True)
    _CACHE[kind]=args
    return args

def detection_run(output):
    import json
    config=output/'training_config.json'
    name=json.loads(config.read_text())['name'] if config.exists() else 'yolo26n_960_seed42'
    return output/'runs'/name

def platform_details():
    import platform
    import shutil
    import subprocess
    info={}
    for command,key in [('lscpu','cpu'),('nvidia-smi','driver')]:
        info[key]=subprocess.check_output([command],text=True) if shutil.which(command) else platform.processor()
    return info
