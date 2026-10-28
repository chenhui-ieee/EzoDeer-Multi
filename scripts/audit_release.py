"""Scan the public code directory without printing potential secret values.

This is a pattern/file inventory audit, not a legal determination of ownership.
Generated audit reports must be written outside the code directory.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re

ALLOWED_SUFFIXES={'.py','.md','.toml','.json','.txt','.yaml','.yml'}
PATTERNS={
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'cloud_access_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'github_token':re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    'provider_token':re.compile(r'\b(?:sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{24,})\b'),
    'credential_assignment':re.compile(r'''(?i)\b(?:api_?key|api_?token|access_?token|password|client_secret)\s*[:=]\s*['"][^'"\n]{8,}['"]'''),
    'machine_path':re.compile(r'''/(?:home|Users|mnt|media)/[^\s'"<>]+|[A-Za-z]:\\Users\\[^\s'"<>]+'''),
}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();report=args.report.resolve()
    if root==report or root in report.parents:parser.error('Write the audit report outside the public code directory.')
    findings=[];manifest=[]
    for path in sorted(root.rglob('*')):
        if any(p in {'.git','__pycache__','build','dist'} or p.endswith('.egg-info') for p in path.relative_to(root).parts):continue
        if path.is_symlink():findings.append(dict(file=path.relative_to(root).as_posix(),kind='symlink'));continue
        if not path.is_file():continue
        relative=path.relative_to(root).as_posix()
        if path.suffix not in ALLOWED_SUFFIXES and path.name not in {'.gitignore'}:findings.append(dict(file=relative,kind='unexpected_file_type'))
        data=path.read_bytes();manifest.append(dict(file=relative,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        if len(data)>1_000_000:findings.append(dict(file=relative,kind='large_file'))
        try:text=data.decode('utf-8')
        except UnicodeDecodeError:findings.append(dict(file=relative,kind='binary_file'));continue
        for kind,pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(dict(file=relative,kind=kind,line=text.count('\n',0,match.start())+1))
    result=dict(scope='public code directory only',files_scanned=len(manifest),total_bytes=sum(f['bytes'] for f in manifest),
        findings=findings,manifest=manifest,limitations='Pattern matching cannot establish copyright ownership, legal authorization or absence of all conceivable secrets. No dataset, videos, model weights, fonts or vendor packages are intended to be bundled.')
    report.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Scanned {len(manifest)} files; {len(findings)} findings. Secret values are never printed.')
    if findings:
        for finding in findings:print(json.dumps(finding))
        raise SystemExit(1)

if __name__=='__main__':main()
