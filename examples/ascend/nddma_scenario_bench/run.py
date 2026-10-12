"""Compile and profile controlled three-dimensional slice consumers on Ascend950."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor

SYMBOLS = ['SliceNddma', 'SliceDmaRows', 'SliceDmaSpanGather', 'SliceDmaFullGather', 'SliceDmaLoop', 'SliceDmaNative', 'SliceDmaMerged']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(cmd, log, timeout=360):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def stats(v):
    s = sorted(v)
    return dict(n=len(v), median_us=statistics.median(v), min_us=s[0], max_us=s[-1],
                p25_us=s[len(s)//4], p75_us=s[3*len(s)//4])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['build', 'check', 'profile'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--binaries', type=Path)
    p.add_argument('--shapes', nargs='+', default=['17:65:4:16'])
    p.add_argument('--tiles', nargs='+', type=int, default=[16, 64])
    p.add_argument('--groups', nargs='+', type=int, default=[257, 4096])
    p.add_argument('--blocks', nargs='+', type=int, default=[32, 64])
    p.add_argument('--rounds', type=int, default=31)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--pipe', action='store_true')
    a = p.parse_args()
    if a.device not in (0, 1) or not 1 <= a.rounds <= 1000: p.error('Invalid device or rounds')
    if any(not 1 <= b <= 64 for b in a.blocks): p.error('Blocks must be in [1,64]')
    if any(not 1 <= g <= 65536 for g in a.groups): p.error('Groups must be in [1,65536]')
    if any(not 1 <= t <= 1024 for t in a.tiles): p.error('Tiles must be in [1,1024]')
    shapes = [tuple(map(int, s.split(':'))) for s in a.shapes]
    for h,w,r,c in shapes:
        if not (0 < r <= h and c >= 8 and c+1 <= w and r&(r-1)==0 and c&(c-1)==0):
            p.error('Requires power-of-two R/C, C>=8, R<=H, C+1<=W')
        if h*w+r*c > 23000 or max(a.groups)*h*w*4 > 512*1024*1024:
            p.error('Shape exceeds UB or input allocation bound')
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    src = Path(__file__).with_name('slice_relu.asc').resolve()
    (root/'source.asc').write_bytes(src.read_bytes())
    report = dict(source_sha256=sha(src), driver_sha256=sha(Path(__file__)), device=a.device,
                  rounds=a.rounds, warmup=20, modes=SYMBOLS, cases=[])
    (root/'npu-before.txt').write_text(subprocess.check_output(['npu-smi','info'], text=True))
    configs = list(itertools.product(shapes, a.tiles))
    def name(shape,t): return 's'+'_'.join(map(str,shape))+f'_t{t}'
    def build(config):
        shape,t = config
        folder = root/name(shape,t)
        folder.mkdir()
        cmd = ['bisheng',str(src),'-o',str(folder/'kernel'),'--npu-arch=dav-3510','-O2','-std=c++17',
               '-B/usr/lib/gcc/x86_64-linux-gnu/9','-L/usr/lib/gcc/x86_64-linux-gnu/9','-lm']
        cmd += [f'-D{k}={v}' for k,v in zip(['SOURCE_ROWS','SOURCE_COLS','TAKE_ROWS','TAKE_COLS','TILE_GROUPS'],[*shape,t])]
        (folder/'command.json').write_text(json.dumps(cmd,indent=2))
        run(cmd,folder/'compile.log')
        return dict(shape=list(shape),tile=t,name=name(shape,t),binary_sha256=sha(folder/'kernel'))
    if a.phase == 'build':
        with ThreadPoolExecutor(max_workers=3) as pool:
            report['cases'] = list(pool.map(build,configs))
    else:
        prior = json.loads((a.binaries/'results.json').read_text())
        if prior['source_sha256'] != sha(src): raise RuntimeError('Source changed since build')
        for shape,t in configs:
            key = name(shape,t)
            entry = next(c for c in prior['cases'] if c['name']==key)
            binary = (a.binaries/key/'kernel').resolve()
            if sha(binary) != entry['binary_sha256']: raise RuntimeError('Binary changed')
            folder=root/key
            folder.mkdir()
            cases=list(itertools.product(a.groups,a.blocks))
            rounds=a.rounds if a.phase=='profile' else 0
            app=[str(binary),str(rounds),str(a.device)]+[f'{g}:{b}' for g,b in cases]
            cmd=app
            if rounds:
                cmd=['msprof',f'--output={folder/"profile"}','--task-time=on','--sys-lp=off',f'--application={shlex.join(app)}']
                cmd += ['--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization'] if a.pipe else ['--ai-core=off']
            (folder/'command.json').write_text(json.dumps(cmd,indent=2))
            run(cmd,folder/'run.log')
            log=(folder/'run.log').read_text()
            if log.count('PASS,pattern=') != 3*len(SYMBOLS)*len(cases) or 'FAIL' in log or 'MISMATCH' in log:
                raise RuntimeError(f'Correctness checks incomplete: {folder}')
            if rounds and log.count('status=PASS') != len(SYMBOLS)*len(cases): raise RuntimeError('Missing post checks')
            data=[]
            for csvpath in (folder/'profile').rglob('op_summary*.csv'):
                with csvpath.open() as f: data.extend(csv.DictReader(f))
            tasks={}
            if rounds:
                for m,symbol in enumerate(SYMBOLS):
                    rows=sorted((v for v in data if symbol in v['Op Name']),key=lambda v:float(v['Task Start Time(us)']))
                    if len(rows)!=(rounds+23)*len(cases): raise RuntimeError(f'Unexpected sample count {symbol}: {len(rows)}')
                    tasks[m]=rows
            for i,(g,b) in enumerate(cases):
                result=dict(entry,groups=g,blocks=b,correctness='PASS',case_index=i)
                if rounds:
                    result['device_tasks']={}
                    for m in range(len(SYMBOLS)):
                        selected=tasks[m][i*(rounds+23)+23:(i+1)*(rounds+23)]
                        result['device_tasks'][str(m)]=stats([float(v['Task Duration(us)']) for v in selected])
                        if a.pipe:
                            result.setdefault('pipe',{})[str(m)]={k:stats([float(v[k]) for v in selected]) for k in selected[0]
                                if k.endswith('_time(us)') and all(v[k] not in ('','N/A') for v in selected)}
                report['cases'].append(result)
                print(key,g,b,result.get('device_tasks','PASS'),flush=True)
            (root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    (root/'npu-after.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))

if __name__=='__main__': main()
