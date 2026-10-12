"""Build multiple shape specializations into a single ACL process."""
import argparse
import csv
import json
from pathlib import Path
import sys
import shlex
import subprocess
from run import sha, run, stats



def main(source_name, SYMBOLS):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['build','profile','check'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--build',type=Path)
    p.add_argument('--configs',type=Path)
    p.add_argument('--groups',nargs='+',type=int,default=[257,4096])
    p.add_argument('--blocks',nargs='+',type=int,default=[32,64])
    p.add_argument('--indices',nargs='+',type=int)
    p.add_argument('--rounds',type=int,default=31)
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--pipe',action='store_true')
    a=p.parse_args()
    if a.device not in (0,1) or not 1<=a.rounds<=1000: p.error('Invalid device or rounds')
    if any(not 1<=b<=64 for b in a.blocks) or any(not 1<=g<=65536 for g in a.groups): p.error('Invalid groups or blocks')
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).with_name(source_name).resolve()
    record=dict(source_sha256=sha(source),driver_sha256=sha(Path(__file__)),launcher_sha256=sha(Path(sys.argv[0])),helpers_sha256=sha(Path(__file__).with_name('run.py')),device=a.device,rounds=a.rounds,warmup=20,modes=SYMBOLS,cases=[])
    (root/'source.asc').write_bytes(source.read_bytes())
    dependencies = [source.with_name('kv_reference_upstream.h')] if source_name == 'kv_cache.asc' else []
    record['dependencies'] = {p.name: sha(p) for p in dependencies}
    for dependency in dependencies:
        (root/dependency.name).write_bytes(dependency.read_bytes())
    (root/'npu-before.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))
    if a.phase=='build':
        configs=json.loads(a.configs.read_text())
        configs=[c+[0,1] if len(c)==5 else c for c in configs]
        lines=source.read_text().split('#ifndef TRANSPOSE')[0].splitlines()
        for i,c in enumerate(configs):
            for k,v in zip(['SOURCE_ROWS','SOURCE_COLS','TAKE_ROWS','TAKE_COLS','TILE_GROUPS','TRANSPOSE','SOURCE_OFFSET'],c): lines.append(f'#define {k} {v}')
            lines += [f'namespace variant{i} {{',f'#include "{source}"','}']
            lines += ['#undef '+k for k in ['SOURCE_ROWS','SOURCE_COLS','TAKE_ROWS','TAKE_COLS','TILE_GROUPS','TRANSPOSE','SOURCE_OFFSET']]
        lines += ['int main(int argc,char **argv) {','std::setvbuf(stdout,nullptr,_IOLBF,0);',
                  'if(argc<4) return 2; int rounds=std::atoi(argv[1]),device=std::atoi(argv[2]);',
                  'if(rounds<0 || rounds>1000 || device<0 || device>1) return 2;',
                  'variant0::Check(aclInit(nullptr),"init"); variant0::Check(aclrtSetDevice(device),"device");',
                  'aclrtStream stream=nullptr; variant0::Check(aclrtCreateStream(&stream),"stream"); int status=0;',
                  'for(int i=3;i<argc && !status;++i) { unsigned index,g,b; char extra;',
                  'if(std::sscanf(argv[i],"%u:%u:%u%c",&index,&g,&b,&extra)!=3) return 2;',
                  'std::printf("VARIANT,%u\\n",index); switch(index) {']
        lines += [f'case {i}: status=variant{i}::Case(g,b,rounds,stream); break;' for i in range(len(configs))]
        lines += ['default: return 2; }} aclrtDestroyStream(stream); aclrtResetDevice(device); aclFinalize(); return status; }']
        wrapper=root/'bundle.asc'; wrapper.write_text('\n'.join(lines)+'\n')
        cmd=['bisheng',str(wrapper),'-o',str(root/'kernel'),'--npu-arch=dav-3510','-O2','-std=c++17',
             '-B/usr/lib/gcc/x86_64-linux-gnu/9','-L/usr/lib/gcc/x86_64-linux-gnu/9','-lm','-I'+str(source.parent)]
        (root/'command.json').write_text(json.dumps(cmd,indent=2)); run(cmd,root/'compile.log',900)
        record.update(configs=configs,binary_sha256=sha(root/'kernel'))
    else:
        prior=json.loads((a.build/'results.json').read_text())
        binary=(a.build/'kernel').resolve()
        if sha(binary)!=prior['binary_sha256'] or sha(source)!=prior['source_sha256']: raise RuntimeError('Source/binary changed')
        if prior.get('dependencies', {}) != record['dependencies']: raise RuntimeError('Dependencies changed')
        configs=prior['configs']; indices=a.indices if a.indices is not None else list(range(len(configs)))
        if any(i<0 or i>=len(configs) for i in indices): p.error('Invalid variant index')
        cases=[(i,g,b) for i in indices for g in a.groups for b in a.blocks]
        rounds=a.rounds if a.phase=='profile' else 0
        app=[str(binary),str(rounds),str(a.device)]+[f'{i}:{g}:{b}' for i,g,b in cases]
        cmd=app
        if rounds:
            cmd=['msprof',f'--output={root/"profile"}','--task-time=on','--sys-lp=off',f'--application={shlex.join(app)}']
            cmd += ['--ai-core=on','--aic-mode=task-based','--aic-metrics=PipeUtilization'] if a.pipe else ['--ai-core=off']
        (root/'command.json').write_text(json.dumps(cmd,indent=2)); run(cmd,root/'run.log',900)
        log=(root/'run.log').read_text()
        if log.count('PASS,pattern=')!=3*len(SYMBOLS)*len(cases) or 'MISMATCH' in log or 'FAIL' in log: raise RuntimeError('Incomplete correctness')
        if rounds and log.count('status=PASS')!=len(SYMBOLS)*len(cases): raise RuntimeError('Incomplete post checks')
        data=[]
        for path in (root/'profile').rglob('op_summary*.csv'):
            with path.open() as f: data.extend(csv.DictReader(f))
        modes={}
        if rounds:
            for m,symbol in enumerate(SYMBOLS):
                rows=sorted((v for v in data if symbol in v['Op Name'] and not any(symbol!=s and symbol in s and s in v['Op Name'] for s in SYMBOLS)),key=lambda v:float(v['Task Start Time(us)']))
                if len(rows)!=(rounds+23)*len(cases): raise RuntimeError(f'Unexpected tasks {symbol}: {len(rows)}')
                modes[m]=rows
        for j,(i,g,b) in enumerate(cases):
            result=dict(index=i,config=configs[i],groups=g,blocks=b,correctness='PASS')
            if rounds:
                result['device_tasks']={}
                for m in range(len(SYMBOLS)):
                    rows=modes[m][j*(rounds+23)+23:(j+1)*(rounds+23)]
                    result['device_tasks'][str(m)]=stats([float(v['Task Duration(us)']) for v in rows])
                    if a.pipe: result.setdefault('pipe',{})[str(m)]={k:stats([float(v[k]) for v in rows]) for k in rows[0]
                        if k.endswith('_time(us)') and all(v[k] not in ('','N/A') for v in rows)}
            record['cases'].append(result)
            print(i, configs[i], g, b, {m: v['median_us'] for m, v in result.get('device_tasks', {}).items()} or 'PASS', flush=True)
        record.update(configs=configs,binary_sha256=prior['binary_sha256'])
    (root/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    (root/'npu-after.txt').write_text(subprocess.check_output(['npu-smi','info'],text=True))

