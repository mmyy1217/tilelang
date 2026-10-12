"""Build and validate paired FP8 quantization kernels on an Ascend950 device."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shlex
import statistics
import subprocess

SYMBOLS = ['QuantNddmaTile', 'QuantUbBroadcast', 'QuantRegisterReuse', 'QuantNddmaRegister', 'QuantUbRegister']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log, timeout=300):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def summary(values):
    return {'n': len(values), 'median_us': statistics.median(values),
            'min_us': min(values), 'max_us': max(values),
            'p25_us': sorted(values)[len(values)//4], 'p75_us': sorted(values)[3*len(values)//4]}


def validate(log, rounds):
    text = log.read_text()
    if text.count('PASS,pattern=') != 3 * len(SYMBOLS) or 'MISMATCH' in text or 'FAIL' in text:
        raise RuntimeError(f'Correctness failure or incomplete checks: {log}')
    if rounds and text.count('status=PASS') != len(SYMBOLS):
        raise RuntimeError(f'Missing post-timing checks: {log}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['build', 'check', 'profile'])
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--binaries', type=Path)
    p.add_argument('--channels', nargs='+', type=int, default=[16])
    p.add_argument('--tiles', nargs='+', type=int, default=[1024, 4096, 8192])
    p.add_argument('--blocks', nargs='+', type=int, default=[32, 64])
    p.add_argument('--rows', nargs='+', type=int, default=[4096, 65536, 1048576])
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--rounds', type=int, default=31)
    p.add_argument('--pipe', action='store_true')
    args = p.parse_args()
    if any(c not in (8, 16) for c in args.channels): p.error('Only C=8 or C=16 is supported')
    if any(t < 64 or t > 8192 or t % 64 for t in args.tiles): p.error('Tile must be a multiple of 64 in [64,8192]')
    if any(b < 1 or b > 64 for b in args.blocks): p.error('Blocks must be in [1,64]')
    if not 1 <= args.rounds <= 1000: p.error('Rounds must be in [1,1000]')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('quantize_broadcast.asc').resolve()
    report = {'source_sha256': sha(source), 'device': args.device, 'warmup': 20,
              'rounds': args.rounds, 'modes': SYMBOLS, 'cases': []}
    (root / 'source.asc').write_bytes(source.read_bytes())
    (root / 'npu-before.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))
    def build(config):
        c, tile = config
        folder = root / f'c{c}_t{tile}'
        folder.mkdir()
        command = ['bisheng', str(source), '-o', str(folder/'kernel'), '--npu-arch=dav-3510', '-O2',
                   '-std=c++17', '-B/usr/lib/gcc/x86_64-linux-gnu/9', '-L/usr/lib/gcc/x86_64-linux-gnu/9',
                   '-lm', f'-DCHANNELS={c}', f'-DTILE_ELEMENTS={tile}']
        (folder/'command.json').write_text(json.dumps(command, indent=2))
        run(command, folder/'compile.log')
        result = {'channels': c, 'tile': tile, 'binary_sha256': sha(folder/'kernel')}
        print('BUILD', result, flush=True)
        return result
    if args.phase == 'build':
        with ThreadPoolExecutor(max_workers=3) as pool:
            report['cases'] = list(pool.map(build, itertools.product(args.channels, args.tiles)))
    else:
        if not args.binaries: p.error('--binaries is required')
        prior = json.loads((args.binaries/'results.json').read_text())
        if prior['source_sha256'] != sha(source): raise RuntimeError('Source differs from compiled source')
        for c, tile, blocks, rows in itertools.product(args.channels, args.tiles, args.blocks, args.rows):
            name = f'c{c}_t{tile}_b{blocks}_r{rows}'
            folder = root/name
            folder.mkdir()
            binary = (args.binaries/f'c{c}_t{tile}'/'kernel').resolve()
            entry = next(x for x in prior['cases'] if x['channels']==c and x['tile']==tile)
            if sha(binary) != entry['binary_sha256']: raise RuntimeError('Binary hash mismatch')
            rounds = args.rounds if args.phase == 'profile' else 0
            app = [str(binary), str(rows), str(blocks), str(rounds), str(args.device)]
            result = dict(entry, name=name, blocks=blocks, rows=rows)
            if args.phase == 'check':
                run(app, folder/'run.log', 90)
                validate(folder/'run.log', 0)
            else:
                command = ['msprof', f'--output={folder / "profile"}', '--task-time=on',
                           '--sys-lp=off', f'--application={shlex.join(app)}']
                command += ['--ai-core=on', '--aic-mode=task-based', '--aic-metrics=PipeUtilization'] if args.pipe else ['--ai-core=off']
                (folder/'command.json').write_text(json.dumps(command, indent=2))
                run(command, folder/'run.log')
                validate(folder/'run.log', rounds)
                data = []
                for path in (folder/'profile').rglob('op_summary*.csv'):
                    with path.open() as stream: data.extend(csv.DictReader(stream))
                result['device_tasks'] = {}
                for mode, symbol in enumerate(SYMBOLS):
                    selected = sorted((x for x in data if symbol in x['Op Name']), key=lambda x: float(x['Task Start Time(us)']))
                    if len(selected) != rounds + 23:
                        raise RuntimeError(f'{name}/{symbol}: expected {rounds+23} tasks, got {len(selected)}')
                    selected = selected[23:]
                    result['device_tasks'][str(mode)] = summary([float(x['Task Duration(us)']) for x in selected])
                    if args.pipe:
                        result.setdefault('pipe', {})[str(mode)] = {
                            key: summary([float(x[key]) for x in selected]) for key in selected[0]
                            if key.endswith('_time(us)') and all(x[key] not in ('', 'N/A') for x in selected)}
            result['correctness'] = 'PASS'
            report['cases'].append(result)
            (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
            print(name, result.get('device_tasks', 'PASS'), flush=True)
    (root/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    (root/'npu-after.txt').write_text(subprocess.check_output(['npu-smi', 'info'], text=True))

if __name__ == '__main__':
    main()
