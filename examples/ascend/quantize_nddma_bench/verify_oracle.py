"""Cross-check the actual C++ host FP8 encoder against ml_dtypes on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import ml_dtypes
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
source = Path(__file__).with_name('quantize_broadcast.asc').read_text()
encoder = source[source.index('float Decode('):source.index('void Check(')]
cpp = a.output / 'oracle.cpp'
cpp.write_text('#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <limits>\n#include <vector>\n' + encoder + '\nint main() { float x; while (std::fread(&x, 4, 1, stdin)==1) { uint8_t y=Encode(x); std::fwrite(&y,1,1,stdout); } }\n')
binary = a.output / 'oracle'
subprocess.run(['g++', '-O2', '-std=c++17', str(cpp), '-o', str(binary)], check=True)
values = np.arange(127, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
mid = (values[:-1]+values[1:])/np.float32(2)
boundary = np.concatenate([values, mid, np.nextafter(mid, np.float32(0)), np.nextafter(mid, np.float32(1000))])
rng = np.random.default_rng(12345)
x = np.concatenate([boundary, -boundary, rng.uniform(-1000,1000,1_000_000).astype(np.float32), np.array([0.0, -0.0],dtype=np.float32)])
golden = np.clip(x, -448, 448).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
actual = np.frombuffer(subprocess.check_output([str(binary.resolve())], input=x.tobytes()), dtype=np.uint8)
np.testing.assert_array_equal(actual, golden)
result = {'status':'PASS', 'samples':len(x), 'numpy':np.__version__, 'ml_dtypes':ml_dtypes.__version__, 'semantics':'finite FP32 -> clamped [-448,448] -> FP8 E4M3FN RNE; signed zero preserved'}
(a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(result)
