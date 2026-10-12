# Quantize：NDDMA 系数广播与普通 DMA 对照

这是从官方 ops-nn Quantize 的 per-channel NDDMA 路径抽取的独立 AscendC 实验，**不是官方完整算子或整网测速**。不依赖 TileLang 编译器。

输入 `X[M,C]` 与 `scale[C]` 为 FP32，输出为 FP8 E4M3FN：`Y[m,c] = saturating_RNE(X[m,c] * scale[c])`。只支持有限输入、C=8/16、无 offset；不覆盖动态 scale 计算、FP4/MX、其他舍入模式或完整 Quantize API。

| 模式 | 系数准备 | 计算时使用 |
| --- | --- | --- |
| 0 `QuantNddmaTile` | NDDMA 将一行系数广播成整个 tile | 每个向量加载对应系数 |
| 1 `QuantUbBroadcast` | 普通 DMA 搬一行，Vector 一次展开成整个 tile | 与模式 0 相同 |
| 2 `QuantRegisterReuse` | 普通 DMA 只搬一行 | 每个 VF 开始时 gather 一次，在内部向量循环中复用寄存器 |
| 3 `QuantNddmaRegister` | NDDMA 只展开成 64 个 FP32 | 每个 VF 加载一次，在内部向量循环中复用寄存器 |

四版每核只准备一次 scale，并跨多个输入 tile 复用；主输入和输出使用相同双缓冲流水。没有把普通版写成每行一次 DMA，也没有让它每 tile 重复搬 scale。

## 运行

仅在配套的 Ascend950 开发机运行；先 source 工作区约定的环境入口。`--device` 必须选择已获授权的空闲设备。以下相对目录为新实验输出，驱动拒绝覆盖已有结果。

```bash
python run_benchmark.py build --output results/build --channels 8 16
python run_benchmark.py check --output results/check --binaries results/build \
  --channels 8 16 --tiles 1024 8192 --blocks 1 32 --rows 1 3 127 257 4097 --device 1
python run_benchmark.py profile --output results/sweep --binaries results/build \
  --channels 16 --tiles 1024 4096 8192 --blocks 32 64 \
  --rows 4096 65536 1048576 --rounds 31 --device 1
```

`--pipe` 增加流水计数器。编译驱动默认至多三个并行编译任务；设备测试串行运行。固定 GCC 9 主机链接目录来自当前测试环境，在其他机器上需要修改。

## 正确性与计时

- 每次运行，三个数据模式 × 四个版本逐字节与 CPU FP8 参考比较；包含随机值、全部正负相邻编码的舍入中点及两侧、溢出饱和、负零。
- scale 在模式间改变，检查复用后不会读取旧值。输出前后各 256B guard，检查越界写入；输入预留尾部空间，**不能据此声称做了越界读检测**。
- 非整 tile 与多轮双缓冲由 `check` 的 shape 集合覆盖。结束计时后再次核验四个版本的结果与 guard。
- CPU 参考可独立用 `verify_oracle.py --output <directory>` 对照 `ml_dtypes`；需要 NumPy、ml_dtypes 和 g++。
- 每版预热 20 次，计时顺序逐轮轮换；以 msprof `Task Duration(us)` 为结论依据。每版跳过前三次正确性调用和 20 次预热，严格检查剩余样本数。
- 重复使用同一缓冲区，属于热缓存实验；不能将有效字节数除耗时当成实测 HBM 带宽。共享机器的残留任务与频率波动需结合现场记录判断。

`results.json` 记录源码/二进制 SHA-256、配置、正确性和设备耗时分布；原始 profiler CSV 保留在各 case 目录。优化结论需比较各版在相同扫描范围内的最佳配置，并以独立复测确认，不能把单轮最低值当普遍收益。

## 官方依据

固定 ops-nn 提交 `2c28f2ba5426ac80347ccedc6ffa3f5a92f0db4c`：

- [host 选择 NDDMA 路径](https://gitcode.com/cann/ops-nn/blob/2c28f2ba5426ac80347ccedc6ffa3f5a92f0db4c/quant/quantize/op_host/arch35/quantize_tiling_arch35.cpp#L614)
- [NDDMA 广播及 scale 跨输入块复用](https://gitcode.com/cann/ops-nn/blob/2c28f2ba5426ac80347ccedc6ffa3f5a92f0db4c/quant/quantize/op_kernel/arch35/quantize_per_channel_nddma_no_offset_regbase.h#L81)
- [FP8 浮点溢出控制与 CastTrait](https://gitcode.com/cann/ops-nn/blob/2c28f2ba5426ac80347ccedc6ffa3f5a92f0db4c/quant/quantize/op_kernel/arch35/quantize.h#L85)
