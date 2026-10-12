# 第二轮：补零、步长抽取、MX 系数与五维切片

这些是独立 AscendC 对照实验，不依赖 TileLang 编译器修改，也不是整网性能结果。第一轮 `slice_relu.asc` 和其原有 driver 保持冻结。

| 入口 | 语义 | 主要基线 |
| --- | --- | --- |
| `pad_bundle.py` | FP32 两侧补零后 ReLU | 逐矩阵 DMA、分组硬件 loop、整块读入后 hoist 索引的 gather、最后写零边界 |
| `strided_bundle.py` | FP32 固定步长取数后 ReLU | 短 burst、整块读入后 gather |
| `byte_bundle.py` | FP8 E4M3／E8M0 的固定步长、逐位保真抽取 | 上述 DMA、32B burst、紧凑 DMA、256／512 线程 SIMT、SIMT 打包写 |
| `mx_bundle.py` | `[G,R,C] → [G,(R+1)/2,C,2]`，奇数 R 的末尾补零 | DMA 分离奇偶行、整块读入＋向量重排、SIMT |
| `crop_bundle.py` | `[G,H,H,H,W] → [G,R,R,R,C]` 的前部矩形切片 | 硬件 loop 轮换 CPU 外循环、全读 gather、只多读行内跨度的 DMA＋gather、按 4B 读写的 SIMT |

`scenario_bundle.py` 是这五个入口共用的编译、正确性和 msprof 任务时间采集器；原始第一轮 `bundle.py` 不受影响。

## 使用与边界

仅在已配置的 Ascend950 设备环境运行。以单字节抽取的确认配置为例：

```shell
python byte_bundle.py build --configs byte_confirm_configs.json --output /path/to/artifacts/build
python byte_bundle.py check --build /path/to/artifacts/build --output /path/to/artifacts/check --groups 1 3 63 65 4097 --blocks 1 64 --device 0
python byte_bundle.py profile --build /path/to/artifacts/build --output /path/to/artifacts/run-a --groups 4096 --blocks 64 --rounds 101 --device 0
```

- 输出目录必须不存在；旧记录不覆盖。配置、源码、二进制 SHA256、命令、原始日志与任务时间均保留。
- JSON 配置的七项依次为 `SOURCE_ROWS, SOURCE_COLS, TAKE_ROWS, TAKE_COLS, TILE_GROUPS, TRANSPOSE, SOURCE_OFFSET`。这些实验复用同一编译入口；各场景如何解释形状以对应源文件的 `constexpr` 和断言为准。
- byte 场景：输入按 `[G,128,STEP]` 理解，输出 `[G,128]`，取最后一维第 0 项。`TRANSPOSE=0` 选择 FP8 E4M3 NDDMA 重载，`TRANSPOSE=1` 选择 E8M0 重载。数据不做数值转换，比较的是全部原始位；不代表完成了量化或者 GEMM。
- 当前 byte 配置支持 `STEP>=32` 的二次幂，范围与 UB／索引长度由静态断言约束。旧筛选里更小 STEP 的源码快照保留在实验记录中；不能用更新后的源码重新标记旧结果。
- MX 场景提取自官方 `DynamicMxQuantPost::CopyInPaddingZero / ComputeInterleaveVF` 的数据排列，使用 E8M0 存储位；独立实现只测这段后处理，不运行完整 DynamicMxQuant。当前仅覆盖奇数轴 `R=2^k-1`。
- crop 场景是通用五维 Slice 的访存模式；没有声称已接入某个真实模型。当前配置使用二次幂切片、对齐的 loop 外层跨度和不超过 64KiB 的单组源数据。
- 三种输入模式、完整 CPU 参考、输出两侧 guard、计时后复检。byte／MX／crop 都要求逐字节相等。没有用近似误差掩盖地址错误。
- 预热 20 次，模式轮换发射；以 msprof 设备任务时间为准。`--pipe` 会改变时间，单独用于解释流水，不能与普通计时混用。
- 所有性能比较都要跨 tile／核数选出有效基线，再固定配置做独立确认。不同版本的筛选记录不能混成一次固定版本实验。
- NDDMA 与 DMA 都可能受缓存命中影响。被选中的有效字节数不等于物理 HBM 事务字节数。

完整结果、固定提交与判断保存在文档仓 `Work/tilelang-gm-ub-loop-nddma/scenario-search/`。
