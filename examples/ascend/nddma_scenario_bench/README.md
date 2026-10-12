# NDDMA：切片与批量转置的强基线实验

独立 AscendC 实验，不依赖 TileLang 编译器。数据模式来自官方 ops-math StridedSlice / Transpose 支持的操作，融合 ReLU 作为消费 UB tile 的最小计算。**不是官方完整算子或整网测速；融合组合也不代表某个模型已经部署。**

## 语义与边界

- 输入 FP32 `[G,H,W]`，先取 `X[:, :R, offset:offset+C]`，随后可交换 R/C 两轴，最后做 ReLU。
- R、C 必须为 2 的幂，C≥8，R≤H，offset+C≤W；测试配置 offset=0/1。
- 本轮只验证有限 FP32 值；不覆盖 NaN、FP8/FP4、MX、非单位切片步长、padding 或任意形状。
- 每 tile 的输入与输出共享相同 UB 预算；多读方案可能因此减少实际 tile 组数。双缓冲分配含向量读取余量。主机限制 groups≤65536、blocks≤64、单输入≤512 MiB。
- R=8 的专用转置将两个矩阵拼成 16 行；奇数尾组多出的矩阵只写 UB 预留空间，不写 GM。原生转置的 repeat 方向按形状在组、行块、列块之间选择。

## 七版对照

| 模式 | 做法 |
| --- | --- |
| 0 `SliceNddma` | NDDMA 直接切片/转置到目标 UB 布局，再 ReLU |
| 1 `SliceDmaRows` | 每组一次二维 DMA；切片直接计算，转置用 Vector gather |
| 2 `SliceDmaSpanGather` | 二维 DMA 搬入各组的行跨度，再 gather 提取/转置 |
| 3 `SliceDmaFullGather` | 连续搬入完整组，再 gather 提取/转置 |
| 4 `SliceDmaLoop` | 源组 stride 满足 32B 对齐时用 GM→UB loop，否则回退模式 1 |
| 5 `SliceDmaNative` | 普通 DMA＋`transto5hd_b32` 专用转置，融合 ReLU；小于 8 行时回退 gather |
| 6 `SliceDmaMerged` | 合并批次做专用转置，再用二维 UB→GM 恢复批次顺序；小于 16 行时回退 gather |

完整连续输入时，普通方案均使用一次连续 DMA，避免不必要的逐组发射。gather 的固定索引计算在向量循环之外。部分模式在特定 shape 下等价，不把它们当作七种不同硬件能力。

## 编译与验证

只在授权的 Ascend950 设备运行；先 source 工作区规定的环境入口。`bundle.py` 将多个编译期特化放进一个二进制、一次 ACL 上下文，减少共享环境的反复初始化。

```bash
python bundle.py build --output results/build --configs final_configs.json
python bundle.py check --output results/check --build results/build \
  --groups 1 3 63 65 --blocks 1 32 --device 0
python bundle.py profile --output results/screen --build results/build \
  --groups 257 4096 --blocks 16 32 64 --rounds 31 --device 0
python bundle.py profile --output results/holdout-a --build results/build \
  --indices 5 --groups 4096 --blocks 32 64 --rounds 101 --device 0
python bundle.py profile --output results/holdout-b --build results/build \
  --indices 5 --groups 4096 --blocks 64 32 --rounds 101 --device 0
```

`--indices 5` 对应最终配置中的 R=8、C=32、tile=32。每组先做三个随机输入模式、七版逐元素对照 CPU 直接索引和 ReLU；输出前后各 256B guard。计时后再次全量验证。输入有余量，不声称做了越界读 sanitizer。

性能取 msprof `Task Duration(us)`：每版排除 3 次正确性调用和 20 次预热，剩余样本数必须严格匹配轮次。启动顺序逐轮轮换，输出 CSV、日志、源码/二进制哈希与分位数。`--pipe` 单独采集流水计数器，其开销不能与普通计时混比。

重复使用同一输入输出缓冲区，是热缓存实验；没有测得 HBM 物理流量或证明带宽上限。核数/tile 必须分别调优，再固定双方配置复测。编译命令中的 GCC 9 路径为当前设备环境设置，换机器需要调整。

## 官方源码依据

固定 ops-math `ba8d7909d0f89ce71a04804b0c585f4033937cfe`：

- [StridedSlice 的搬运策略选择](https://gitcode.com/cann/ops-math/blob/ba8d7909d0f89ce71a04804b0c585f4033937cfe/conversion/strided_slice/op_host/arch35/strided_slice_tiling_arch35.cpp#L539)
- [Transpose 的调度与策略选择](https://gitcode.com/cann/ops-math/blob/ba8d7909d0f89ce71a04804b0c585f4033937cfe/conversion/transpose/op_host/arch35/transpose_tiling_arch35.cpp#L83)

完整阶段数据、版本对应关系和结论归档在文档仓 `Work/tilelang-gm-ub-loop-nddma/scenario-search/`；早期筛选源码在各实验的 `source.asc` 中冻结，不能把旧测量直接标成最终七版结果。

第二轮补零、FP8／E8M0 步长抽取、MX 系数和五维切片实验见 [ROUND2.md](ROUND2.md)。

KV cache 官方 NZ 路径的页表聚集实验见 [KV_CACHE.md](KV_CACHE.md)。
