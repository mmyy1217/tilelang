# copy lowering 结果快照

这里归档 2026-10-09 本地生成的 13 个用例结果。原始实验文件仍保留在本地实验目录；这里的副本随分支推送，方便在其他机器直接阅读。

## 从这里开始

先看 [原 kernel](../kernels.py#L7) 中的 dma_roundtrip，再看 [初始 IR](verified-context/dma/input.tir)。接着对照 AscendLowerTileOp [输入](verified-context/dma/pass.log#L1886)和[输出](verified-context/dma/pass.log#L1916)，最后看 [CCE 源码](verified-context/dma/kernel.cpp#L6)。核心变化是区域 copy 变成具体 DMA intrinsic、指针与搬运参数。

passes.tsv 的 before_line / after_line 指向同目录 pass.log；text_changed 只表示打印文本变化。日志行号在本快照中保持不变，使用编辑器跳转最方便。建议顺序：dma → strided → tail → dual_dma → nz_scatter，然后再看其余用例。

| 原始 IR | 首个建议阅读的 pass | 完整日志位置 | 更多结果 |
| --- | --- | --- | --- |
| [dma](verified-context/dma/input.tir) | tl.AscendLowerTileOp | [前](verified-context/dma/pass.log#L1886) / [后](verified-context/dma/pass.log#L1916) | [pass 索引](verified-context/dma/passes.tsv) / [CCE](verified-context/dma/kernel.cpp) |
| [strided](verified-context/strided/input.tir) | tl.AscendLowerTileOp | [前](verified-context/strided/pass.log#L1886) / [后](verified-context/strided/pass.log#L1916) | [pass 索引](verified-context/strided/passes.tsv) / [CCE](verified-context/strided/kernel.cpp) |
| [padding](verified-context/padding/input.tir) | tl.AscendLowerTileOp | [前](verified-context/padding/pass.log#L1999) / [后](verified-context/padding/pass.log#L2030) | [pass 索引](verified-context/padding/passes.tsv) / [CCE](verified-context/padding/kernel.cpp) |
| [tail](verified-context/tail/input.tir) | tl.AscendInsertOOBPadding | [前](verified-context/tail/pass.log#L916) / [后](verified-context/tail/pass.log#L940) | [pass 索引](verified-context/tail/passes.tsv) / [CCE](verified-context/tail/kernel.cpp) |
| [dynamic](verified-context/dynamic/input.tir) | tl.AscendLowerTileOp | [前](verified-context/dynamic/pass.log#L2075) / [后](verified-context/dynamic/pass.log#L2108) | [pass 索引](verified-context/dynamic/passes.tsv) / [CCE](verified-context/dynamic/kernel.cpp) |
| [gemm](verified-context/gemm/input.tir) | tl.AscendInsertOOBPadding | [前](verified-context/gemm/pass.log#L12908) / [后](verified-context/gemm/pass.log#L13558) | [pass 索引](verified-context/gemm/passes.tsv) / [CCE](verified-context/gemm/kernel.cpp) |
| [dual_dma](verified-context/dual_dma/input.tir) | tl.RewriteDualCopy | [前](verified-context/dual_dma/pass.log#L360) / [后](verified-context/dual_dma/pass.log#L380) | [pass 索引](verified-context/dual_dma/passes.tsv) / [CCE](verified-context/dual_dma/kernel.cpp) |
| [dual_gemm](verified-context/dual_gemm/input.tir) | tl.RewriteDualCopy | [前](verified-context/dual_gemm/pass.log#L3505) / [后](verified-context/dual_gemm/pass.log#L3710) | [pass 索引](verified-context/dual_gemm/passes.tsv) / [CCE](verified-context/dual_gemm/kernel.cpp) |
| [nz_scatter](verified-context/nz_scatter/input.tir) | tl.InsertNd2Nz | [前](verified-context/nz_scatter/pass.log#L7835) / [后](verified-context/nz_scatter/pass.log#L8038) | [pass 索引](verified-context/nz_scatter/passes.tsv) / [CCE](verified-context/nz_scatter/kernel.cpp) |
| [simd](verified-context/simd/input.tir) | tl.AscendSimdVFLowerParallel | [前](verified-context/simd/pass.log#L2429) / [后](verified-context/simd/pass.log#L2468) | [pass 索引](verified-context/simd/passes.tsv) / [CCE](verified-context/simd/kernel.cpp) |
| [simt](verified-context/simt/input.tir) | tl.AscendLowerTileOp | [前](verified-context/simt/pass.log#L2738) / [后](verified-context/simt/pass.log#L2780) | [pass 索引](verified-context/simt/passes.tsv) / [CCE](verified-context/simt/kernel.cpp) |
| [pipeline](verified-context/pipeline/input.tir) | tl.MaterializeMultiBuffer | [前](verified-context/pipeline/pass.log#L1707) / [后](verified-context/pipeline/pass.log#L1762) | [pass 索引](verified-context/pipeline/passes.tsv) / [CCE](verified-context/pipeline/kernel.cpp) |
| [fp4](verified-context/fp4/input.tir) | tl.RewriteFp4ToFp4x2 | [前](verified-context/fp4/pass.log#L2801) / [后](verified-context/fp4/pass.log#L2824) | [pass 索引](verified-context/fp4/passes.tsv) / [CCE](verified-context/fp4/kernel.cpp) |

## 逐用例学习笔记

- [dma：嵌套 IR、DMA 参数、长度限制与硬件执行](verified-context/dma/学习笔记.md)

## 来源与验证边界

- 编译器基线：47976f210de3598444003e9b05de8204dfa7ff88；TVM：4211874e9e4b8770fd20ad0d98ab80afd1df5028。
- 观察工具提交：c6703ac6；kernel 文件 SHA256：89235cd30b08f0658482e85d6ecfbc99c58e3fdd164b572707b652d2b0c80aef。
- 快照生成于工具提交之前，因此 manifest 中的 tilelang_revision 记录编译器基线；kernel 哈希与本分支 kernels.py 一致。
- 13 个内置用例离线 lowering 成功，共 1122 次 pass 回调。每例保留 input.tir、host.tir、device.tir、pass.log、passes.tsv、manifest.json 和 kernel.cpp。
- verification.json 是原实验的验证汇总，其中 custom_pass_count 和 failure_capture 对应本地额外的工具自检；这里归档的是 13 个内置用例。
- manifest 中的绝对路径、耗时保留原运行环境信息，不要求其他机器有相同路径。仅阅读结果无需安装依赖。
- 结果验证范围是 TIR lowering 与 CCE 源码生成，没有运行 NPU、编译设备二进制或测量性能。

重建环境和生成新结果见[工具说明](../README.md)。
