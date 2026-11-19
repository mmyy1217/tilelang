# A5 DMA 1215 开发自测

以当前 31 项 L3 特性为 1215 验收基准，按下表的 L3 顺序 review。数值用例运行 kernel 后逐元素比较设备结果与 CPU 参考。L2 Cache 用例核对生成指令中的策略枚举，覆盖 DSL 到 Ascend C API 的参数传递。

GM→L1 的普通 DMA dtype 包括 `int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`float16`、`bfloat16`、`float32` 和 `float8_e4m3fn`。1215 的 FP8 验收范围仅包含 E4M3（`float8_e4m3fn`）。Packed FP4 的设备数值用例走 ND。DN 待核实。数值用例以原 dtype 完成 DMA，再用同位宽的 Cube 支持类型做单位矩阵乘读回 L1。FP4 主用例用 blockscaled GEMM 读回。

**验证记录**　A5-BLUE 上 481 个用例通过代码生成。L1→L0A/B 的 26 个组合、MX Scale 的 56 个组合、L0C→UB 保留 NZ 及 UB→L1 原样搬入通过 Bisheng 设备编译，数值结果待上卡。原有 OOB 用例在 GM→L1 实测中通过，16×64 和 32×64 各 20 个 OOB 组合通过设备编译，数值结果待上卡。

## 逐项 review

M、N、K 依次为矩阵乘的行、输出列和归约维。多项特性可共用端到端测试。

| ID | 方向与特性 | 用例文件 | shape、dtype 与判据 |
|---|---|---|---|
| 01 | GM→L1 ND→NZ | [test_nd_to_nz](test_gm_l1.py) | 10 种普通 dtype 完整搬入 16×32，读回原矩阵。FP4 完整块见第 05 项 |
| 02 | GM→L1 DN→NZ | [test_dn_to_nz](test_gm_l1.py) | 10 种普通 dtype，从连续 32×16 搬到 16×32，与转置参考比较 |
| 03 | GM→L1 批量 ND/DN→NZ | [test_batch_nd_to_nz / test_batch_dn_to_nz](test_gm_l1.py) | 3 个 16×32 矩阵，覆盖紧凑批次、批次间有间隔和交错行。10 种普通 dtype 逐批读回。[另测批次尾块和相邻矩阵保护区](test_batch.py)，FP4 批量数值用例待补 |
| 04 | GM→L1 越界尾块补零 | [test_oob_tail_padding_nd / dn](test_gm_l1.py) | 从列 96 请求完整的 16×32、16×64 或 32×64 NZ tile。覆盖仅行尾、仅列尾、双尾与整块越界。16×64 双尾用例有 31 列有效，检查有效 C0 组内剩余列及后续完整 C0 组。32×64 用例只有 4 列有效，读回确认两个行分形都补齐。10 种普通 dtype 逐元素验零，FP4 尾块数值用例待补 |
| 05 | GM→L1 FP4 双值打包 | [test_fp4_packed_nd_to_nz](test_gm_l1.py) | M=N=16，K=256，E2M1。独立解码高低半字节，覆盖原生 packed dtype 和 INT8 字节视图 |
| 06 | GM→L1 L2 读策略 | [test_gm_to_l1_l2_policy_nd / dn](test_gm_l1.py) | 16×32 目标，10 种普通 dtype 各走 ND 和 DN。FP4 走 16×64 ND。核对指令中的 `normal_fv`、`normal_lv`、`notalloc_clean` |
| 07 | L1→L0A/B 二维加载 | [test_l1_to_l0_matrix](test_l0_transpose.py) | 显式把 16×32 的 FP16/BF16 矩阵加载到 L0A/B，通过 GEMM 读回 |
| 08 | L1→L0A/B 转置加载 | [test_l1_to_l0a_transpose / test_l1_to_l0b_transpose / test_l1_to_l0_transpose_fp4](test_l0_transpose.py) | 两侧各覆盖 INT8/16/32、UINT8/16/32、FP16、BF16、FP32、FP8 E4M3、FP4 E2M1。普通 dtype 从 64×32 转置为 32×64，单位矩阵读回。L0A 另测 32×32 取 24 列（BF16）和 25 列（FP32）。FP4 走 64×64，使用单位 E8M0 缩放读回 |
| 09 | L1→L0A/B_MX 缩放因子 | [test_blockscaled_l0_gemm](test_mx_fp4.py) | 56 组。原有 7 个形状与转置场景分别覆盖主矩阵 FP8 E4M3、打包 FP4 E2M1，以及 A/B 两侧 scale 的 `uint16`、`uint8`、`int8`、`float8_e8m0fnu`。K=128，每行 4 个 E8M0，`uint16` 每元素装两个指数，其余每元素一个。A/B 使用独立 scale，以 CPU 展开后的矩阵乘逐元素校验 |
| 10 | L0C→GM NZ→ND | [test_l0c_to_gm_nd](test_fix_output.py) | 32×64 的 FP32 累加结果按 ND 写回，和 CPU GEMM 参考比较 |
| 11 | L0C→GM 降精度 | [fix_output](test_fix_output.py) | 32×64，FP32→FP16/BF16，与 CPU Cast 逐元素比较 |
| 12 | L0C→GM Atomic Add | [test_l0c_atomic_add](test_atomic.py) | 两核写同一 16×16 FP32 矩阵，预置 7，检查原值与两块 GEMM 结果之和 |
| 13 | L0C→GM L2 写策略 | [test_l0c_to_gm_l2_policy](test_fix_output.py) | 32×64 输出，FP32→FP32/FP16/BF16 和 INT32→INT32。核对 Fixpipe 指令的 `normal_fv`、`normal_lv`、`notalloc_ci`、`notalloc_pw` |
| 14 | L0C→UB NZ→ND | [test_l0c_to_ub_nd](test_fix_output.py) | 32×64，UB 再写回 GM 检查行列顺序 |
| 15 | L0C→UB NZ→NZ | [test_l0c_to_ub_nz](test_fix_output.py) | 32×64 FP32，UB 显式标注 NZ，SIMT 按逻辑坐标读回并与 CPU GEMM 比较 |
| 16 | L0C→UB 降精度 | [fix_output](test_fix_output.py) | 32×64，FP32→FP16/BF16，单目标 UB，回读后与 CPU 转换逐元素比较 |
| 17 | L0C→UB 指定 AIV | [fix_output](test_fix_output.py) | 两个 AIV 的 UB 预填 −7，逐一指定目标，分别回读 16×16 FP32 矩阵 |
| 18 | L0C→UB 硬件双写 | [test_l0c_to_ub_dual_m / dual_n](test_fix_output.py) | FP32 的 32×64 按 M 分成 16×64，按 N 分成 32×32，重组后比较完整矩阵 |
| 19 | UB→L1 连续原样搬入 | [test_ub_to_l1_contiguous_nz](test_ub_l1.py) | 32×32 BF16，CPU 先排好 NZ，AIV 将整块搬入 L1，单位矩阵乘读回 |
| 20 | UB→L1 ND→NZ | [test_ub_to_l1_nd_to_nz](test_ub_l1.py) | 32×64 FP32；32×128 FP32→BF16、BF16→FP32。单位矩阵乘读回重排结果 |
| 21 | UB→L1 软件双写 | [test_ub_to_l1_dual_nd_to_nz](test_ub_l1.py) | 同 20，以单 AIV 和双 AIV 对照 |
| 22 | GM→UB 连续/跨步 | [test_gm_to_ub](test_gm_ub.py) | 1×1、4×15、4×16、4×17。INT8/16/32、FP16/BF16/FP32；跨步时源行宽增加 7 元素，UB 行宽 64 |
| 23 | GM→UB 右补齐 | [test_gm_to_ub_pad_value / data_select](test_padding.py) | FP32，2×3→2×8、2×30→2×32，检查有效列及每个填充位置。pad_value 指定 −1；data_select 复用预设的 −1 寄存器值。当前 DSL 的这条路径没有自动取源首元素的语义 |
| 24 | GM→UB L2 读策略 | [test_gm_to_ub_l2_policy](test_gm_ub.py) | 4×16 FP32。核对对齐搬入指令的 `normal_fv`、`normal_lv`、`notalloc_clean` |
| 25 | GM→UB 软件分半 | [test_ub_to_l1_dual_nd_to_nz](test_ub_l1.py) | 与第 21 项共用端到端用例，两个 AIV 从 GM 读入各自半块，再写回同一 L1 |
| 26 | UB→GM 连续/跨步 | [test_ub_to_gm_preserves_guards](test_gm_ub.py) | 与 22 同形状和 dtype。UB 行宽 64，目标上下各一行、左右各八元素预填 −77，核对整个输出 |
| 27 | UB→GM L2 写策略 | [test_ub_to_gm_l2_policy](test_gm_ub.py) | 4×16 FP32。核对对齐写回指令的 `normal_fv`、`normal_lv`、`notalloc_ci`、`notalloc_pw` |
| 28 | UB→GM 软件分半 | [test_l0c_to_ub_dual_m / dual_n](test_fix_output.py) | 与第 18 项共用端到端用例，两个 AIV 分别沿 M 或 N 将各自的 UB 半块写回 GM |
| 29 | UB→GM Atomic Add | [test_ub_to_gm_atomic_add](test_atomic.py) | 2×32→32，两核叠加带符号输入与非零初值。INT8/16/32、FP16/BF16/FP32 |
| 30 | UB→GM Atomic Max/Min | [test_ub_to_gm_atomic_max / min](test_atomic.py) | 同 29，分别与三路逐元素最大、最小值比较。各 dtype 的 A5 硬件结果待验证 |
| 31 | UB→UB Scatter ND→NZ | [test_ub_to_ub_nd_to_nz / inside_simd_vf](test_ub_scatter.py) | 32×128，FP16/BF16/FP32 同类型及 BF16↔FP32。按目标元素宽度解分形，去掉额外一行后比较逻辑矩阵 |

## 运行

从已构建 Ascend 后端的仓库根目录运行。A5-BLUE 实验目录先执行 `source ./test-env.sh`。下面的命令把收集范围限定在 1215 DMA 套件，`--noconftest` 跳过实验目录用于记录历史通过项的钩子。可用 `-k` 按函数名筛选。

```bash
python -m pytest --noconftest testing/ascend/dma_1215 --collect-only -q
python -m pytest --noconftest testing/ascend/dma_1215 -k l2_policy -v
python -m pytest --noconftest testing/ascend/dma_1215/test_gm_l1.py -k test_oob_tail_padding -v
python -m pytest --noconftest testing/ascend/dma_1215 -v --junitxml=/tmp/dma_1215.xml
```

验收记录应同时保留提交版本、设备型号、CANN 和 torch_npu 版本及 pytest 报告。
