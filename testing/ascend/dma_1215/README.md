# A5 DMA 1215 开发自测

以当前 31 项 L3 特性为 1215 验收基准，按下表的 L3 顺序一一对应独立测试文件 review。数值用例运行 kernel 后逐元素比较设备结果与 CPU 参考。L2 Cache 用例核对生成指令中的策略枚举，覆盖 DSL 到 Ascend C API 的参数传递。

GM→L1 的普通 DMA dtype 包括 `int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`float16`、`bfloat16`、`float32` 和 `float8_e4m3fn`。1215 的 FP8 验收范围仅包含 E4M3（`float8_e4m3fn`）。Packed FP4 的设备数值用例走 ND。DN 待核实。数值用例以原 dtype 完成 DMA，再用同位宽的 Cube 支持类型做单位矩阵乘读回 L1。FP4 主用例用 blockscaled GEMM 读回。

## 逐项 review

M、N、K 依次为矩阵乘的行、输出列和归约维。31 项 1215 特性各自对应独立的测试文件：

| ID | 方向与特性 | 用例文件 | shape、dtype 与判据 |
|---|---|---|---|
| 01 | GM→L1 ND→NZ | [test_gm_l1_nd2nz.py](test_gm_l1_nd2nz.py) | 10 种普通 dtype 覆盖 16×32 整块、非 32B 对齐行步长的 (20, 37) GM 界内切片、非整分形 (13, 29) 分配与子范围写入、(32, 64) L1 的分形对齐子区域写入、5D GM 单平面→3D 流水 stage L1、结合 `T.dynamic`/分核 `bx`/循环 `ko`/标量偏移的动态切片及源区域超出目的区域的编译期拒绝。FP4 完整块见第 05 项 |
| 02 | GM→L1 DN→NZ | [test_gm_l1_dn2nz.py](test_gm_l1_dn2nz.py) | 10 种普通 dtype 覆盖 32×16→16×32 整块转置、非 32B 对齐行步长的 (37, 20) GM 界内切片、非整分形 (13, 29) 分配与子范围转置写入、(32, 64) L1 的分形对齐子区域转置写入、5D GM 单平面→3D 流水 stage L1，以及结合 `T.dynamic`/分核/循环/标量偏移的动态转置搬入 |
| 03 | GM→L1 批量 ND/DN→NZ | [test_gm_l1_batch.py](test_gm_l1_batch.py) | 3 个 16×32 矩阵，覆盖紧凑批次、批次间有间隔和交错行，10 种普通 dtype 逐批读回；另测批次起点切片、L1 更大矩阵 pitch、批次越界跳过、行列尾块与相邻矩阵保护区、5D GM→4D L1 外层单平面批量搬入及 4D 双外层 `extent>1` 的编译期拒绝 |
| 04 | GM→L1 越界尾块补零 | [test_gm_l1_oob.py](test_gm_l1_oob.py) | 从列 96 请求完整的 16×32、16×64 或 32×64 NZ tile，另含动态 `col_offset` 越界用例。覆盖仅行尾、仅列尾、双尾与整块越界。16×64 双尾用例有 31 列有效，检查有效 C0 组内剩余列及后续完整 C0 组。32×64 用例只有 4 列有效，读回确认两个行分形都补齐。10 种普通 dtype 逐元素验零，FP4 尾块数值用例待补 |
| 05 | GM→L1 FP4 双值打包 | [test_gm_l1_fp4.py](test_gm_l1_fp4.py) | M=N=16，K=256，E2M1。独立解码高低半字节，覆盖原生 packed dtype、INT8 字节视图及非 C0 对齐行步长的偶数列起点界内切片 |
| 06 | GM→L1 L2 读策略 | [test_gm_l1_l2_policy.py](test_gm_l1_l2_policy.py) | 16×32 目标，10 种普通 dtype 各走 ND 和 DN。FP4 走 16×64 ND。核对指令中的 `normal_fv`、`normal_lv`、`notalloc_clean` |
| 07 | L1→L0A/B 二维加载 | [test_l1_l0_matrix.py](test_l1_l0_matrix.py) | L0A/B 两侧覆盖 10 种普通 dtype 及打包 FP4 的整块加载，另测 L1 分形对齐子块切片、目的 L0 非零起点 `l0_pitch` 拼装、非整分形 `(13, 29)` 逻辑 shape 加载、3D 流水 stage 循环切片加载及非对齐分形起点的编译期拒绝 |
| 08 | L1→L0A/B 转置加载 | [test_l1_l0_transpose.py](test_l1_l0_transpose.py) | 两侧各覆盖 INT8/16/32、UINT8/16/32、FP16、BF16、FP32、FP8 E4M3、FP4 E2M1。普通 dtype 从 64×32 转置为 32×64，单位矩阵读回。L0A 另测 32×32 取 24 列（BF16）和 25 列（FP32），以及 FP32 取 24 列的编译期拒绝。FP4 走 64×64，使用单位 E8M0 缩放读回；另测目的 L0 非零起点 `l0_pitch=2` 分块转置拼装 |
| 09 | L1→L0A/B_MX 缩放因子 | [test_l1_l0_mx_sf.py](test_l1_l0_mx_sf.py) | 56 组全量组合 + 2 组源 L1 scale 子块切片。7 个形状与转置场景分别覆盖主矩阵 FP8 E4M3、打包 FP4 E2M1，以及 A/B 两侧 scale 的 `uint16`、`uint8`、`int8`、`float8_e8m0fnu`。K=128，每行 4 个 E8M0，`uint16` 每元素装两个指数，其余每元素一个。A/B 使用独立 scale，以 CPU 展开后的矩阵乘逐元素校验 |
| 10 | L0C→GM NZ→ND | [test_l0c_gm_nd.py](test_l0c_gm_nd.py) | 32×64 的 FP32 与 INT32 累加结果按 ND 写回，覆盖整块直出、非 32B 对齐 GM 子区域写出与四周保护区、非整分形裁剪 `(20, 37)` + 显式 `l0_pitch=2`，以及结合 `T.dynamic` 与标量偏移写回 5D GM 单平面子区域 |
| 11 | L0C→GM 降精度 | [test_l0c_gm_cast.py](test_l0c_gm_cast.py) | 32×64，FP32→FP16/BF16，覆盖整块写出、非 32B 对齐 GM 子区域写出与四周保护区、非整分形裁剪 `(20, 37)` + `l0_pitch=2`，以及结合 `T.dynamic` 与标量偏移写回 5D GM 单平面子区域 |
| 12 | L0C→GM Atomic Add | [test_l0c_gm_atomic_add.py](test_l0c_gm_atomic_add.py) | 两核写同一 16×16 矩阵，覆盖 FP32、FP16、BF16、INT32，预置初值 7，检查原值与两块 GEMM 结果之和 |
| 13 | L0C→GM L2 写策略 | [test_l0c_gm_l2_policy.py](test_l0c_gm_l2_policy.py) | 32×64 输出，FP32→FP32/FP16/BF16 和 INT32→INT32。核对 Fixpipe 指令的 `normal_fv`、`normal_lv`、`notalloc_ci`、`notalloc_pw` |
| 14 | L0C→UB NZ→ND | [test_l0c_ub_nd.py](test_l0c_ub_nd.py) | 32×64 的 FP32 与 INT32 累加结果转 ND 搬入 UB，另测 32B 对齐宽行距 UB `(32, 80)` 裁剪写出 + `l0_pitch=2` |
| 15 | L0C→UB NZ→NZ | [test_l0c_ub_nz.py](test_l0c_ub_nz.py) | 32×64 FP32 / INT32，UB 显式标注 `make_ascend_l0c_layout`，SIMT 按逻辑坐标读回；另测 16 对齐偏移子区域 `[16:48, 16:80]` |
| 16 | L0C→UB 降精度 | [test_l0c_ub_cast.py](test_l0c_ub_cast.py) | 32×64，FP32→FP16/BF16，单目标 UB，覆盖整块及 32B 对齐宽行距 UB `(32, 80)` 切片写入 |
| 17 | L0C→UB 指定 AIV | [test_l0c_ub_sub_blockid.py](test_l0c_ub_sub_blockid.py) | 两个 AIV 的 UB 预填 −7，逐一指定 `sub_blockid=0/1`，分别回读 16×16 FP32 矩阵验证另一侧保留保护值 |
| 18 | L0C→UB 硬件双写 | [test_l0c_ub_dual.py](test_l0c_ub_dual.py) | 32×64 按 M 分成 16×64、按 N 分成 32×32，覆盖 FP32/INT32 的 ND 写出以及 FP32 的 NZ 保留写出 |
| 19 | UB→L1 连续原样搬入 | [test_ub_l1_contiguous.py](test_ub_l1_contiguous.py) | 32×32，BF16/FP16/FP32 的 CPU 预排紧凑 NZ 块经 AIV 原样搬入 L1，用单位矩阵乘读回校验；INT8 单独验证编译器拒绝该类型 |
| 20 | UB→L1 ND→NZ | [test_ub_l1_nd2nz.py](test_ub_l1_nd2nz.py) | 覆盖全部 5 种支持组合：FP16→FP16、BF16→BF16、FP32→FP32（32×64）及 FP32↔BF16（32×128），单位矩阵乘读回重排结果 |
| 21 | UB→L1 软件双写 | [test_ub_l1_dual.py](test_ub_l1_dual.py) | 同 20，双 AIV 各自通过 `T.dual_copy` 写入共享 L1 的半块，读回完整矩阵验证 |
| 22 | GM→UB 连续/跨步 | [test_gm_ub_copy.py](test_gm_ub_copy.py) | 覆盖 Normal 跨步（1×1、4×15、4×16、4×17，INT8/16/32、FP16/BF16/FP32，UB 行宽 64）、Compact 紧凑搬运、打包 FP4，以及 5D GM 结合 `T.dynamic`/分核 `bx`/标量偏移跨步搬入 3D 流水 UB |
| 23 | GM→UB 右补齐 | [test_gm_ub_padding.py](test_gm_ub_padding.py) | 覆盖 INT8/16/32、FP16/BF16/FP32，`cols=3, 30` 右补齐至 32B 边界，`pad_value` 与 `data_select` 均核对有效列与填充位；另测 `T.dynamic` 行维+动态行起点，以及动态列长的编译期拒绝 |
| 24 | GM→UB L2 读策略 | [test_gm_ub_l2_policy.py](test_gm_ub_l2_policy.py) | 4×32（FP32/BF16/INT8）。核对对齐搬入指令的 `normal_fv`、`normal_lv`、`notalloc_clean` |
| 25 | GM→UB 软件分半 | [test_gm_ub_dual.py](test_gm_ub_dual.py) | 双 AIV 分别按 M（16×64）和 N（32×32）用 `T.dual_copy` 从 GM 读入各自半块（FP32/BF16）并写回校验 |
| 26 | UB→GM 连续/跨步 | [test_ub_gm_copy.py](test_ub_gm_copy.py) | 覆盖 Normal 跨步（UB 行宽 64，目标四周预填 −77 保护区，6 种标量）、Compact 紧凑写回，以及 3D 流水 UB 结合 `T.dynamic`/分核 `bx`/标量偏移跨步写回 5D GM 单平面子区域 |
| 27 | UB→GM L2 写策略 | [test_ub_gm_l2_policy.py](test_ub_gm_l2_policy.py) | 4×32（FP32/BF16/INT8）。核对对齐写回指令的 `normal_fv`、`normal_lv`、`notalloc_ci`、`notalloc_pw` |
| 28 | UB→GM 软件分半 | [test_ub_gm_dual.py](test_ub_gm_dual.py) | 双 AIV 分别沿 M（16×64）或 N（32×32）通过 `T.dual_copy` 将各自的 UB 半块写回 GM |
| 29 | UB→GM Atomic Add | [test_ub_gm_atomic_add.py](test_ub_gm_atomic_add.py) | 1D（2×32→32，6 种标量 dtype）与 2D 跨步子区域保护区原子累加，带符号输入与非零初值 |
| 30 | UB→GM Atomic Max/Min | [test_ub_gm_atomic_max_min.py](test_ub_gm_atomic_max_min.py) | 2×32→32，6 种标量 dtype，分别与三路逐元素最大、最小值比较（各 dtype 的 A5 硬件结果待上卡确认） |
| 31 | UB→UB Scatter ND→NZ | [test_ub_ub_scatter_nd2nz.py](test_ub_ub_scatter_nd2nz.py) | 32×128，FP16/BF16/FP32 同类型及 BF16↔FP32，含显式 `T.SimdVF()` 块内散列。按目标元素宽度解分形，去掉额外一行后比较逻辑矩阵 |

## 运行

从已构建 Ascend 后端的仓库根目录运行。A5-BLUE 实验目录先执行 `source ./test-env.sh`。下面的命令把收集范围限定在 1215 DMA 套件，`--noconftest` 跳过实验目录用于记录历史通过项的钩子。可用 `-k` 按函数名筛选。

```bash
python -m pytest --noconftest testing/ascend/dma_1215 --collect-only -q
python -m pytest --noconftest testing/ascend/dma_1215 -k l2_policy -v
python -m pytest --noconftest testing/ascend/dma_1215/test_gm_l1_oob.py -v
python -m pytest --noconftest testing/ascend/dma_1215 -v --junitxml=/tmp/dma_1215.xml
```

验收记录应同时保留提交版本、设备型号、CANN 和 torch_npu 版本及 pytest 报告。
