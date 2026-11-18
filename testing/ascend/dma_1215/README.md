# A5 DMA 1215 开发自测

本目录对应验收日期为 1215 的 32 项 DMA 特性。GM→L1 的 7 项验收特性集中在 `test_gm_l1.py`，每项按 ND、DN 或 FP4 路径写独立的测试函数。其他特性按下表定位。

GM→L1 的普通 DMA dtype 包括 `int8`、`uint8`、`int16`、`uint16`、`int32`、`uint32`、`float16`、`bfloat16`、`float32` 和 `float8_e4m3fn`。本阶段 GM→L1 的 FP8 验收只包含 E4M3。FP4 E2M1 打包用例固定使用 ND 路径。数值用例以原 dtype 完成 DMA，再用同位宽的 Cube 支持类型做单位矩阵乘读回 L1。FP4 主用例用 blockscaled GEMM 读回。

**当前验证**　A5-BLUE 收集 463 例，101 个 IR 用例通过，362 个数值用例完成 TileLang 代码生成。此次重构的 21 个代表用例通过 Bisheng 编译抽查。新版设备数值待验证。

**拆分前的 A5 实测**　A5-BLUE（Ascend950PR），分支基线 `d2dbf20c`，CANN 9.2.0-beta.2 运行时，torch_npu 2.10.0.post2。pytest 收集 156 例。33 个 IR 用例全部通过。物理卡 1 上，8 个不同的数值用例通过，1 个失败，其余 114 个尚无确认结果。共享卡让卡后重复执行的用例只计一次。

- GM↔UB 往返通过 7 例，包括 1×1 的全部 6 种 dtype 和 4×15 跨步 INT8。GM→L1 的完整块 ND、FP16 矩阵乘通过 1 例。
- 首个失败为 `test_format_padding[15-16-False]`。FP16 的 15×16 矩阵按 ND 搬入预填 −7 的 16×16 L1，单位矩阵乘读回后，256 个元素中有 16 个与预期不同，最大绝对差为 7，首个最大差位于 (15,0)。本例的目标切片仅覆盖前 15 行，验收断言要求额外一行也清零。测试在首次失败处停止，DN 和列尾的行为仍待实测。
- 远端记录位于 `A5-BLUE:~/work/experiments/dma-1215.4dD9KJ/`。收集结果见 `collect.log`，IR 报告见 `verified-ir.xml`，设备通过清单见 `device-passes.txt`，失败详情见 `pytest-1794974600-10.xml`。

原 `test_batch.py` 保留批次尾块与相邻矩阵保护区的专项回归。

## 逐项 review

M、N、K 依次为矩阵乘的行、输出列和归约维。下表中的“IR”表示断言检查编译器指令或分块表达式，其余为设备数值用例。多项特性可以共用一次端到端测试。

| ID | 方向与特性 | 用例文件 | shape、dtype 与判据 |
|---|---|---|---|
| 01 | GM→L1 ND→NZ | [test_nd_to_nz](test_gm_l1.py) | 10 种普通 dtype 完整搬入 16×32，读回原矩阵。FP4 完整块见第 06 项 |
| 02 | GM→L1 DN→NZ | [test_dn_to_nz](test_gm_l1.py) | 10 种普通 dtype，从连续 32×16 搬到 16×32，与转置参考比较 |
| 03 | GM→L1 批量 ND/DN→NZ | [test_batch_nd_to_nz / test_batch_dn_to_nz](test_gm_l1.py) | 3 个 16×32 矩阵，覆盖紧凑批次、批次间有间隔和交错行。10 种普通 dtype 逐批读回。FP4 检查单条 DMA 的批次参数 |
| 04 | GM→L1 越界尾块补零 | [test_oob_tail_padding_nd / dn](test_gm_l1.py) | 从列 96 请求 16×32 tile，覆盖仅行尾、仅列尾、两侧尾块和整块越界。10 种普通 dtype 逐元素验零，FP4 检查裁剪和填充指令 |
| 05 | GM→L1 ND/DN→NZ 行列格式补零 | [test_nz_format_padding_nd / dn](test_gm_l1.py) | 10 种普通 dtype，源区域全部有效。L1 预填 7，逐元素检查一个 16×C0 分形。C0 为 32、16 或 8，行列各验完整和差一元素 |
| 06 | GM→L1 FP4 双值打包 | [test_fp4_packed_nd_to_nz](test_gm_l1.py) | M=N=16，K=256，E2M1。独立解码高低半字节，覆盖原生 packed dtype 和 INT8 字节视图 |
| 07 | GM→L1 L2 读策略 | [test_gm_to_l1_l2_policy_nd / dn](test_gm_l1.py) | IR，普通 dtype 走 ND/DN，FP4 走 ND。断言 normal_fv、normal_lv、notalloc_clean 枚举 |
| 08 | L1→L0A/B 二维加载 | [test_l1_to_l0_matrix](test_l0_transpose.py) | 显式把 16×32 的 FP16/BF16 矩阵加载到 L0A/B，通过 GEMM 读回 |
| 09 | L1→L0A/B 转置加载 | [l0_transpose](test_l0_transpose.py)、[descriptors](test_descriptors.py) | L0A 数值测试从 32×32 取 24 或 25 列转置，分别用 BF16、FP32。另对 A/B 分别检查 32×16 BF16 转置指令 |
| 10 | L1→L0A/B_MX 缩放因子 | [mx_fp4](test_mx_fp4.py) | FP8 E4M3，K=128，(M,N)=(1,15)/(15,15)/(32,32)/(100,100)/(64,64)。A/B 独立 E8M0，每 32 个 K 元素一组，以 CPU 展开 scale 后的乘积校验 |
| 11 | L0C→GM NZ→ND | [test_l0c_to_gm_nd](test_fix_output.py) | 32×64 的 FP32 累加结果按 ND 写回，和 CPU GEMM 参考比较 |
| 12 | L0C→GM 降精度 | [fix_output](test_fix_output.py)、[descriptors](test_descriptors.py) | 32×64，FP32→FP16/BF16，核对舍入后的矩阵及 IR Cast 字段 |
| 13 | L0C→GM Atomic Add | [atomic](test_atomic.py) | 两核写同一 16×16 FP32 矩阵，预置 7，检查原值与两块 GEMM 结果之和 |
| 14 | L0C→GM L2 写策略 | [descriptors](test_descriptors.py) | IR，16×32 FP32，两种策略，检查 FIX 写缓存字段 |
| 15 | L0C→UB NZ→ND | [test_l0c_to_ub_nd](test_fix_output.py)、[descriptors](test_descriptors.py) | 32×64，UB 再写回 GM 检查行列顺序，另检查 NZ2ND 标志 |
| 16 | L0C→UB NZ→NZ | [fix_nz](test_fix_nz.py) | IR，FP32，源 32×64、64×64、32×128；检查保留 NZ 的标志、物理列距及目标切片偏移 |
| 17 | L0C→UB 降精度 | [fix_output](test_fix_output.py) | 32×64，FP32→FP16/BF16，单目标 UB，回读后与 CPU 转换逐元素比较 |
| 18 | L0C→UB 指定 AIV | [fix_output](test_fix_output.py)、[descriptors](test_descriptors.py) | 两个 AIV 的 UB 预填 −7，逐一指定目标，分别回读 16×16 FP32 矩阵；另对 32×64 的 FP32/FP16/BF16 检查路由字段 |
| 19 | L0C→UB 硬件双写 | [test_l0c_to_ub_dual_m / dual_n](test_fix_output.py)、[fix_nz](test_fix_nz.py) | FP32 的 32×64 按 M 分成 16×64，按 N 分成 32×32，重组后比较完整矩阵；NZ 另查 IR |
| 20 | UB→L1 连续原样搬入 | [descriptors](test_descriptors.py) | IR，INT16[256]，检查原样搬运 512 B 和两端零间隔 |
| 21 | UB→L1 ND→NZ | [test_ub_to_l1_nd_to_nz](test_ub_l1.py) | 32×64 FP32；32×128 FP32→BF16、BF16→FP32。单位矩阵乘读回重排结果 |
| 22 | UB→L1 软件双写 | [test_ub_to_l1_dual_nd_to_nz](test_ub_l1.py)、[dual](test_dual.py) | 同 21，以单 AIV 和双 AIV 对照；另查 FP32 一维半区偏移 |
| 23 | GM→UB 连续/跨步 | [test_gm_to_ub](test_gm_ub.py) | 1×1、4×15、4×16、4×17。INT8/16/32、FP16/BF16/FP32；跨步时源行宽增加 7 元素，UB 行宽 64 |
| 24 | GM→UB 右补齐 | [test_gm_to_ub_pad_value / data_select](test_padding.py) | FP32，2×3→2×8、2×30→2×32，检查有效列及每个填充位置。pad_value 指定 −1；data_select 复用预设的 −1 寄存器值。当前 DSL 的这条路径没有自动取源首元素的语义 |
| 25 | GM→UB L2 读策略 | [descriptors](test_descriptors.py) | IR，16×32 FP32，两种策略，检查读指令参数 |
| 26 | GM→UB 软件分半 | [dual](test_dual.py)、[ub_l1](test_ub_l1.py) | IR 查 FP32 128→64、64×64→32×64 的 AIV 源偏移；双 AIV 重排用例还覆盖从 GM 分半读入的数值路径 |
| 27 | UB→GM 连续/跨步 | [test_ub_to_gm_preserves_guards](test_gm_ub.py) | 与 23 同形状和 dtype。UB 行宽 64，目标上下各一行、左右各八元素预填 −77，核对整个输出 |
| 28 | UB→GM L2 写策略 | [descriptors](test_descriptors.py) | IR，16×32 FP32，两种策略，检查写指令参数 |
| 29 | UB→GM 软件分半 | [dual](test_dual.py)、[fix_output](test_fix_output.py) | IR 查 FP32 64→128、64×32→64×64 的目标偏移；设备用例分别沿 M/N 拼回 32×64 |
| 30 | UB→GM Atomic Add | [atomic](test_atomic.py) | 2×32→32，两核叠加带符号输入与非零初值。INT8/16/32、FP16/BF16/FP32 |
| 31 | UB→GM Atomic Max/Min | [atomic](test_atomic.py) | 同 30，分别与三路逐元素最大、最小值比较。各 dtype 的 A5 硬件结果待验证 |
| 32 | UB→UB Scatter ND→NZ | [test_ub_to_ub_nd_to_nz / inside_simd_vf](test_ub_scatter.py) | 32×128，FP16/BF16/FP32 同类型及 BF16↔FP32。按目标元素宽度解分形，去掉额外一行后比较逻辑矩阵 |

## 运行

在已构建本分支 Ascend 后端，并配好 CANN 与 torch_npu 的 A5 环境，从仓库根目录执行。可按 pytest 函数名筛选单项，例如只看格式补零。

```bash
python -m pytest testing/ascend/dma_1215 --collect-only -q
python -m pytest testing/ascend/dma_1215/test_gm_l1.py -k test_nz_format_padding -v
python -m pytest testing/ascend/dma_1215 -v --junitxml=/tmp/dma_1215.xml
```

验收记录应同时保留提交版本、设备型号、CANN 和 torch_npu 版本及 pytest 报告。缓存策略的用例检查参数传递；实际缓存命中率需另行测量。
