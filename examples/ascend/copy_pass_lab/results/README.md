# copy lowering 学习用例与结果

2026-10-09 扩展为 **70 个用例：62 个离线 lowering 到 CCE 源码，8 个预期拒绝，共 5611 次 pass 回调**。指南 A～L 的主要类别、此前覆盖表中的缺口均已纳入。这里没有穷举全部 dtype、shape、布局和参数组合，也没有执行 NPU 数值/性能验证。

## 从这里开始

按 dma → strided → padding → tail/empty_tail → dynamic_shape → gemm → dual_copy → raw/NZ → VF cast → MX/FP4 → 状态/流水 → 拒绝边界的顺序阅读。每个用例目录都有原 kernel 快照 kernel.py 和学习笔记.md；已讨论的笔记继续追加，其余笔记目前给出目标、验证状态和真实日志入口。

先读 [dma 笔记](verified-context/dma/学习笔记.md)，然后读 [strided 笔记](verified-context/strided/学习笔记.md)。passes.tsv 的 before_line / after_line 指向同目录 pass.log。日志较大时建议 clone 后在编辑器按行号跳转。

## 场景覆盖

| 类别或能力 | 代表用例 |
| --- | --- |
| A/B GM↔UB 连续、pitch、非零起点、padding、尾块 | dma、strided、offset_strided、padding、explicit_pad、tail、empty_tail |
| C GM→L1 ND/DN→NZ 与 tail fill | gemm、gm_l1_transpose |
| D L1→L0A/B、转置、layout major、fractal 子块 | gemm、l1_l0a_transpose、l1_l0b_transpose、l1_l0_layout_transpose、l1_l0_subtile |
| E/F L0C 输出、普通/双目标、dtype 转换 | gemm、l0c_ub、l0c_ub_bf16、l0c_gm_bf16/fp16、dual_gemm、dual_l0c_n |
| G raw UB→L1 | raw_ub_l1、dual_raw_ub_l1 |
| H UB→L1 NZ、cast、单 C0/预打包快路径 | nz_scatter、nz_l1_cast_down/up、nz_l1_single_c0、prepacked_nz_l1 |
| I UB→UB compact NZ、callee、手动 post-copy | compact_nz、compact_nz_cast_down/up、compact_nz_simd、manual_post_copy |
| J SIMD/SIMT copy/cast、SIMT GM 访问、scalar fast path | simd、simt、simd_cast_down/fp16、simt_cast、simt_global、simt_scalar、scalar_global |
| 软件与硬件 dual、M/N 分区 | dual_dma、dual_gemm、dual_l0c_n、dual_raw_ub_l1、dual_nz_l1、dual_prepacked_nz_l1 |
| K MX scales、SF handle、scale 寿命、FP8/FP4 | mx_sf_tile、mx_sf_lifetime、mx_sf_byte、mx_fp4、fp8_e4m3/e5m2、fp4 |
| L fill、pad register、GEMM 内部加载、atomic | gemm、explicit_pad、dual_gemm、atomic_fixpipe、atomic_ub_add/max/min |
| 真正动态 buffer shape / 动态 UB | dynamic_shape、dynamic_ub；dynamic 另用于动态 region |
| 单/双/三/四版本、unit flag、L2 policy | pipeline_1、pipeline、pipeline_3、pipeline_4、unit_flag、cache_policy |
| 不支持边界 | 8 个 reject_* 用例，保留失败日志与明确 diagnostic |

## 逐用例索引

| 用例与学习笔记 | 观察目标 | 验证状态 | 首个建议阅读的 pass |
| --- | --- | --- | --- |
| [atomic_fixpipe](verified-context/atomic_fixpipe/学习笔记.md) | L0C→GM 的原子累加模式及 set/reset 状态依赖。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/atomic_fixpipe/pass.log#L25807)/[后](verified-context/atomic_fixpipe/pass.log#L26296) |
| [atomic_ub_add](verified-context/atomic_ub_add/学习笔记.md) | UB→GM 的原子加 store mode。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/atomic_ub_add/pass.log#L2112)/[后](verified-context/atomic_ub_add/pass.log#L2144) |
| [atomic_ub_max](verified-context/atomic_ub_max/学习笔记.md) | UB→GM 的原子 max store mode。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/atomic_ub_max/pass.log#L2112)/[后](verified-context/atomic_ub_max/pass.log#L2144) |
| [atomic_ub_min](verified-context/atomic_ub_min/学习笔记.md) | UB→GM 的原子 min store mode。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/atomic_ub_min/pass.log#L2112)/[后](verified-context/atomic_ub_min/pass.log#L2144) |
| [cache_policy](verified-context/cache_policy/学习笔记.md) | 显式 L2 load/store 策略在 CCE 参数中的位置。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/cache_policy/pass.log#L1886)/[后](verified-context/cache_policy/pass.log#L1916) |
| [compact_nz](verified-context/compact_nz/学习笔记.md) | 独立 UB ND→UB compact NZ 预打包，多一行物理 padding。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/compact_nz/pass.log#L11189)/[后](verified-context/compact_nz/pass.log#L11478) |
| [compact_nz_cast_down](verified-context/compact_nz_cast_down/学习笔记.md) | UB 预打包融合 float32→bfloat16 转换。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/compact_nz_cast_down/pass.log#L11189)/[后](verified-context/compact_nz_cast_down/pass.log#L11478) |
| [compact_nz_cast_up](verified-context/compact_nz_cast_up/学习笔记.md) | UB 预打包融合 bfloat16→float32 转换。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/compact_nz_cast_up/pass.log#L11189)/[后](verified-context/compact_nz_cast_up/pass.log#L11478) |
| [compact_nz_simd](verified-context/compact_nz_simd/学习笔记.md) | 在已有 SimdVF 中调用 scatter 的 callee 入口。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/compact_nz_simd/pass.log#L11384)/[后](verified-context/compact_nz_simd/pass.log#L11678) |
| [dma](verified-context/dma/学习笔记.md) | GM→UB→GM 的连续区域、元素单位和字节单位。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/dma/pass.log#L1886)/[后](verified-context/dma/pass.log#L1916) |
| [dual_dma](verified-context/dual_dma/学习笔记.md) | GM↔UB 的双 AIV 软件分区。 | 已生成 CCE | tl.RewriteDualCopy [前](verified-context/dual_dma/pass.log#L360)/[后](verified-context/dual_dma/pass.log#L380) |
| [dual_gemm](verified-context/dual_gemm/学习笔记.md) | L0C→UB 的硬件 M 方向双目标与软件写回分区。 | 已生成 CCE | tl.RewriteDualCopy [前](verified-context/dual_gemm/pass.log#L3505)/[后](verified-context/dual_gemm/pass.log#L3710) |
| [dual_l0c_n](verified-context/dual_l0c_n/学习笔记.md) | N 方向硬件双目标与每行写回的 sid 偏移。 | 已生成 CCE | tl.RewriteDualCopy [前](verified-context/dual_l0c_n/pass.log#L3505)/[后](verified-context/dual_l0c_n/pass.log#L3710) |
| [dual_nz_l1](verified-context/dual_nz_l1/学习笔记.md) | 普通 UB ND→L1 NZ 的软件 dual 分区。 | 已生成 CCE | tl.RewriteDualCopy [前](verified-context/dual_nz_l1/pass.log#L3386)/[后](verified-context/dual_nz_l1/pass.log#L3584) |
| [dual_prepacked_nz_l1](verified-context/dual_prepacked_nz_l1/学习笔记.md) | 预打包 UB NZ→L1 的 dual_copy 分区。 | 已生成 CCE | tl.RewriteDualCopy [前](verified-context/dual_prepacked_nz_l1/pass.log#L4899)/[后](verified-context/dual_prepacked_nz_l1/pass.log#L5186) |
| [dual_raw_ub_l1](verified-context/dual_raw_ub_l1/学习笔记.md) | raw UB→L1 的 sid 分区与目的偏移。 | 已生成 CCE | tl.RewriteDualCopy [前](verified-context/dual_raw_ub_l1/pass.log#L360)/[后](verified-context/dual_raw_ub_l1/pass.log#L380) |
| [dynamic](verified-context/dynamic/学习笔记.md) | 运行时 region extent 与跨固定中间轴的 stride。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/dynamic/pass.log#L2075)/[后](verified-context/dynamic/pass.log#L2108) |
| [dynamic_shape](verified-context/dynamic_shape/学习笔记.md) | T.dynamic buffer shape、动态启动数与静态 UB tile。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/dynamic_shape/pass.log#L2024)/[后](verified-context/dynamic_shape/pass.log#L2056) |
| [dynamic_ub](verified-context/dynamic_ub/学习笔记.md) | T.dynamic buffer shape 与符号 UB 分配长度，容量仍需满足设备限制。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/dynamic_ub/pass.log#L2167)/[后](verified-context/dynamic_ub/pass.log#L2202) |
| [empty_tail](verified-context/empty_tail/学习笔记.md) | 第三个 block 完全越界时，不发出有效 DMA 的执行条件。 | 已生成 CCE | tl.AscendInsertOOBPadding [前](verified-context/empty_tail/pass.log#L916)/[后](verified-context/empty_tail/pass.log#L940) |
| [explicit_pad](verified-context/explicit_pad/学习笔记.md) | 显式 pad register 配置与 data_select 的关联。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/explicit_pad/pass.log#L1999)/[后](verified-context/explicit_pad/pass.log#L2030) |
| [fp4](verified-context/fp4/学习笔记.md) | 64 个 FP4 元素如何成为 32 字节存储。 | 已生成 CCE | tl.RewriteFp4ToFp4x2 [前](verified-context/fp4/pass.log#L2801)/[后](verified-context/fp4/pass.log#L2824) |
| [fp8_e4m3](verified-context/fp8_e4m3/学习笔记.md) | FP8 E4M3 的 GM↔UB 字节搬运，不做数值转换。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/fp8_e4m3/pass.log#L1886)/[后](verified-context/fp8_e4m3/pass.log#L1916) |
| [fp8_e5m2](verified-context/fp8_e5m2/学习笔记.md) | FP8 E5M2 的 GM↔UB 字节搬运。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/fp8_e5m2/pass.log#L1886)/[后](verified-context/fp8_e5m2/pass.log#L1916) |
| [gemm](verified-context/gemm/学习笔记.md) | GM→L1 尾块 fill、显式 L1→L0A/B 和 L0C→GM。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/gemm/pass.log#L35746)/[后](verified-context/gemm/pass.log#L36466) |
| [gm_l1_transpose](verified-context/gm_l1_transpose/学习笔记.md) | copy 的显式转置如何选择 GM→L1 DN→NZ。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/gm_l1_transpose/pass.log#L15313)/[后](verified-context/gm_l1_transpose/pass.log#L15516) |
| [l0c_gm_bf16](verified-context/l0c_gm_bf16/学习笔记.md) | Fixpipe L0C float32→GM bfloat16 的 quant_pre=16。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l0c_gm_bf16/pass.log#L25581)/[后](verified-context/l0c_gm_bf16/pass.log#L26068) |
| [l0c_gm_fp16](verified-context/l0c_gm_fp16/学习笔记.md) | Fixpipe L0C float32→GM float16 的 quant_pre=1。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l0c_gm_fp16/pass.log#L25581)/[后](verified-context/l0c_gm_fp16/pass.log#L26068) |
| [l0c_ub](verified-context/l0c_ub/学习笔记.md) | 普通非 dual 的 L0C→UB Fixpipe 参数及跨核依赖。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l0c_ub/pass.log#L25806)/[后](verified-context/l0c_ub/pass.log#L26302) |
| [l0c_ub_bf16](verified-context/l0c_ub_bf16/学习笔记.md) | Fixpipe L0C float32→UB bfloat16 的 quant_pre=16。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l0c_ub_bf16/pass.log#L25806)/[后](verified-context/l0c_ub_bf16/pass.log#L26302) |
| [l1_l0_layout_transpose](verified-context/l1_l0_layout_transpose/学习笔记.md) | 不写 transpose=True，通过 major layout 差异选择转置。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l1_l0_layout_transpose/pass.log#L24015)/[后](verified-context/l1_l0_layout_transpose/pass.log#L24332) |
| [l1_l0_subtile](verified-context/l1_l0_subtile/学习笔记.md) | 非零 L1 子块起点如何换算成 fractal start/step。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l1_l0_subtile/pass.log#L24015)/[后](verified-context/l1_l0_subtile/pass.log#L24332) |
| [l1_l0a_transpose](verified-context/l1_l0a_transpose/学习笔记.md) | L1→L0A 的显式 transpose 搬运。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l1_l0a_transpose/pass.log#L24015)/[后](verified-context/l1_l0a_transpose/pass.log#L24332) |
| [l1_l0b_transpose](verified-context/l1_l0b_transpose/学习笔记.md) | L1→L0B 的显式 transpose 搬运。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/l1_l0b_transpose/pass.log#L24015)/[后](verified-context/l1_l0b_transpose/pass.log#L24332) |
| [manual_post_copy](verified-context/manual_post_copy/学习笔记.md) | 显式 post-copy API 的参数及物理排布责任。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/manual_post_copy/pass.log#L22516)/[后](verified-context/manual_post_copy/pass.log#L22814) |
| [mx_fp4](verified-context/mx_fp4/学习笔记.md) | FP4 GM/L1/L0 存储、MX scale load 与 blockscaled GEMM。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/mx_fp4/pass.log#L34013)/[后](verified-context/mx_fp4/pass.log#L34772) |
| [mx_sf_byte](verified-context/mx_sf_byte/学习笔记.md) | uint8 scale 存储与 uint16 打包 scale 的几何差异。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/mx_sf_byte/pass.log#L36249)/[后](verified-context/mx_sf_byte/pass.log#L37060) |
| [mx_sf_lifetime](verified-context/mx_sf_lifetime/学习笔记.md) | 一次 scale load 服务多次 data reload 的状态寿命。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/mx_sf_lifetime/pass.log#L34042)/[后](verified-context/mx_sf_lifetime/pass.log#L34802) |
| [mx_sf_tile](verified-context/mx_sf_tile/学习笔记.md) | 每轮加载 MX scales，SF handle 绑定 data L0 地址。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/mx_sf_tile/pass.log#L34013)/[后](verified-context/mx_sf_tile/pass.log#L34772) |
| [nz_l1_cast_down](verified-context/nz_l1_cast_down/学习笔记.md) | UB→L1 NZ 路径融合 float32→bfloat16。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/nz_l1_cast_down/pass.log#L7757)/[后](verified-context/nz_l1_cast_down/pass.log#L7958) |
| [nz_l1_cast_up](verified-context/nz_l1_cast_up/学习笔记.md) | UB→L1 NZ 路径融合 bfloat16→float32。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/nz_l1_cast_up/pass.log#L7757)/[后](verified-context/nz_l1_cast_up/pass.log#L7958) |
| [nz_l1_single_c0](verified-context/nz_l1_single_c0/学习笔记.md) | 一行恰好一个 C0 时跳过 scatter 的快路径。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/nz_l1_single_c0/pass.log#L6743)/[后](verified-context/nz_l1_single_c0/pass.log#L6918) |
| [nz_scatter](verified-context/nz_scatter/学习笔记.md) | UB ND→L1 NZ 的同 dtype scatter 和后续 DMA。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/nz_scatter/pass.log#L7835)/[后](verified-context/nz_scatter/pass.log#L8038) |
| [offset_strided](verified-context/offset_strided/学习笔记.md) | 非零二维 min、显式源 pitch 与独立目的 pitch。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/offset_strided/pass.log#L1886)/[后](verified-context/offset_strided/pass.log#L1916) |
| [padding](verified-context/padding/学习笔记.md) | pad_value 如何设置寄存器并补齐 GM→UB 的行末。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/padding/pass.log#L1999)/[后](verified-context/padding/pass.log#L2030) |
| [pipeline](verified-context/pipeline/学习笔记.md) | 两版本 UB 与循环中的搬运依赖。 | 已生成 CCE | tl.MaterializeMultiBuffer [前](verified-context/pipeline/pass.log#L1707)/[后](verified-context/pipeline/pass.log#L1762) |
| [pipeline_1](verified-context/pipeline_1/学习笔记.md) | num_stages=1 与单 UB 版本。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/pipeline_1/pass.log#L2155)/[后](verified-context/pipeline_1/pass.log#L2190) |
| [pipeline_3](verified-context/pipeline_3/学习笔记.md) | 三版本 UB 的 i%3 地址与同步事件。 | 已生成 CCE | tl.MaterializeMultiBuffer [前](verified-context/pipeline_3/pass.log#L1713)/[后](verified-context/pipeline_3/pass.log#L1774) |
| [pipeline_4](verified-context/pipeline_4/学习笔记.md) | 四版本 UB 的 i&3 地址与同步事件。 | 已生成 CCE | tl.MaterializeMultiBuffer [前](verified-context/pipeline_4/pass.log#L1719)/[后](verified-context/pipeline_4/pass.log#L1786) |
| [prepacked_nz_l1](verified-context/prepacked_nz_l1/学习笔记.md) | 复用已经打包好的 UB NZ，避免第二次 scatter。 | 已生成 CCE | tl.InsertNd2Nz [前](verified-context/prepacked_nz_l1/pass.log#L11228)/[后](verified-context/prepacked_nz_l1/pass.log#L11518) |
| [raw_ub_l1](verified-context/raw_ub_l1/学习笔记.md) | 不做 ND→NZ 的连续 raw UB→L1，payload 按 32 字节计。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/raw_ub_l1/pass.log#L1890)/[后](verified-context/raw_ub_l1/pass.log#L1920) |
| [reject_dma_3d](verified-context/reject_dma_3d/学习笔记.md) | 不能用当前二维 DMA 计划表示的三维 stride。 | 预期拒绝 | [拒绝原因](verified-context/reject_dma_3d/manifest.json) |
| [reject_dma_cast](verified-context/reject_dma_cast/学习笔记.md) | VF 外普通 GM→UB 不支持数值 cast。 | 预期拒绝 | [拒绝原因](verified-context/reject_dma_cast/manifest.json) |
| [reject_dual_cast](verified-context/reject_dual_cast/学习笔记.md) | 硬件 L0C→UB dual_copy 不支持不同 dtype。 | 预期拒绝 | [拒绝原因](verified-context/reject_dual_cast/manifest.json) |
| [reject_fp4_odd](verified-context/reject_fp4_odd/学习笔记.md) | 3 个 FP4 元素不是完整字节。 | 预期拒绝 | [拒绝原因](verified-context/reject_fp4_odd/manifest.json) |
| [reject_raw_payload](verified-context/reject_raw_payload/学习笔记.md) | raw UB→L1 的 20 字节 payload 不满足 32 字节要求。 | 预期拒绝 | [拒绝原因](verified-context/reject_raw_payload/manifest.json) |
| [reject_raw_stride](verified-context/reject_raw_stride/学习笔记.md) | raw UB→L1 不接受非连续源 region。 | 预期拒绝 | [拒绝原因](verified-context/reject_raw_stride/manifest.json) |
| [reject_simd_cast_up](verified-context/reject_simd_cast_up/学习笔记.md) | 普通 SIMD copy 不支持 bfloat16→float32，与 NZ scatter 的能力不同。 | 预期拒绝 | [拒绝原因](verified-context/reject_simd_cast_up/manifest.json) |
| [reject_simd_short](verified-context/reject_simd_short/学习笔记.md) | 普通 SIMD copy 的元素数不足一个 VReg。 | 预期拒绝 | [拒绝原因](verified-context/reject_simd_short/manifest.json) |
| [scalar_global](verified-context/scalar_global/学习笔记.md) | 单元素普通 copy 的 scalar GM store，不是 GM→GM DMA。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/scalar_global/pass.log#L1650)/[后](verified-context/scalar_global/pass.log#L1676) |
| [simd](verified-context/simd/学习笔记.md) | SIMD VF 内同 dtype UB copy 的寄存器加载/存储。 | 已生成 CCE | tl.AscendSimdVFLowerParallel [前](verified-context/simd/pass.log#L2429)/[后](verified-context/simd/pass.log#L2468) |
| [simd_cast_down](verified-context/simd_cast_down/学习笔记.md) | 普通 SIMD VF copy 的 float32→bfloat16 寄存器转换。 | 已生成 CCE | tl.AscendSimdVFLowerParallel [前](verified-context/simd_cast_down/pass.log#L2429)/[后](verified-context/simd_cast_down/pass.log#L2468) |
| [simd_cast_fp16](verified-context/simd_cast_fp16/学习笔记.md) | 普通 SIMD VF copy 的 float32→float16 寄存器转换。 | 已生成 CCE | tl.AscendSimdVFLowerParallel [前](verified-context/simd_cast_fp16/pass.log#L2429)/[后](verified-context/simd_cast_fp16/pass.log#L2468) |
| [simt](verified-context/simt/学习笔记.md) | SIMT VF 内同 dtype copy 的线程分工。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/simt/pass.log#L2738)/[后](verified-context/simt/pass.log#L2780) |
| [simt_cast](verified-context/simt_cast/学习笔记.md) | SIMT copy 中的线程加载、数值转换和存储。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/simt_cast/pass.log#L2738)/[后](verified-context/simt_cast/pass.log#L2780) |
| [simt_global](verified-context/simt_global/学习笔记.md) | SIMT VF 内 GM→UB 变成线程 load/store，而 VF 外 UB→GM 仍是 DMA。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/simt_global/pass.log#L2502)/[后](verified-context/simt_global/pass.log#L2540) |
| [simt_scalar](verified-context/simt_scalar/学习笔记.md) | 单元素 SIMT copy 的线程条件与标量访问。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/simt_scalar/pass.log#L2738)/[后](verified-context/simt_scalar/pass.log#L2780) |
| [strided](verified-context/strided/学习笔记.md) | 源 GM、UB、目的 GM 的行距分别如何进入 DMA 描述符。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/strided/pass.log#L1886)/[后](verified-context/strided/pass.log#L1916) |
| [tail](verified-context/tail/学习笔记.md) | 部分有效尾块如何收缩 copy 的 region。 | 已生成 CCE | tl.AscendInsertOOBPadding [前](verified-context/tail/pass.log#L916)/[后](verified-context/tail/pass.log#L940) |
| [unit_flag](verified-context/unit_flag/学习笔记.md) | GEMM 与 Fixpipe 的 unit_flag_ctrl=3 协议，与软件 flag 区分。 | 已生成 CCE | tl.AscendLowerTileOp [前](verified-context/unit_flag/pass.log#L25539)/[后](verified-context/unit_flag/pass.log#L26024) |

## 来源、快照与复现

- 编译器基线 47976f210de3598444003e9b05de8204dfa7ff88，TVM 4211874e9e4b8770fd20ad0d98ab80afd1df5028；没有修改编译器实现。
- 首批 13 个用例保留原日志与行号；扩展用例来自本地 expanded-final 批次，工具及扩展 kernel 提交 c0fa5050。
- 扩展批次生成于工具提交之前，manifest 的 tilelang_revision 记录当时 HEAD c790465d；kernel_sha256 / kernel_snapshot_sha256 记录实际源码，不能仅凭该 Git 字段反推当时尚未提交的工具文件。
- 原 13 例汇总保留在 verification.json；当前 70 例汇总在 [verification-expanded.json](verification-expanded.json)。验证器核对 pass 前后行号、TSV 索引、源码快照哈希、关键指令/转换模式、预期拒绝，以及不应出现 scatter/DMA 的分支。
- 初始、最终 TIR 与 pass 日志保留原样。kernel.py 是生成时的源模块快照，附带 make_kernel 工厂；共享工厂的具体参数见 manifest 和学习笔记。
- 某些用例只为观察某通路，可能没有消费者或完整数值计算；动态 UB、普通 L0C→UB 等还需设备侧约束验证。成功源码生成不能代替设备编译与运行。

在实验 worktree 根目录验证归档（只读取文本，不加载编译器）：

```bash
python examples/ascend/copy_pass_lab/verify_results.py
```

复现某个目录的 kernel：

```bash
examples/ascend/copy_pass_lab/run.sh --kernel examples/ascend/copy_pass_lab/results/verified-context/strided/kernel.py:make_kernel
```

预期拒绝用例用 --case reject_* 运行时，只有匹配预设 diagnostic 才算观察成功；manifest 仍记 failed。用 --kernel 直接运行同一非法 kernel 会正常返回非零退出码。重建环境与批量生成见[工具说明](../README.md)。
