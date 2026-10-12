# PA_NZ KV cache：设备对照入口

这是从官方 GatherPaKvCache NZ 路径提取的独立 AscendC 对照，包含页表读取、K/V 两路、序列长度、块偏移和连续输出。不是安装包中完整 aclnn 算子的端到端测试，也不是 vLLM 整网测试。真实消费者与实验结果归档在文档仓 `Work/tilelang-gm-ub-loop-nddma/kv-cache/`。

## 来源与边界

`kv_reference_upstream.h` 是 ops-transformer `0608fbb5ff1e699ddc39878fb3ced63655b2b79b` 的 `attention/gather_pa_kv_cache/op_kernel/arch35/gather_pa_kv_cache_nz.h` 原样副本，许可证见 `kv_reference_LICENSE`。[固定原始文件](https://gitcode.com/cann/ops-transformer/blob/0608fbb5ff1e699ddc39878fb3ced63655b2b79b/attention/gather_pa_kv_cache/op_kernel/arch35/gather_pa_kv_cache_nz.h)。不修改参考文件的算法、队列或同步。

适配器手工提供 tiling 字段，统一使用字节单位，参考 kernel 实例化 `uint8_t` 数据和 `int32_t` 索引；DTYPE_KEY/DTYPE_VALUE 设为 uint8，tiling hiddenSize 对应真实张量的字节宽。结构只供内部调用，不冒充官方序列化 tiling ABI。它保留原始 kernel，但没有复用官方 host tiling、格式元数据分派和 API 连续化。

- 源单块 `[Dbytes/32, block_size, 32]`，目的 `[有效token, Dbytes]`。
- 数据逐字节检查，无数值转换。512B/1024B token 宽可描述 FP8/BF16 的 512 维数据；这不等于测试 Torch FP8 dtype 的整条 API 链路。不支持 FP4 元素抽取或 MX 系数计算。
- 仅连续输出、有效非负页号、块对齐 seq_offset，INT32 范围内长度。未验证负页号、任意 strided 输出、非块对齐 seq_offset、越界读 sanitizer。
- B≤256，K/V token 宽是32B倍数且不超过96KiB，单批次最多32页，最多1024个请求，输入K+V合计≤512MiB，核数1～64。
- 各版本统一每个UB槽96KiB，最多两槽；官方参考另外保留索引临时区。这个固定预算是实验条件，不声称复现官方 host 对全部 UB 的动态分配。
- 各版本按同一连续请求分核。少请求不会用满64核；未将修改分核混入搬运收益。

## 十个版本

| mode | 名称 | 内容 |
| --- | --- | --- |
| 0 | KvOriginal | C API 复现官方控制流：K/V同时放得下时整块NDDMA，否则逐token普通DMA；同步串行 |
| 1 | KvNdSync | 多token分片NDDMA，同步串行 |
| 2 | KvLoopSync | 多token分片普通DMA＋硬件loop，同步串行 |
| 3 | KvNdDouble | 分片NDDMA＋双缓冲 |
| 4 | KvLoopDouble | 普通DMA＋loop＋双缓冲 |
| 5 | KvIndependent | 在mode0上仅把K/V整块判断分开 |
| 6 | KvTail | 在mode0上仅让整块NDDMA读取有效token |
| 7 | KvOfficial | 上述官方冻结kernel，加手工tiling适配器 |
| 8 | KvNdOrderSync | 在mode1上交换NDDMA的token/通道组遍历顺序 |
| 9 | KvNdOrderDouble | 在mode3上交换同样的遍历顺序 |

普通loop覆盖token维，二维DMA覆盖通道组，每个burst32B；两侧loop stride均32B对齐。所有版本都逐物理页查表，不将随机页号当成固定步长。

## 配置与复现

复用 `scenario_bundle.py` 的七个配置字段，但本实验含义为：

```text
[block_size, K_bytes, pages_per_request, V_bytes, tile_tokens, cumsum, tail_mode]
```

`--groups` 表示请求数量；`tail_mode=0/1/2` 分别表示最后一页有效B/1/(B−1)个token（B=1时只测模式0）。tile自动受单槽容量限制。`kv_configs.json` 是初筛；`kv_tune_configs.json` 扩大tile扫描；`kv_boundary_configs.json` 覆盖其他块大小、奇数tile和两种长度语义。

```bash
python kv_bundle.py build --configs kv_tune_configs.json --output NEW_BUILD --device 1
python kv_bundle.py profile --build NEW_BUILD --output NEW_RUN \
  --groups 1 8 64 --blocks 64 --rounds 31 --device 1
python kv_bundle.py build --configs kv_boundary_configs.json --output BOUNDARY_BUILD --device 1
python kv_bundle.py check --build BOUNDARY_BUILD --output BOUNDARY_RUN \
  --groups 1 3 65 --blocks 1 64 --device 1
```

只能在已授权且确认空闲的Ascend950设备运行，先source环境。构建和运行目录必须新建；保存源码、官方头文件、哈希、命令、占用快照和原始msprof输出。

每个配置/请求数/核数组合进行3种输入逐位校验：随机位模式、空/单token/尾块、页表重排及跨请求重复页、非零块偏移。两路输出从前64B guard到全部容量（含未使用尾部）完整比较。20轮预热，测量轮次中轮换版本顺序，最后逐版本复检。性能只取msprof设备任务时间，严格核对每个符号的采样数。重复输入属于热缓存实验，不能据有效字节数声称物理HBM流量。
