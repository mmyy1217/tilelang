# Ascend copy 的本地 pass 观察环境

在没有 NPU 的机器上运行真实 Ascend TIR 流水线，记录每个 pass 前后的 IR，并生成 CCE 源码。入口使用 `tilelang.lower(..., enable_host_codegen=False, enable_device_compile=False)`；不启动 kernel，也不调用 bisheng 编译设备二进制。工具会同时进入 Ascend Target context，满足 VF 布局推导的要求。

## 已归档结果

直接阅读 [13 个用例的完整结果与 pass 阅读索引](results/README.md)。无需安装环境。

## 使用

在本 worktree 根目录执行：

```bash
examples/ascend/copy_pass_lab/run.sh --case dma
examples/ascend/copy_pass_lab/run.sh --all
examples/ascend/copy_pass_lab/run.sh --list
```

默认输出放在工作区根目录的 `work/experiments/ascend-copy-pass-lab/runs/<时间戳>/<用例>/`；每次新建目录，不覆盖旧结果。`--out <目录>` 可以指定本次输出的根目录。

每个用例包含：

| 文件 | 内容 |
| --- | --- |
| `pass.log` | 初始 IR、每个 pass 的 BEFORE/AFTER IR（含 metadata）、codegen 输入及最终 CCE 源码；出错时保留异常 |
| `manifest.json` | 源码版本、Python 路径、kernel 哈希、pass 顺序/嵌套关系/日志行号、完成状态及文本是否变化 |
| `passes.tsv` | 简短的 pass 顺序和变化索引，可直接用编辑器查看 |
| `input.tir` | 前端构造出的 kernel |
| `host.tir`、`device.tir` | 流水线结束后分离的 host/device IR |
| `kernel.cpp` | CCE 源码；不是已编译设备二进制 |

```bash
rg -n 'IR AFTER.*(RewriteDualCopy|LayoutInference|InsertNd2Nz|OOBPadding|LowerTileOp)' <用例目录>/pass.log
```

`changed` 比较的是打印出的完整 IR 文本哈希；它是阅读索引，不是形式化的语义等价性判断。pass 次数以实际回调为准，包含重复调用及嵌套 pass，不等于源码里列出的顶层 pass 数。日志是 TVM TIR，不是 MLIR。

## 用例

| 用例名 | 观察目标 |
| --- | --- |
| `dma` | GM→UB→GM 基本 DMA |
| `strided` | 显式 GM stride 与 UB 行距不同 |
| `padding` | pad_value 和行尾补齐 |
| `tail` | 最后一个 block 的 DMA 越界尾块 |
| `dynamic` | 运行时行数和固定中间轴的 stride |
| `gemm` | GM→L1 尾块填充、L1→L0A/B、GEMM、L0C→GM |
| `dual_dma` | 软件 dual_copy 的 sid 分区 |
| `dual_gemm` | L0C→UB 硬件双目标与 UB→GM 软件分区 |
| `nz_scatter` | UB→L1 的 ND→NZ scatter + post-copy；L1 写入只用于观察 lowering |
| `simd`、`simt` | VF 内 UB→UB copy 的不同 lowering |
| `pipeline` | 两版本 UB 与流水循环中的 copy |
| `fp4` | 4-bit 存储的打包地址处理 |

这些用例用于观察编译，不代表已经通过设备上的数值或性能验证。

## 编译自己写的 kernel

写一个 Python 文件，提供无参数函数，返回 `@T.prim_func` 构造的 PrimFunc：

```python
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(src: T.Tensor((64,), "float32"), dst: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.copy(src, ub)
            T.copy(ub, dst)
    return main
```

```bash
examples/ascend/copy_pass_lab/run.sh --kernel /path/to/my_kernel.py:make_kernel
```

不要在这个文件顶层创建 NPU tensor 或调用 JIT kernel，因为导入时会执行顶层代码。可直接修改 `kernels.py` 增加用例，或将自己的文件保存在工作区 `work/experiments/` 下。

## 重建环境

专用 venv 和 build 保存在 `work/experiments/ascend-copy-pass-lab/`；worktree 的 `build` 是指向该目录的符号链接。venv 使用 `--system-site-packages` 只读复用基础环境的 PyTorch，新依赖装入 venv，不修改基础环境。

```bash
COPY_PASS_BASE_PYTHON=/path/to/python3.11 examples/ascend/copy_pass_lab/setup.sh
```

基础 Python 需要有可导入的 PyTorch；系统需要 GCC/G++、CMake ≥3.26 和 Ninja。默认 4 个构建任务，可用 `COPY_PASS_JOBS` 调整。仅启用 Ascend，并使用仓库自带 runtime stub；关闭 CUDA、ROCm、LLVM 和 Metal。G++ 使用 `-O1` 加快本地观察环境的构建。

脚本以进程内 Git 配置跳过本环境不用的 CUDA/OpenCL 子模块，避免 CMake 自动拉取它们；TVM、tvm-ffi、DLPack、libbacktrace 都使用仓库指定版本。不会修改其他 checkout 的源码或子模块更新配置。网络需要镜像时，可临时设置 `PIP_INDEX_URL`。

配置、构建日志和 Python 依赖快照都在上述实验目录。首次构建完成后，修改 Python kernel 无需重编 C++；修改 C++ pass 后，在 worktree 根目录增量编译：

```bash
CCACHE_DIR=../work/experiments/ascend-copy-pass-lab/ccache cmake --build build --parallel 4
```

## 本地验证结果（2026-10-09）

13 个内置用例全部降低到 CCE 源码，共记录 1122 次 pass 回调；一般用例 86 次，gemm/dual_gemm 各 88 次。已逐项核对 BEFORE/AFTER 的配对、日志行号及对应 CCE 指令形式。外部 Python kernel 入口通过，刻意失败的 pass 也正确保留输入和失败状态并返回非零退出码。

验证产物在实验目录的 `runs/verified-context/`，简明结论在 `verification.json`。首次试跑的 VF 用例缺少 Target context，已在观察工具入口补齐；没有修改编译器实现。以上是离线 lowering/codegen 验证，不包含 NPU 数值、设备二进制或性能验证。
