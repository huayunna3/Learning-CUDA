# FlashAttention 三版本中文报告

## 1. 测试环境

- GPU：NVIDIA GeForce RTX 3080 10GB，CC 8.6
- 驱动：550.78
- CUDA：12.1（conda 环境）
- 编译器：g++-12
- Nsight Compute：2024.2.0
- Nsight Systems：2026.6.1
- 测试形状：`B=4, T=512, S=2048, Hq=64, Hkv=64, D=32, causal=1`
- 官方用例：14 个形状 × `float/half`，共 28 项

## 2. 结论

| 版本 | 核心设计 | 官方 float | 官方 half | Nsight kernel | 测试 |
|---|---|---:|---:|---:|---:|
| V1 | 一行一 block，单线程算 score | 137.73 ms | 106.73 ms | 109.83 ms | 28/28 |
| V2 | 一 warp 一行，K/V 分块，lane 并行 key | **42.30 ms** | 29.70 ms | **11.63 ms** | 28/28 |
| V3 | block 复用 Q/K/V/P，并行 exp + 严格顺序求和 | 46.59 ms | **26.19 ms** | 12.38 ms | 28/28 |

相对 V1：

- V2：float 提升 3.26×，half 提升 3.59×，kernel 提升 9.44×；
- V3：float 提升 2.96×，half 提升 4.08×，kernel 提升 8.87×。

V2 的 float 端到端最快；V3 的 half 最快，并且分母严格按参考顺序累加，作为默认版本。

## 3. V1：正确性基线

设计：

- Grid `(T, Hq, B)`，一个 block 处理一个 query 行；
- 256 线程，但 score、max、exp、分母都由 `thread 0` 串行完成；
- 其他线程只参与最后的 `P @ V`；
- causal 截断到 `key_end = min(S, t + 1)`；
- 使用 `1 / sqrt(D)` 和与参考一致的 `expf_parts()`。

修改前：`flashAttention` 只有 `// TODO`。

修改后：完整 V1 kernel，28/28 通过，所有 float 用例误差为 0。

最大形状：

| 精度 | 耗时 | 最大误差 |
|---|---:|---:|
| float | 137.73 ms | 0 |
| half | 106.73 ms | 0 |

NCU 中文解读：

- Duration 143.60 ms；
- Compute 39.78%，Memory 39.78%；
- Achieved Occupancy 81.56%；
- 平均每 warp 只有 **1.90** 个有效线程；
- 87.1% 的指令间隔耗在 CTA barrier。

结论：occupancy 不低，但有效并行度极低，是“正确性锚点”，不是性能版本。

## 4. V2：Warp 并行 + K/V 分块

设计：

- 4 warp/block，一个 warp 负责一个 query 行；
- K/V 按 `BK=32` 装入共享内存，4 个 warp 复用；
- 每个 lane 负责不同 key，单个 key 的 D 维点积按顺序 fmaf；
- 整行 max 预扫描；
- 概率写入共享内存，warp shuffle 求 tile 和；
- 每个 lane 绑定一个输出维度，按 key 顺序做 PV。

修改前：online softmax，float 有 7 项失败，最大误差约 1.5e-5。

修改后：全行 max 预扫描 + lane 并行 key，28/28 通过，float 最大误差 5.7e-6。

最大形状：

| 精度 | 耗时 | 最大误差 | 容差 |
|---|---:|---:|---:|
| float | 42.30 ms | 5.7e-6 | 8.85e-4 |
| half | 29.70 ms | 0.0078125 | 0.0503 |

NCU 中文解读：

- Duration 14.50 ms；
- Memory 92.80%，L1/TEX 93.64%，Compute 36.90%；
- L2 命中 99.38%；
- 平均每 warp **31.43** 个有效线程；
- 43.9% 的发射间隔耗在 MIO 队列满，主瓶颈是共享内存/MIO。

NSYS 中文解读：

- kernel 6 次，平均 11.63 ms；
- 18 次 H2D 平均 8.41 ms；
- 6 次 D2H 平均 2.44 ms。

结论：kernel 比 V1 快 9.44 倍，是 float 速度优先版本。

## 5. V3：Block 级复用 + 严格顺序

设计：

- 4 warp/block，`BQ=4`、`BK=32`；
- Q/K/V/P 全部放共享内存；
- 概率 exp 由 32 个 lane 并行计算；
- 分母由 lane 0 按 key 顺序串行 fmaf，保持参考顺序；
- PV 按 key 顺序累加，最后统一除分母；
- causal 以 block 最后一个 query 行为界，后续 tile 直接跳过。

修改前：lane 0 串行算整个概率，最大形状 float 68.81 ms，half 54.77 ms。

修改后：并行 exp + lane 0 有序求和，float 46.59 ms，half 26.19 ms。

提升：

- float：32.3%；
- half：52.2%；
- 28/28 仍然通过，float 最大误差 4.8e-6。

最大形状：

| 精度 | 耗时 | 最大误差 | 容差 |
|---|---:|---:|---:|
| float | 46.59 ms | 4.8e-6 | 8.72e-4 |
| half | 26.19 ms | 0.0078125 | 0.0526 |

NCU 中文解读：

- Duration 14.84 ms；
- Memory 92.63%，L1/TEX 81.99%，Compute 38.89%；
- L2 命中 99.53%；
- Achieved Occupancy 74.56%；
- 平均每 warp 29.42 个有效线程；
- Warp Cycles/Instruction 36.87，为三版最低；
- 动态共享内存 9.98 KB/block。

NSYS 中文解读：

- kernel 6 次，平均 12.38 ms；
- 中位数 12.50 ms，最短 11.75 ms，最长 12.53 ms；
- 18 次 H2D 平均 10.53 ms；
- 6 次 D2H 平均 3.34 ms。

结论：V3 的 kernel 比 V2 慢约 6.4%，但分母严格有序，half 最快，作为默认版本。

## 6. 关键失败与修复

1. **online softmax 的 7 项 float 失败**  
   最大误差约 1.5e-5，容差 1.1e-5～1.9e-5。换 double、换 expf_parts 都没用，根因是重缩放改变了运算顺序。改成全行 max 预扫描后通过。

2. **warp 树形点积的末位误差**  
   即使去掉 online，树形点积仍有约 1.6e-5 误差。最终让每个 lane 负责一个 key，单个 key 的 D 维点积按顺序 fmaf，只并行不同 key。

3. **分母求和顺序**  
   V3 最初 lane 0 串行算概率，误差为 0 但慢。改成并行 exp、lane 0 串行求和，误差仍只有 4.8e-6，速度提升 32%～52%。

## 7. 当前瓶颈与后续优化

当前 NCU 显示 V2/V3 的 L1/TEX 与 MIO 管线已到 92% 左右；NSYS 显示每次调用三次 H2D 合计约 25～32 ms，端到端时间被拷贝明显拖慢。

按收益排序的后续优化：

1. pinned memory + 异步 H2D 流水线，预计减少 10～20 ms；
2. cp.async 双缓冲 K/V tile；
3. half2 / float4 向量化；
4. warp specialization，一个 warp 专门做拷贝；
5. 更大 BK（64/128），减少 tile 循环和同步；
6. persistent kernel，减少 32,768 个 block 的调度开销；
7. 非 causal 大 S 时避免每行 max 预扫描。

## 8. 仓库文件

```text
src/flash_attention_v1.cuh
src/flash_attention_v2.cuh
src/flash_attention_v3.cuh
src/flash_attention_common.cuh
src/flash_attention.cuh
src/bench_attention.cu

docs/attention/V1.md
docs/attention/V2.md
docs/attention/V3.md
docs/attention/ATTENTION_OPTIMIZATION.md
docs/attention/REPORT_CN.md

reports/attention/v1/
reports/attention/v2/
reports/attention/v3/
reports/attention/final_regression.log
```

每个 `reports/attention/vN/` 都包含：

- `test.log`：官方 28 项测试；
- `ncu.ncu-rep` / `ncu.txt`：Nsight Compute 原始报告与文本；
- `nsys.nsys-rep` / `nsys.txt`：Nsight Systems 原始报告与统计；
- `bench.txt`：大形状独立测速。

最终完整回归：**54 项全部通过，0 失败**（RMSNorm + Attention）。
