# FlashAttention 优化总览

本文汇总第二题 FlashAttention 从基线到最终版的设计、实测结果、失败尝试和后续优化方向。所有关键计算均为自写 CUDA kernel，没有使用 cuBLAS、PyTorch SDPA 等库函数代替。

## 1. 结论速览

测试形状：`B=4, T=512, S=2048, Hq=64, Hkv=64, D=32, causal=1`，RTX 3080 10GB。

| 版本 | 核心设计 | 官方测试 float | 官方测试 half | Nsight kernel | 官方测试 |
|---|---|---:|---:|---:|---:|
| V1 | 一行一 block，单线程算整行 score | 137.73 ms | 106.73 ms | 109.83 ms | 28/28 |
| V2 | 一 warp 一行，K/V 分块，lane 并行 key | **42.30 ms** | 29.70 ms | **11.63 ms** | 28/28 |
| V3 | block 复用 Q/K/V/P，并行 exp + 严格顺序求和 | 46.59 ms | **26.19 ms** | 12.38 ms | 28/28 |

相对 V1：

- V2 官方测试 float 提升 **3.26×**，half 提升 **3.59×**，kernel 提升 **9.44×**；
- V3 官方测试 float 提升 **2.96×**，half 提升 **4.08×**，kernel 提升 **8.87×**。

V2 的 float 端到端最快；V3 的 half 最快，且分母严格按参考顺序累加，作为默认版本。

## 2. 版本演进

### V1：先把正确性钉死

- Grid `(T, Hq, B)`，一个 block 负责一个 query 行；
- `thread 0` 串行计算整行 score、max、exp 和分母；
- 所有线程只参与最后的 `P @ V`；
- causal 截断到 `key_end = min(S, t + 1)`；
- 使用与参考一致的 `expf_parts()` 和 `1 / sqrt(D)`。

结果：28/28 通过，所有 float 用例误差为 0。代价是每个 warp 只有 1.90 个线程有效，NCU 显示 87.1% 的指令间隔耗在 CTA barrier。

### V2：把并行度放出来

- 一个 block 4 个 warp，每个 warp 负责一个 query 行；
- K/V 按 `BK=32` 装共享内存，block 内 4 个 warp 复用；
- 每个 lane 负责不同 key，顺序做 `D` 维点积；
- 整行 max 预扫描，消除 online softmax 的反复重缩放；
- 概率写入共享内存 `p_s`，warp shuffle 求 tile 和；
- 每个 lane 绑定一个输出维度，按 key 顺序累加 `P @ V`。

结果：28/28 通过，float 最大误差 5.7e-6，kernel 从 109.83 ms 降到 11.63 ms。

### V3：block 级复用 + 严格求和顺序

- 保持 4 warp/block、`BK=32`；
- `Q`、`K`、`V`、`P` 全部放共享内存；
- 概率 `exp` 由 32 个 lane 并行计算；
- 分母由 `lane 0` 按 key 顺序串行累加，保持参考实现的求和顺序；
- PV 仍按 key 顺序累加，最后统一除分母。

V3 还把旧版 `lane 0` 串行算概率的瓶颈改掉：最大形状 float 从 68.81 ms 降到 46.59 ms（32.3%），half 从 54.77 ms 降到 26.19 ms（52.2%）。

## 3. 实测对比

### 3.1 官方 28 项测试

| 版本 | 通过数 | float 最大误差 | half 最大误差 |
|---|---:|---:|---:|
| V1 | 28/28 | 0 | 0 |
| V2 | 28/28 | 5.7e-6 | 0.0078125 |
| V3 | 28/28 | 4.8e-6 | 0.0078125 |

### 3.2 Nsight Compute 关键指标

| 指标 | V1 | V2 | V3 | 中文含义 |
|---|---:|---:|---:|---|
| Duration | 143.60 ms | 14.50 ms | 14.84 ms | NCU 回放单次 kernel |
| Compute (SM) | 39.78% | 36.90% | 38.89% | SM 计算吞吐 |
| Memory Throughput | 39.78% | 92.80% | 92.63% | 内存/MIO 管线利用率 |
| L1/TEX Hit Rate | 94.74% | 81.73% | 81.99% | L1 命中率 |
| L2 Hit Rate | 98.63% | 99.38% | 99.53% | L2 命中率 |
| Achieved Occupancy | 81.56% | 74.58% | 74.56% | 实际占用率 |
| Avg. Active Threads/Warp | 1.90 | 31.43 | 29.42 | 每个 warp 有效线程数 |
| Warp Cycles/Instruction | 48.72 | 39.92 | 36.87 | 指令发射间隔 |
| Registers/Thread | 44 | 56 | 56 | 每线程寄存器数 |

解读：

- V1 的问题不是 occupancy，而是有效线程太少；
- V2/V3 的 occupancy 略低，但有效线程从 1.9 升到约 30，实际吞吐高一个数量级；
- V2/V3 的主瓶颈已经变成 L1/TEX 与 MIO 管线（约 92%），共享内存访问压力接近饱和；
- V3 的 Warp Cycles/Instruction 最低，但 lane 0 串行求和带来长依赖链，kernel 比 V2 慢约 6%。

### 3.3 Nsight Systems

| 项目 | V1 | V2 | V3 |
|---|---:|---:|---:|
| kernel 平均 | 109.83 ms | 11.63 ms | 12.38 ms |
| kernel 中位数 | 105.39 ms | 12.19 ms | 12.50 ms |
| 18 次 H2D 平均 | 10.50 ms | 8.41 ms | 10.53 ms |
| 6 次 D2H 平均 | 3.27 ms | 2.44 ms | 3.34 ms |

V2/V3 的 kernel 已经压到 12 ms 左右，但每次调用仍有三次 H2D，合计约 25–32 ms。端到端测试时间（42–47 ms）里，拷贝占了很大比例。

## 4. 做过的优化

1. **GQA 下标**：`hkv = hq / (Hq / Hkv)`，确保多个 query head 正确共享 KV head。
2. **causal 左上角语义**：按 `q_idx` 截断 `row_key_end`，不是简单的方形下三角。
3. **warp-per-query**：V2/V3 让一个 warp 负责一行，替代 V1 的单线程 score。
4. **K/V tile 共享**：按 `BK=32` 装入共享内存，block 内 4 个 warp 复用，L2 命中 99% 以上。
5. **共享内存 padding**：`stride = D + 1`，打散 bank，降低 32 lane 同时访问同一 bank 的冲突。
6. **全行 max 预扫描**：替代 online softmax 的反复 alpha 重缩放，float 精度稳定。
7. **参考一致的 exp**：`expf_parts()` 使用 `ex2.approx.ftz` 和高低位数拆分，float 误差压到 5e-6 以下。
8. **并行 exp + 有序求和**：V3 让 32 个 lane 并行算 exp，lane 0 只按顺序累加，兼顾速度和数值顺序。
9. **causal tile 跳过**：`block_key_end` 之后直接结束循环，避免无效 K/V 读取。
10. **host 端显存缓存**：`FlashDeviceCache` 复用 Q/K/V/O 显存，避免每次调用重新 cudaMalloc。

## 5. 失败尝试与修复

### 5.1 online softmax 的 7 项 float 失败

最初 V2 使用逐 tile 更新：

```cpp
new_max = max(row_max, score);
alpha = exp(row_max - new_max);
p = exp(score - new_max);
row_sum = row_sum * alpha + p;
out = out * alpha + p * v;
```

结果：**21/28**，7 个 float 用例失败，最大误差约 `1.5e-5`，容差只有 `1.1e-5 ~ 1.9e-5`。

尝试过把 double 累加改成 float、把 expf 换成 expf_parts、把 PV 累加改成 double，都没有消除误差。根因是 online 重缩放改变了参考实现的运算顺序，不是精度不够。

修复：改成全行 max 预扫描，概率只算一次，PV 按 key 顺序累加。修复后 28/28。

### 5.2 点积求和顺序

即使去掉 online 重缩放，warp 树形点积仍会让 float 误差在 `1.6e-5` 左右。V2 最终让每个 lane 负责一个 key，单个 key 的 D 维点积按顺序 fmaf，只把不同 key 分配给不同 lane。这样既并行，又保持了参考的点积顺序。

### 5.3 分母求和顺序

V3 最初让 lane 0 串行算整个概率，虽然误差为 0，但速度慢。改成并行算 exp、lane 0 串行求和后，误差仍只有 4.8e-6，速度提升 32%–52%。

## 6. 当前瓶颈

1. **H2D 拷贝**：每次调用 Q/K/V 三次拷贝，约 25–32 ms，已经接近 kernel 时间；
2. **L1/TEX 与 MIO 管线**：NCU 显示 92% 左右，共享内存读写是主瓶颈；
3. **lane 0 串行求和**：V3 的分母依赖链，导致 kernel 比 V2 慢约 6%；
4. **重复 max 预扫描**：每个 query block 仍要单独扫描 K，causal 下虽然截断，但非 causal 大 S 时开销明显。

## 7. 后续可做的优化

按预期收益排序：

1. **pinned memory + 异步 H2D 流水线**：把三次输入拷贝与 kernel 重叠，端到端最直接，预计能吃掉 10–20 ms；
2. **cp.async 双缓冲 K/V tile**：让下一块 K/V 的装载与当前块计算重叠，进一步压 MIO 等待；
3. **half2 / float4 向量化**：K/V 的全局读取和共享内存写入改成 128-bit，减少指令数；
4. **warp specialization**：一个 warp 专门做拷贝，其他 warp 做计算，适合更大 BK；
5. **更大的 BK（64/128）**：减少 tile 循环和 __syncthreads 次数，但会增加共享内存占用；
6. **persistent kernel**：减少 32,768 个 block 的调度开销，适合固定形状推理；
7. **避免每行 max 预扫描**：非 causal 大 S 时可以用分块 max + 修正，减少一次 K 读取。

注意：当前测试的 Avg Time 包含 host-device 拷贝，单看 kernel 会低估拷贝的影响；优化端到端时应优先处理 H2D，而不是只盯 kernel。

## 8. 复现命令

```bash
# 构建默认版本（V3）
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12 \
  -I/root/anaconda3/targets/x86_64-linux/include" make build

# 运行 Attention 测试
SKIP_RMS_NORM=1 ./test_kernels --verbose

# 单独构建某个版本
NVCC_PREPEND_FLAGS="-DFLASH_ATTENTION_VERSION=2 \
  -ccbin /usr/bin/g++-12 \
  -I/root/anaconda3/targets/x86_64-linux/include" make build

# 大形状 benchmark
nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA \
  -DFLASH_ATTENTION_VERSION=3 \
  -ccbin /usr/bin/g++-12 \
  -I/root/anaconda3/targets/x86_64-linux/include \
  -o /tmp/bench_attn_v3 src/bench_attention.cu
/tmp/bench_attn_v3 3
```

## 9. 文件索引

```text
src/flash_attention.cuh          # 版本选择与 host 端缓存
src/flash_attention_common.cuh   # expf_parts 与显存缓存结构
src/flash_attention_v1.cuh       # V1 正确性基线
src/flash_attention_v2.cuh       # V2 warp 并行 + K/V 分块
src/flash_attention_v3.cuh       # V3 block 复用 + 严格顺序
src/bench_attention.cu           # 大形状独立测速

reports/attention/v1/            # V1 测试、NCU、NSYS
reports/attention/v2/            # V2 测试、NCU、NSYS、失败尝试日志
reports/attention/v3/            # V3 测试、NCU、NSYS、旧版对比日志

docs/attention/V1.md
docs/attention/V2.md
docs/attention/V3.md
docs/attention/ATTENTION_OPTIMIZATION.md
```
