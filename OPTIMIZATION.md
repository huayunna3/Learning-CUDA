# RMSNorm CUDA 优化记录

## 1. 环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3080，10 GB |
| 架构 | CC 8.6，68 SM |
| 驱动 | 550.78 |
| CUDA | 12.1.105 |
| 编译器 | nvcc 12.1 + g++-12 |
| 系统 | Ubuntu 24.04 |
| 测试规模 | Batch=4096，hidden_dim=4096，float |
| 性能工具 | Nsight Systems 2026.6.1 / Nsight Compute 2024.2 |

## 2. 结果总览

| 版本 | Kernel 耗时 | 相对基线 |
|---|---:|---:|
| 朴素 RMSNorm | 0.290 ms | — |
| 标量寄存器缓存 | 0.198 ms | -31.7% |
| float4 寄存器缓存 | **0.196 ms** | **-32.4%** |

关键指标：

- 全局访存从 3 遍降到 2 遍。
- 估算数据搬运量从约 195.7 MB 降到约 133.1 MB，减少约 32%。
- ncu 执行指令数从 14,843,904 降到 4,685,824，减少约 68.4%。
- float4 版本 DRAM 利用率达到 92.12%，已接近显存带宽上限。
- 每线程寄存器从 16 增加到 34，但实际占用率仍为 94.82%。

Host 端优化：

| 版本 | Host 端到端 |
|---|---:|
| 全同步 cudaMemcpy | 约 35.0 ms |
| 异步流 + 输出锁页 | 约 32.9 ms |
| 异步 + 设备显存复用 | 断线前最好约 30.1 ms |

说明：服务器重启后 Host 侧测得约 35.5~40.5 ms，存在明显环境波动；Kernel 始终稳定在 0.196 ms 左右。

## 3. 优化过程

### 3.1 寄存器缓存

基线实现中，每个线程先读一遍输入求平方和，归约结束后再从全局内存读取输入计算输出，共 3 遍访存：

1. 读 x 求平方和；
2. 再读 x；
3. 写 y。

优化后，每个线程把自己负责的 ITEMS 个元素保存在寄存器数组中，第二遍直接复用：

```cpp
float cache[ITEMS];
float local_sum = 0.0f;

#pragma unroll
for (int k = 0; k < ITEMS; ++k) {
    float val = to_float(row_in[tid + k * BLOCK_SIZE]);
    cache[k] = val;
    local_sum = fmaf(val, val, local_sum);
}
```

效果：

- Kernel：0.290 ms -> 0.198 ms；
- 全局访存：3 遍 -> 2 遍；
- 提升：约 31.7%。

### 3.2 float4 向量化

float 路径增加 float4 版本，每个线程缓存 `ITEMS` 个 float4：

```cpp
float4 cache[ITEMS];

#pragma unroll
for (int k = 0; k < ITEMS; ++k) {
    float4 v = row_in[tid + k * BLOCK_SIZE];
    cache[k] = v;
    local_sum = fmaf(v.x, v.x, local_sum);
    local_sum = fmaf(v.y, v.y, local_sum);
    local_sum = fmaf(v.z, v.z, local_sum);
    local_sum = fmaf(v.w, v.w, local_sum);
}
```

写回同样使用 128-bit 存储：

```cpp
float4 w = weight4[j];
float4 o;
o.x = cache[k].x * rms_inv * w.x;
o.y = cache[k].y * rms_inv * w.y;
o.z = cache[k].z * rms_inv * w.z;
o.w = cache[k].w * rms_inv * w.w;
row_out[j] = o;
```

D=4096 时：

- 每行 1024 个 float4；
- 每个 block 256 线程；
- 每个线程处理 4 个 float4，即 16 个 float。

效果：

- Kernel：0.198 ms -> 0.196 ms；
- 提升约 1%；
- 执行指令数减少 18.8%；
- 由于已经是 2 遍访存，向量化主要减少地址计算和加载指令，无法突破 DRAM 上限。

### 3.3 异步拷贝与输出锁页

旧版本：

```cpp
cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);
kernel<<<...>>>();
cudaDeviceSynchronize();
cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);
```

新版本：

```cpp
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

cudaMemcpyAsync(d_input, h_input.data(), input_bytes,
                cudaMemcpyHostToDevice, stream);
kernel<<<rows, BLOCK_SIZE, 0, stream>>>(...);
cudaMemcpyAsync(h_output.data(), d_output, input_bytes,
                cudaMemcpyDeviceToHost, stream);
cudaStreamSynchronize(stream);
```

同时，在调用期间对输出向量执行临时锁页：

```cpp
bool output_registered =
    cudaHostRegister(h_output.data(), input_bytes,
                     cudaHostRegisterDefault) == cudaSuccess;

// ...

if (output_registered) {
    cudaHostUnregister(h_output.data());
}
```

效果：

- D2H 平均耗时从 18.26 ms 降到 7.27 ms，降低约 60%；
- Host 端到端从约 35.0 ms 降到约 32.9 ms；
- 输入也锁页时 pin/unpin 成本过高，最终只锁定输出。

### 3.4 设备显存复用

每次调用重新 cudaMalloc 会产生约 7 ms 开销，因此使用线程本地缓存：

```cpp
template <typename T>
struct RmsNormDeviceCache {
    T *input = nullptr;
    T *output = nullptr;
    T *weight = nullptr;
    size_t input_cap = 0;
    size_t weight_cap = 0;
};

static thread_local RmsNormDeviceCache<T> cache;
```

只有容量不足时才扩容，容量足够时直接复用：

```cpp
if (cache.input_cap < total) {
    if (cache.input) cudaFree(cache.input);
    if (cache.output) cudaFree(cache.output);
    cudaMalloc(&cache.input, input_bytes);
    cudaMalloc(&cache.output, input_bytes);
    cache.input_cap = total;
}
```

效果：

- 消除了重复 cudaMalloc/cudaFree；
- Host 端到端断线前最好从约 32.9 ms 降到约 30.1 ms。

### 3.5 尝试但放弃的方案

| 方案 | 结果 | 原因 |
|---|---|---|
| 输入和输出都 cudaHostRegister | 约 33.9 ms | 输入锁页和解除锁页的成本超过 H2D 收益 |
| cudaMallocAsync + 默认内存池 | 34.7~37.0 ms | 驱动异步分配器开销没有被摊薄，反而变慢 |

## 4. Profiling 数据

### 4.1 ncu：float4 kernel

| 指标 | 数值 |
|---|---:|
| Kernel Duration | 195.52 us |
| Memory Throughput | 680.90 GB/s |
| DRAM Throughput | 92.12% |
| Compute (SM) Throughput | 13.46% |
| L1/TEX Throughput | 25.69% |
| L2 Throughput | 40.39% |
| Registers Per Thread | 34 |
| Theoretical Occupancy | 100% |
| Achieved Occupancy | 94.82% |
| Executed Instructions | 4,685,824 |
| Waves Per SM | 10.04 |
| Long Scoreboard | 63.3 cycles |

结论：

- kernel 已接近显存带宽上限；
- 继续优化计算单元或减少 FMA 对总耗时影响很小；
- ncu 估计的本地理论剩余提升约 7.88%；
- 真正的耗时大头仍在 Host 端数据搬运。

### 4.2 nsys：Kernel 与拷贝

| 项目 | 数值 |
|---|---:|
| rmsnorm_naive | 289.01 us |
| rmsnorm_cached<float, 16> | 197.39 us |
| rmsnorm_cached_float4<4> | 195.50 us |
| H2D 平均 | 约 8.21 ms |
| D2H 优化前平均 | 18.26 ms |
| D2H 优化后平均 | 7.27 ms |

D2H 提升来自输出锁页。输入保持普通页内存，是因为每次 cudaHostRegister/cudaHostUnregister 的开销超过 H2D 收益。

## 5. 测试结果

使用以下命令跳过尚未实现的 Attention 测试：

```bash
cd /root/Learning-CUDA
SKIP_ATTENTION=1 ./test_kernels
```

结果：

- Test #1 ~ #13：全部 Passed；
- 每个测试点包含 float 和 half；
- 共 26 项校验全部通过；
- Attention 测试仍失败，因为 flashAttention 尚未实现。

## 6. 复现命令

编译：

```bash
cd /root/Learning-CUDA
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12 -I/root/anaconda3/targets/x86_64-linux/include" make build
```

测试：

```bash
SKIP_ATTENTION=1 ./test_kernels
```

基准：

```bash
NVCC_PREPEND_FLAGS="-ccbin /usr/bin/g++-12 -I/root/anaconda3/targets/x86_64-linux/include" \
  nvcc -std=c++17 -O0 -DPLATFORM_NVIDIA src/bench.cu -o bench
./bench
```

## 7. 后续优化方向

### 高收益但需要接口配合

1. 输入输出使用持久 pinned buffer；
2. 提供设备常驻接口，避免每次 H2D/D2H；
3. 将 RMSNorm 与前后算子融合，消除中间数据搬运；
4. 分块双缓冲流水线，让 CPU 拷贝与 GPU DMA 重叠。

### Kernel 内部的小收益项

1. 使用 warp shuffle 替代 shared memory 树形归约，减少 __syncthreads，预计 1~3%；
2. 为 half 路径增加 half2 或 16 字节向量化；
3. 多行合并到一个 block，减少权重重复读取和调度开销。

当前结论：Kernel 已达到 92.12% DRAM 利用率，继续优化的空间很小；Host 端数据搬运才是主要瓶颈。

## 8. 备注

- 服务器重启后 Host 端时间有明显波动，Kernel 时间保持稳定。
- 设备缓冲区采用 thread_local 缓存，便于重复调用复用。
- 测试环境中的 tester_nv.o 为适配驱动 550 做过本地二进制修改，不在本次提交范围内。
