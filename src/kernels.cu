#include <vector>
#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../tester/utils.h"

#define BLOCK_SIZE 256


__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(half x) { return __half2float(x); }

__device__ __forceinline__ void from_float(float &dst, float v) { dst = v; }
__device__ __forceinline__ void from_float(half &dst, float v) { dst = __float2half_rn(v); }

#include "flash_attention.cuh"

// 通用兜底版本：不缓存 x，适用于任意 D
template <typename T>
__global__ void rmsnorm_naive(const T *input, const T *weight, T *output,
                              int Batch, int D, float eps) {
  int bid = blockIdx.x;

  float local_sum = 0.0f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float val = to_float(input[bid * D + i]);
    local_sum += val * val;
  }

  __shared__ float shared_sum[BLOCK_SIZE];
  shared_sum[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
    }
    __syncthreads();
  }

  float rms_inv = rsqrtf(shared_sum[0] / static_cast<float>(D) + eps);
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float val = to_float(input[bid * D + i]);
    from_float(output[bid * D + i], val * rms_inv * to_float(weight[i]));
  }
}

// 寄存器缓存版本：要求 D == ITEMS * BLOCK_SIZE
// 每个线程把自己负责的 ITEMS 个元素留在寄存器里，第二遍不再读全局内存
template <typename T, int ITEMS>
__global__ void rmsnorm_cached(const T *input, const T *weight, T *output,
                               int D, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const T *row_in = input + static_cast<size_t>(bid) * D;
  T *row_out = output + static_cast<size_t>(bid) * D;

  float cache[ITEMS];
  float local_sum = 0.0f;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    float val = to_float(row_in[tid + k * BLOCK_SIZE]);
    cache[k] = val;
    local_sum = fmaf(val, val, local_sum);
  }

  __shared__ float shared_sum[BLOCK_SIZE];
  shared_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  float rms_inv = rsqrtf(shared_sum[0] / static_cast<float>(D) + eps);
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int j = tid + k * BLOCK_SIZE;
    from_float(row_out[j], cache[k] * rms_inv * to_float(weight[j]));
  }
}

// float4 版本：每个线程缓存 ITEMS 个 float4，一次访存 16 字节
template <int ITEMS>
__global__ void rmsnorm_cached_float4(const float *input, const float *weight,
                                      float *output, int D, float eps) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const float4 *row_in = reinterpret_cast<const float4 *>(
      input + static_cast<size_t>(bid) * D);
  float4 *row_out = reinterpret_cast<float4 *>(
      output + static_cast<size_t>(bid) * D);
  const float4 *weight4 = reinterpret_cast<const float4 *>(weight);

  float4 cache[ITEMS];
  float local_sum = 0.0f;

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    float4 v = row_in[tid + k * BLOCK_SIZE];
    cache[k] = v;
    local_sum = fmaf(v.x, v.x, local_sum);
    local_sum = fmaf(v.y, v.y, local_sum);
    local_sum = fmaf(v.z, v.z, local_sum);
    local_sum = fmaf(v.w, v.w, local_sum);
  }

  __shared__ float shared_sum[BLOCK_SIZE];
  shared_sum[tid] = local_sum;
  __syncthreads();

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
    }
    __syncthreads();
  }

  float rms_inv = rsqrtf(shared_sum[0] / static_cast<float>(D) + eps);

#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    int j = tid + k * BLOCK_SIZE;
    float4 w = weight4[j];
    float4 o;
    o.x = cache[k].x * rms_inv * w.x;
    o.y = cache[k].y * rms_inv * w.y;
    o.z = cache[k].z * rms_inv * w.z;
    o.w = cache[k].w * rms_inv * w.w;
    row_out[j] = o;
  }
}

template <typename T, int ITEMS>
static void launch_cached(const T *d_input, const T *d_weight, T *d_output,
                          size_t rows, int D, float eps, cudaStream_t stream) {
  rmsnorm_cached<T, ITEMS><<<static_cast<int>(rows), BLOCK_SIZE, 0, stream>>>(
      d_input, d_weight, d_output, D, eps);
}

template <int ITEMS>
static void launch_cached_float4(const float *d_input, const float *d_weight,
                                 float *d_output, size_t rows, int D,
                                 float eps, cudaStream_t stream) {
  rmsnorm_cached_float4<ITEMS><<<static_cast<int>(rows), BLOCK_SIZE, 0, stream>>>(
      d_input, d_weight, d_output, D, eps);
}

template <typename T>
struct RmsNormDeviceCache {
  T *input = nullptr;
  T *output = nullptr;
  T *weight = nullptr;
  size_t input_cap = 0;
  size_t weight_cap = 0;
};

template <typename T>
void rmsNorm(const std::vector<T> &h_input, const std::vector<T> &h_weight,
             std::vector<T> &h_output, size_t rows, size_t hidden_dim,
             float eps) {
  if (rows == 0 || hidden_dim == 0) return;

  const size_t total = rows * hidden_dim;
  const size_t input_bytes = total * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);

  // 每个线程缓存一组设备缓冲区，容量不足时才扩容
  static thread_local RmsNormDeviceCache<T> cache;
  if (cache.input_cap < total) {
    if (cache.input) cudaFree(cache.input);
    if (cache.output) cudaFree(cache.output);
    cudaMalloc(&cache.input, input_bytes);
    cudaMalloc(&cache.output, input_bytes);
    cache.input_cap = total;
  }
  if (cache.weight_cap < hidden_dim) {
    if (cache.weight) cudaFree(cache.weight);
    cudaMalloc(&cache.weight, weight_bytes);
    cache.weight_cap = hidden_dim;
  }

  T *d_input = cache.input;
  T *d_weight = cache.weight;
  T *d_output = cache.output;
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // 输出端锁页，避免 D2H 退化成低带宽的页表拷贝；
  // 输入端每次 pin/unpin 的开销比收益更大，因此保持普通页内存。
  bool output_registered =
      cudaHostRegister(h_output.data(), input_bytes,
                       cudaHostRegisterDefault) == cudaSuccess;
  if (!output_registered) cudaGetLastError();

  cudaMemcpyAsync(d_input, h_input.data(), input_bytes,
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_weight, h_weight.data(), weight_bytes,
                  cudaMemcpyHostToDevice, stream);

  const int D = static_cast<int>(hidden_dim);
  bool launched = false;

  // float4 要求 hidden_dim 是 4 * BLOCK_SIZE 的整数倍
  if constexpr (std::is_same<T, float>::value) {
    if (hidden_dim % (4 * BLOCK_SIZE) == 0) {
      const int items4 = D / (4 * BLOCK_SIZE);
      switch (items4) {
        case 1:  launch_cached_float4<1>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
        case 2:  launch_cached_float4<2>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
        case 4:  launch_cached_float4<4>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
        case 8:  launch_cached_float4<8>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
        case 16: launch_cached_float4<16>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
        default: break;
      }
    }
  }

  // 其他情况回退到标量寄存器缓存版本
  if (!launched && hidden_dim % BLOCK_SIZE == 0) {
    const int items = D / BLOCK_SIZE;
    switch (items) {
      case 1:  launch_cached<T, 1>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
      case 2:  launch_cached<T, 2>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
      case 4:  launch_cached<T, 4>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
      case 8:  launch_cached<T, 8>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
      case 16: launch_cached<T, 16>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
      case 32: launch_cached<T, 32>(d_input, d_weight, d_output, rows, D, eps, stream); launched = true; break;
      default: break;
    }
  }

  if (!launched) {
    rmsnorm_naive<T><<<static_cast<int>(rows), BLOCK_SIZE, 0, stream>>>(
        d_input, d_weight, d_output, static_cast<int>(rows), D, eps);
  }

  cudaMemcpyAsync(h_output.data(), d_output, input_bytes,
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (output_registered) cudaHostUnregister(h_output.data());
  cudaStreamDestroy(stream);
}

template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  flashAttentionImpl<T>(h_q, h_k, h_v, h_o, batch_size, target_seq_len,
                        src_seq_len, query_heads, kv_heads, head_dim,
                        is_causal);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float> &, const std::vector<float> &,
  std::vector<float> &, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half> &, const std::vector<half> &,
  std::vector<half> &, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float> &,
  const std::vector<float> &, const std::vector<float> &, std::vector<float> &,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half> &,
  const std::vector<half> &, const std::vector<half> &, std::vector<half> &,
  int, int, int, int, int, int, bool);
