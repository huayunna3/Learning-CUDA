#pragma once

#include "flash_attention_common.cuh"

template <typename T>
__global__ void flash_attention_v1_kernel(
    const T *__restrict__ q, const T *__restrict__ k,
    const T *__restrict__ v, T *__restrict__ o,
    int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal, float scale) {
  const int t = blockIdx.x;
  const int hq = blockIdx.y;
  const int b = blockIdx.z;
  const int group_size = query_heads / kv_heads;
  const int hkv = hq / group_size;
  const int tid = threadIdx.x;

  extern __shared__ float smem[];
  float *q_s = smem;
  float *score_s = q_s + head_dim;
  float *red = score_s + src_seq_len;

  if (tid < head_dim) {
    const size_t offset =
        (((size_t)b * target_seq_len + t) * query_heads + hq) *
            head_dim + tid;
    q_s[tid] = to_float(q[offset]);
  }
  __syncthreads();

  const int key_end = is_causal
      ? min(src_seq_len, t + 1)
      : src_seq_len;

  float row_max;
  float row_sum;
  if (tid == 0) {
    row_max = -FLT_MAX;
    for (int j = 0; j < key_end; ++j) {
      const size_t offset =
          (((size_t)b * src_seq_len + j) * kv_heads + hkv) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot = fmaf(q_s[d], to_float(k[offset + d]), dot);
      }
      row_max = fmaxf(row_max, dot * scale);
    }
    red[0] = row_max;
  }
  __syncthreads();
  row_max = red[0];
  if (tid == 0) {
    row_sum = 0.0f;
    for (int j = 0; j < key_end; ++j) {
      const size_t offset =
          (((size_t)b * src_seq_len + j) * kv_heads + hkv) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot = fmaf(q_s[d], to_float(k[offset + d]), dot);
      }
      float exp_scale;
      const float exp_value = expf_parts(dot * scale - row_max, exp_scale);
      const float p = exp_value * exp_scale;
      score_s[j] = p;
      row_sum = __fmaf_rn(exp_value, exp_scale, row_sum);
    }
    red[0] = row_sum;
  }
  __syncthreads();
  row_sum = red[0];
  __syncthreads();
  row_sum = red[0];
  const float inv_row_sum = 1.0f / row_sum;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < key_end; ++j) {
      const size_t offset =
          (((size_t)b * src_seq_len + j) * kv_heads + hkv) *
              head_dim + d;
      acc = fmaf(score_s[j] * inv_row_sum, to_float(v[offset]), acc);
    }
    const size_t out_offset =
        (((size_t)b * target_seq_len + t) * query_heads + hq) *
            head_dim + d;
    from_float(o[out_offset], acc);
  }
}

template <typename T>
static void launch_flash_v1(
    const T *q, const T *k, const T *v, T *o,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, cudaStream_t stream) {
  constexpr int threads = 256;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const size_t smem =
      (head_dim + src_seq_len + threads) * sizeof(float);
  dim3 grid(target_seq_len, query_heads, batch_size);
  flash_attention_v1_kernel<T><<<grid, threads, smem, stream>>>(
      q, k, v, o, target_seq_len, src_seq_len, query_heads,
      kv_heads, head_dim, is_causal, scale);
}

