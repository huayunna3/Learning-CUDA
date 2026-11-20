#pragma once

#include "flash_attention_common.cuh"

template <typename T, int BQ, int BK>
__global__ __launch_bounds__(BQ * 32) void flash_attention_v3_kernel(
    const T *__restrict__ q, const T *__restrict__ k,
    const T *__restrict__ v, T *__restrict__ o,
    int target_seq_len, int src_seq_len, int query_heads,
    int kv_heads, int head_dim, bool is_causal, float scale) {
  constexpr int kThreads = BQ * 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int q_idx = blockIdx.x * BQ + warp;
  const int hq = blockIdx.y;
  const int b = blockIdx.z;
  const int group_size = query_heads / kv_heads;
  const int hkv = hq / group_size;
  const int stride = head_dim + 1;

  extern __shared__ float smem[];
  float *q_s = smem;
  float *k_s = q_s + BQ * head_dim;
  float *v_s = k_s + BK * stride;
  float *p_val_s = v_s + BK * stride;
  float *p_scale_s = p_val_s + BQ * BK;

  for (int idx = threadIdx.x; idx < BQ * head_dim;
       idx += kThreads) {
    const int row = idx / head_dim;
    const int d = idx - row * head_dim;
    const int row_idx = blockIdx.x * BQ + row;
    q_s[idx] = row_idx < target_seq_len
        ? to_float(q[(((size_t)b * target_seq_len + row_idx) *
                          query_heads + hq) * head_dim + d])
        : 0.0f;
  }
  __syncthreads();

  const int q_block_end = min(target_seq_len, blockIdx.x * BQ + BQ);
  const int block_key_end = is_causal
      ? min(src_seq_len, q_block_end)
      : src_seq_len;

  float row_max = -FLT_MAX;
  float row_sum = 0.0f;
  float out0 = 0.0f;
  float out1 = 0.0f;
  if (q_idx < target_seq_len) {
    const int row_key_end = is_causal
        ? min(src_seq_len, q_idx + 1)
        : src_seq_len;
    float local_max = -FLT_MAX;
    for (int j = lane; j < row_key_end; j += 32) {
      const size_t offset =
          (((size_t)b * src_seq_len + j) * kv_heads + hkv) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot = fmaf(q_s[warp * head_dim + d], to_float(k[offset + d]), dot);
      }
      local_max = fmaxf(local_max, dot * scale);
    }
    for (int delta = 16; delta > 0; delta >>= 1) {
      local_max = fmaxf(local_max,
                        __shfl_down_sync(0xffffffffu, local_max, delta));
    }
    row_max = __shfl_sync(0xffffffffu, local_max, 0);
  }

  for (int key_base = 0; key_base < block_key_end; key_base += BK) {
    const int tile_end = min(key_base + BK, block_key_end);
    for (int idx = threadIdx.x; idx < BK * head_dim;
         idx += kThreads) {
      const int j = idx / head_dim;
      const int d = idx - j * head_dim;
      if (key_base + j < tile_end) {
        const size_t offset =
            (((size_t)b * src_seq_len + key_base + j) * kv_heads + hkv) *
                head_dim + d;
        k_s[j * stride + d] = to_float(k[offset]);
        v_s[j * stride + d] = to_float(v[offset]);
      }
    }
    __syncthreads();

    if (q_idx < target_seq_len) {
      const int row_key_end = is_causal
          ? min(tile_end, q_idx + 1)
          : tile_end;
        const int count = row_key_end - key_base;
        if (count > 0) {
          for (int jj = lane; jj < count; jj += 32) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
              dot = fmaf(q_s[warp * head_dim + d],
                         k_s[jj * stride + d], dot);
            }
            float exp_scale;
            const float exp_value =
                expf_parts(dot * scale - row_max, exp_scale);
            p_val_s[warp * BK + jj] = exp_value;
            p_scale_s[warp * BK + jj] = exp_scale;
          }
          __syncwarp();
          if (lane == 0) {
            for (int jj = 0; jj < count; ++jj) {
              row_sum = __fmaf_rn(p_val_s[warp * BK + jj],
                                  p_scale_s[warp * BK + jj], row_sum);
            }
          }
          __syncwarp();
          for (int jj = 0; jj < count; ++jj) {
            const float p =
                p_val_s[warp * BK + jj] * p_scale_s[warp * BK + jj];
            if (lane < head_dim) {
              out0 = fmaf(p, v_s[jj * stride + lane], out0);
            }
            if (lane + 32 < head_dim) {
              out1 = fmaf(p, v_s[jj * stride + lane + 32], out1);
            }
          }
        }
      }
    __syncthreads();
  }

  if (q_idx < target_seq_len) {
    const float inv_sum = 1.0f / __shfl_sync(0xffffffffu, row_sum, 0);
    const size_t out_base =
        (((size_t)b * target_seq_len + q_idx) * query_heads + hq) *
        head_dim;
    if (lane < head_dim) {
      from_float(o[out_base + lane], out0 * inv_sum);
    }
    if (lane + 32 < head_dim) {
      from_float(o[out_base + lane + 32], out1 * inv_sum);
    }
  }
}

template <typename T>
static void launch_flash_v3(
    const T *q, const T *k, const T *v, T *o,
    int batch_size, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    bool is_causal, cudaStream_t stream) {
  constexpr int BQ = 4;
  constexpr int BK = 32;
  constexpr int threads = BQ * 32;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  const size_t smem =
      (BQ * head_dim + 2 * BK * (head_dim + 1) + 2 * BQ * BK) * sizeof(float);
  const dim3 grid((target_seq_len + BQ - 1) / BQ,
                  query_heads, batch_size);
  flash_attention_v3_kernel<T, BQ, BK><<<grid, threads, smem, stream>>>(
      q, k, v, o, target_seq_len, src_seq_len, query_heads,
      kv_heads, head_dim, is_causal, scale);
}
