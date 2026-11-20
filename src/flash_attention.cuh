#pragma once

#include "flash_attention_common.cuh"

// 1 = correctness baseline, 2 = online softmax, 3 = tiled final version.
#ifndef FLASH_ATTENTION_VERSION
#define FLASH_ATTENTION_VERSION 1
#endif

#if FLASH_ATTENTION_VERSION == 1
#include "flash_attention_v1.cuh"
#elif FLASH_ATTENTION_VERSION == 2
#include "flash_attention_v2.cuh"
#elif FLASH_ATTENTION_VERSION == 3
#include "flash_attention_v3.cuh"
#else
#error "FLASH_ATTENTION_VERSION must be 1, 2, or 3"
#endif

template <typename T>
void flashAttentionImpl(const std::vector<T> &h_q,
                        const std::vector<T> &h_k,
                        const std::vector<T> &h_v,
                        std::vector<T> &h_o,
                        int batch_size, int target_seq_len,
                        int src_seq_len, int query_heads,
                        int kv_heads, int head_dim, bool is_causal) {
  if (batch_size <= 0 || target_seq_len <= 0 || src_seq_len <= 0 ||
      query_heads <= 0 || kv_heads <= 0 || head_dim <= 0) {
    return;
  }

  const size_t q_elems =
      (size_t)batch_size * target_seq_len * query_heads * head_dim;
  const size_t kv_elems =
      (size_t)batch_size * src_seq_len * kv_heads * head_dim;
  static thread_local FlashDeviceCache<T> cache;
  if (cache.q_cap < q_elems) {
    if (cache.q) cudaFree(cache.q);
    if (cache.o) cudaFree(cache.o);
    cudaMalloc(&cache.q, q_elems * sizeof(T));
    cudaMalloc(&cache.o, q_elems * sizeof(T));
    cache.q_cap = q_elems;
    cache.o_cap = q_elems;
  }
  if (cache.k_cap < kv_elems) {
    if (cache.k) cudaFree(cache.k);
    if (cache.v) cudaFree(cache.v);
    cudaMalloc(&cache.k, kv_elems * sizeof(T));
    cudaMalloc(&cache.v, kv_elems * sizeof(T));
    cache.k_cap = kv_elems;
    cache.v_cap = kv_elems;
  }

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  cudaMemcpyAsync(cache.q, h_q.data(), q_elems * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(cache.k, h_k.data(), kv_elems * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(cache.v, h_v.data(), kv_elems * sizeof(T),
                  cudaMemcpyHostToDevice, stream);

#if FLASH_ATTENTION_VERSION == 1
  launch_flash_v1<T>(cache.q, cache.k, cache.v, cache.o,
                     batch_size, target_seq_len, src_seq_len,
                     query_heads, kv_heads, head_dim,
                     is_causal, stream);
#elif FLASH_ATTENTION_VERSION == 2
  launch_flash_v2<T>(cache.q, cache.k, cache.v, cache.o,
                     batch_size, target_seq_len, src_seq_len,
                     query_heads, kv_heads, head_dim,
                     is_causal, stream);
#else
  launch_flash_v3<T>(cache.q, cache.k, cache.v, cache.o,
                     batch_size, target_seq_len, src_seq_len,
                     query_heads, kv_heads, head_dim,
                     is_causal, stream);
#endif

  cudaMemcpyAsync(h_o.data(), cache.o, q_elems * sizeof(T),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}
