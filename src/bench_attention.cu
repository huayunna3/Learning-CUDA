#include "kernels.cu"
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  const int B = 4, T = 512, S = 2048, Hq = 64, Hkv = 64, D = 32;
  const bool causal = true;
  const int iters = argc > 1 ? atoi(argv[1]) : 3;
  std::vector<float> q((size_t)B * T * Hq * D);
  std::vector<float> k((size_t)B * S * Hkv * D);
  std::vector<float> v(k.size());
  std::vector<float> o(q.size());
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = (float)((int)(i % 257) - 128) * 0.00390625f;
  }
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = (float)((int)(i % 251) - 125) * 0.00390625f;
    v[i] = (float)((int)(i % 239) - 119) * 0.00390625f;
  }

  flashAttentionImpl(q, k, v, o, B, T, S, Hq, Hkv, D, causal);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    flashAttentionImpl(q, k, v, o, B, T, S, Hq, Hkv, D, causal);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  printf("shape=B%d T%d S%d Hq%d Hkv%d D%d causal=%d iters=%d avg_ms=%.6f checksum=%.6f\n",
         B, T, S, Hq, Hkv, D, (int)causal, iters,
         ms / (float)iters, (double)o[(o.size() / 2)]);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
