#include "kernels.cu"
#include <chrono>
#include <cstdio>

int main() {
  const int rows = 4096, D = 4096;
  const float eps = 1e-5f;
  const int iters = 50;

  std::vector<float> h_in((size_t)rows * D), h_w(D), h_out((size_t)rows * D, 0.f);
  for (size_t i = 0; i < h_in.size(); ++i)
    h_in[i] = (float)((int)(i % 131) - 65) * 0.01f;
  for (int i = 0; i < D; ++i) h_w[i] = 1.0f + (i % 7) * 0.01f;

  float *d_in, *d_w, *d_out;
  cudaMalloc(&d_in, h_in.size() * sizeof(float));
  cudaMalloc(&d_w, D * sizeof(float));
  cudaMalloc(&d_out, h_out.size() * sizeof(float));
  cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_w, h_w.data(), D * sizeof(float), cudaMemcpyHostToDevice);

  auto run_naive = [&]() {
    rmsnorm_naive<float><<<rows, BLOCK_SIZE>>>(d_in, d_w, d_out, rows, D, eps);
  };
  auto run_cached = [&]() {
    rmsnorm_cached<float, 16><<<rows, BLOCK_SIZE>>>(d_in, d_w, d_out, D, eps);
  };
  auto run_float4 = [&]() {
    rmsnorm_cached_float4<4><<<rows, BLOCK_SIZE>>>(d_in, d_w, d_out, D, eps);
  };
  auto time_kernel = [&](auto launch) {
    for (int i = 0; i < 5; ++i) launch();
    cudaDeviceSynchronize();
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return ms / iters;
  };

  const double one_pass_bytes = (double)rows * D * sizeof(float);
  double naive_ms = time_kernel(run_naive);
  double cached_ms = time_kernel(run_cached);
  double float4_ms = time_kernel(run_float4);
  printf("float4_kernel: %.3f ms/iter, %.1f GB/s (2 passes)\n",
         float4_ms, one_pass_bytes * 2.0 / (float4_ms * 1e6));
  printf("naive_kernel : %.3f ms/iter, %.1f GB/s (3 passes)\n",
         naive_ms, one_pass_bytes * 3.0 / (naive_ms * 1e6));
  printf("cached_kernel: %.3f ms/iter, %.1f GB/s (2 passes)\n",
         cached_ms, one_pass_bytes * 2.0 / (cached_ms * 1e6));

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 5; ++i) rmsNorm<float>(h_in, h_w, h_out, rows, D, eps);
  auto t1 = std::chrono::high_resolution_clock::now();
  double hms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 5.0;
  printf("host_api_e2e : %.3f ms/call\n", hms);
  printf("out[0]=%.4f out[mid]=%.4f\n", h_out[0], h_out[h_out.size() / 2]);

  cudaFree(d_in); cudaFree(d_w); cudaFree(d_out);
  return 0;
}
